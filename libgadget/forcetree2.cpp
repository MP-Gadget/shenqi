#include <mpi.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#include "domain.h"
#include "forcetree2.h"
#include "walltime.h"
#include "checkpoint.h"
#include "slotsmanager.h"
#include "partmanager.h"
#include "utils/endrun.h"
#include "utils/system.h"
#include "utils/mymalloc.h"

/* Get the subnode for a given particle and parent node.
 * This splits a parent node into 8 subregions depending on the particle position.
 * node is the parent node to split, p_i is the index of the particle we
 * are currently inserting
 * Returns a value between 0 and 7.
 * */
static inline int get_subnode(const struct NODE * node, const double Pos[3])
{
    /*Loop is unrolled to help out the compiler,which normally only manages it at -O3*/
     return (Pos[0] > node->center[0]) +
            ((Pos[1] > node->center[1]) << 1) +
            ((Pos[2] > node->center[2]) << 2);
}

/*Check whether a particle is inside the volume covered by a node,
 * by checking whether each dimension is close enough to center (L1 metric).
 * 'Nugget' is the allowed floating point error.*/
static inline int inside_node(const struct NODE * node, const double Pos[3], const double nugget)
{
    /*One can also use a loop, but the compiler unrolls it only at -O3,
     *so this is a little faster*/
    int inside =
        (fabs(2*(Pos[0] - node->center[0])) <= node->len + nugget) *
        (fabs(2*(Pos[1] - node->center[1])) <= node->len + nugget) *
        (fabs(2*(Pos[2] - node->center[2])) <= node->len + nugget);
    return inside;
}

/*Initialise an internal node at nfreep. The parent is assumed to be locked, and
 * we have assured that nothing else will change nfreep while we are here.*/
void init_internal_node(struct NODE *nfreep, struct NODE *parent, int subnode)
{
    int j;
    const MyFloat lenhalf = 0.25 * parent->len;
    nfreep->len = 0.5 * parent->len;
    nfreep->sibling = -10;
    nfreep->father = -10;
    nfreep->f.TopLevel = 0;
    nfreep->f.InternalTopLevel = 0;
    nfreep->f.DependsOnLocalMass = 0;
    nfreep->f.ChildType = PARTICLE_NODE_TYPE;
    nfreep->f.unused = 0;

    for(j = 0; j < 3; j++) {
        /* Detect which quadrant we are in by testing the bits of subnode:
         * if (subnode & [1,2,4]) is true we add lenhalf, otherwise subtract lenhalf*/
        const int sign = (subnode & (1 << j)) ? 1 : -1;
        nfreep->center[j] = parent->center[j] + sign*lenhalf;
    }
    for(j = 0; j < NMAXCHILD; j++)
        nfreep->s.suns[j] = -1;
    nfreep->s.noccupied = 0;
    memset(&(nfreep->mom.cofm),0,3*sizeof(MyFloat));
    nfreep->mom.mass = 0;
    nfreep->mom.hmax = 0;
}

/* Size of the free Node thread cache.
 * 12 8-node rows (works out at 8kB) was found
 * to be optimal for an Intel skylake and
 * an AMD Zen2 with 12 threads.*/
#define NODECACHE_SIZE (8*12)

/*Structure containing thread-local parameters of the tree build*/
struct NodeCache {
    int nnext_thread;
    int nrem_thread;
};

/*Get a pointer to memory for 8 free nodes, from our node cache. */
int get_freenode(int64_t * nnext, struct NodeCache *nc)
{
    /*Get memory for an extra node from our cache.*/
    if(nc->nrem_thread < 8) {
        nc->nnext_thread = atomic_fetch_and_add_64(nnext, NODECACHE_SIZE);
        nc->nrem_thread = NODECACHE_SIZE;
    }
    const int ninsert = nc->nnext_thread;
    nc->nnext_thread += 8;
    nc->nrem_thread -= 8;
    return ninsert;
}

/* Create a new layer of nodes beneath the current node, and place the particle.
 * Must have node lock.*/
template <typename DerivedTree>
int
ForceTree<DerivedTree>::create_new_node_layer(int firstparent, int p_toplace, int64_t *nnext, struct NodeCache *nc)
{
    /* This is so we can defer changing
     * the type of the existing node until the end.*/
    int parent = firstparent;

    do {
        int i;
        struct NODE *nprnt = &Nodes[parent];

        /* Braces to scope oldsuns and newsuns*/
        {
        int newsuns[NMAXCHILD];

        int * oldsuns = nprnt->s.suns;

        /*We have two particles here, so create a new child node to store them both.*/
        /* if we are here the node must be large enough, thus contain exactly one child. */
        /* The parent is already a leaf, need to split */
        /* Get memory for 8 extra nodes from our cache.*/
        newsuns[0] = get_freenode(nnext, nc);
        /*If we already have too many nodes, exit loop.*/
        if(nc->nnext_thread >= lastnode) {
            /* This means that we have > NMAXCHILD particles in the same place,
            * which usually indicates a bug in the particle evolution. Print some helpful debug information.*/
            message(1, "Failed placing %d at %g %g %g, type %d, ID %ld. Others were %d (%g %g %g, t %d ID %ld) and %d (%g %g %g, t %d ID %ld).\n",
                p_toplace, Part[p_toplace].Pos[0], Part[p_toplace].Pos[1], Part[p_toplace].Pos[2], Part[p_toplace].Type, Part[p_toplace].ID,
                oldsuns[0], Part[oldsuns[0]].Pos[0], Part[oldsuns[0]].Pos[1], Part[oldsuns[0]].Pos[2], Part[oldsuns[0]].Type, Part[oldsuns[0]].ID,
                oldsuns[1], Part[oldsuns[1]].Pos[0], Part[oldsuns[1]].Pos[1], Part[oldsuns[1]].Pos[2], Part[oldsuns[1]].Type, Part[oldsuns[1]].ID
            );
            nc->nnext_thread = lastnode + 10 * NODECACHE_SIZE;
            /* If this is not the first layer created,
                * we need to mark the overall parent as a node node
                * while marking this one as a particle node */
            if(firstparent != parent)
            {
                nprnt->f.ChildType = PARTICLE_NODE_TYPE;
                nprnt->s.noccupied = NMAXCHILD;
                Nodes[firstparent].f.ChildType = NODE_NODE_TYPE;
                Nodes[firstparent].s.noccupied = NODEFULL;
            }
            return 1;
        }
        for(i=0; i<8; i++) {
            newsuns[i] = newsuns[0] + i;
            struct NODE *nfreep = &Nodes[newsuns[i]];
            /* We create a new leaf node.*/
            init_internal_node(nfreep, nprnt, i);
            /*Set father of new node*/
            nfreep->father = parent;
        }
        /*Initialize the remaining entries to empty*/
        for(i=8; i<NMAXCHILD;i++)
            newsuns[i] = -1;

        for(i=0; i < NMAXCHILD; i++) {
            /* Re-attach each particle to the appropriate new leaf.
            * Notice that since we have NMAXCHILD slots on each child and NMAXCHILD particles,
            * we will always have a free slot. */
            int subnode = get_subnode(nprnt, Part[oldsuns[i]].Pos);
            int child = newsuns[subnode];
            struct NODE * nchild = &Nodes[child];
            static_cast<DerivedTree*>(this)->modify_internal_node(child, nchild->s.noccupied, oldsuns[i]);
            nchild->s.noccupied++;
        }
        /* Copy the new node array into the node*/
        memcpy(nprnt->s.suns, newsuns, NMAXCHILD * sizeof(int));
        } /* After this brace oldsuns and newsuns are invalid*/

        /* Set sibling for the new rank. Since empty at this point, point onwards.*/
        for(i=0; i<7; i++) {
            int child = nprnt->s.suns[i];
            struct NODE * nchild = &Nodes[child];
            nchild->sibling = nprnt->s.suns[i+1];
        }
        /* Final child needs special handling: set to the parent's sibling.*/
        Nodes[nprnt->s.suns[7]].sibling = nprnt->sibling;
        /* Zero the momenta for the parent*/
        memset(&nprnt->mom, 0, sizeof(nprnt->mom));

        /* Now try again to add the new particle*/
        int subnode = get_subnode(nprnt, Part[p_toplace].Pos);
        int child = nprnt->s.suns[subnode];
        struct NODE * nchild = &Nodes[child];
        if(nchild->s.noccupied < NMAXCHILD) {
            static_cast<DerivedTree*>(this)->modify_internal_node(child, nchild->s.noccupied, p_toplace);
            nchild->s.noccupied++;
            break;
        }
        /* The attached particles are already within one subnode of the new node.
         * Iterate, creating a new layer beneath.*/
        else {
            /* The current child is going to have new nodes created beneath it,
             * so mark it a Node-containing node. It cannot be accessed until
             * we mark the top-level parent, so no need for atomics.*/
            Nodes[child].f.ChildType = NODE_NODE_TYPE;
            Nodes[child].s.noccupied = NODEFULL;
            parent = child;
        }
    } while(1);

    /* A new node is created. Mark the (original) parent as an internal node with node children.
     * This goes last so that we don't access the child before it is constructed.*/
    Nodes[firstparent].f.ChildType = NODE_NODE_TYPE;
    Nodes[firstparent].s.noccupied = NODEFULL;
    return 0;
}

/* Add a particle to the tree, extending the tree as necessary. Locking is done,
 * so may be called from a threaded context*/
template <typename DerivedTree>
int ForceTree<DerivedTree>::add_particle_to_tree(int i, int cur_start, struct NodeCache *nc, int64_t* nnext)
{
    int child, nocc;
    int cur = cur_start;
    /*Walk the main tree until we get something that isn't an internal node.*/
    do
    {
        /*No lock needed: if we have an internal node here it will be stable*/
        nocc = Nodes[cur].s.noccupied;

        /* This node still has space for a particle (or needs conversion)*/
        if(nocc < NODEFULL)
            break;

        /* This node has child subnodes: find them.*/
        int subnode = get_subnode(&Nodes[cur], Part[i].Pos);
        /*No lock needed: if we have an internal node here it will be stable*/
        child = Nodes[cur].s.suns[subnode];

        if(child > lastnode || child < 0)
            endrun(1,"Corruption in tree build: N[%d].[%d] = %d > lastnode (%ld)\n",cur, subnode, child, lastnode);
        cur = child;
    }
    while(child >= 0);

    /* We have a guaranteed spot.*/
    nocc = Nodes[cur].s.noccupied;
    Nodes[cur].s.noccupied++;

    /* Now we have something that isn't an internal node. We can place the particle! */
    if(nocc < NMAXCHILD)
        static_cast<DerivedTree*>(this)->modify_internal_node(cur, nocc, i);
    /* In this case we need to create a new layer of nodes beneath this one*/
    else if(nocc < NODEFULL) {
        if(create_new_node_layer(cur, i, nnext, nc))
            return -1;
    } else
        endrun(2, "Tried to convert already converted node %d with nocc = %d\n", cur, nocc);
    return cur;
}

/* Merge two partial trees together. Trees are walked simultaneously.
 * A merge is done when a particle node is encountered in one of the side trees.
 * The merge rule is that the node node is attached to the
 * left-most old parent and the particles are re-attached to the node node*/
template <typename DerivedTree>
int
ForceTree<DerivedTree>::merge_partial_force_trees(int left, int right, struct NodeCache * nc, int64_t * nnext)
{
    int this_left = left;
    int this_right = right;
    const int left_end = Nodes[left].sibling;
    const int right_end = Nodes[right].sibling;
//     message(5, "Ends: %d %d\n", left_end, right_end);
    while(this_left != left_end && this_right != right_end)
    {
        if(this_left < 0 || this_right < 0)
            endrun(10, "Encountered invalid node: %d %d < 0\n", this_left, this_right);
        struct NODE * nleft = &Nodes[this_left];
        struct NODE * nright = &Nodes[this_right];
        if(nc->nnext_thread >= lastnode)
            return 1;
#ifdef DEBUG
        /* Stop when we reach another topnode*/
        if((nleft->f.TopLevel && this_left != left) || (nright->f.TopLevel && this_right != right))
            endrun(6, "Encountered another topnode: left %d == right %d! type %d\n", this_left, this_right, nleft->f.ChildType);
        if(this_left == this_right)
            endrun(6, "Odd: left %d == right %d! type %d\n", this_left, this_right, nleft->f.ChildType);
//         message(1, "left %d right %d\n", this_left, this_right);
        /* Trees should be synced*/
        if(fabs(nleft->len / nright->len-1) > 1e-6)
            endrun(6, "Merge unsynced trees: %d %d len %g %g\n", this_left, this_right, nleft->len, nright->len);
#endif
        /* Two node nodes: keep walking down*/
        if(nleft->f.ChildType == NODE_NODE_TYPE && nright->f.ChildType == NODE_NODE_TYPE) {
            if(Nodes[nleft->s.suns[0]].father < 0 || Nodes[nright->s.suns[0]].father < 0)
                endrun(7, "Walking to nodes (%d %d) from (%d %d) fathers (%d %d)\n",
                       nleft->s.suns[0], nright->s.suns[0], this_left, this_right, Nodes[nleft->s.suns[0]].father, Nodes[nright->s.suns[0]].father);
            this_left = nleft->s.suns[0];
            this_right = nright->s.suns[0];
            continue;
        }
        /* If the right node has particles, add them to the left node, go to sibling on right and left.*/
        else if(nright->f.ChildType == PARTICLE_NODE_TYPE) {
            int i;
            for(i = 0; i < nright->s.noccupied; i++) {
                if(nright->s.suns[i] >= 0)
                    endrun(8, "Bad child %d of %d in %d\n", i, nright->s.suns[i], this_right);
                if(add_particle_to_tree(nright->s.suns[i], this_left, nc, nnext) < 0)
                    return 1;
            }
            /* Make sure that nodes which have
             * this_right as a sibling (there will
             * be a max of one, as it is a particle node
             * with no children) point to the replacement
             * on the left*/
            /* This condition is checking for the root node, which has no siblings*/
            if(this_right > right) {
                /* Find the father, then the next child*/
                struct NODE * fat = &Nodes[nright->father];
                /* Find the position of this child in the father*/
                int sunloc = 0;
                for(i = 0; i < 8; i++)
                {
                    if(fat->s.suns[i] == this_right) {
                        sunloc = i;
                        break;
                    }
                }
                /* Change the sibling of the child next to this one*/
                if(sunloc > 0) {
                    if(Nodes[fat->s.suns[i-1]].sibling == this_right)
                        Nodes[fat->s.suns[i-1]].sibling = this_left;
                }
            }
            /* Mark the right node as now invalid*/
            nright->father = -5;
            /* Now go to sibling*/
            this_left = nleft->sibling;
            this_right = nright->sibling;
            continue;
        }
        /* If the left node has particles, add them to the right node,
         * then copy the right node over the left node and go to (old) sibling on right and left.*/
        else if(nleft->f.ChildType == PARTICLE_NODE_TYPE && nright->f.ChildType == NODE_NODE_TYPE) {
            /* Add the left particles to the right*/
            int i;
            for(i = 0; i < nleft->s.noccupied; i++) {
                if(nleft->s.suns[i] >= 0)
                    endrun(8, "Bad child %d of %d in left %d\n", i, nleft->s.suns[i], this_left);
                if(add_particle_to_tree(nleft->s.suns[i], this_right, nc, nnext) < 0)
                    return 1;
            }
            /* Copy the right node over the left*/
            memmove(&nleft->s, &nright->s, sizeof(nleft->s));
            nleft->f.ChildType = NODE_NODE_TYPE;
            /* Zero the momenta for the parent*/
            memset(&nleft->mom, 0, sizeof(nleft->mom));
            /* Reset children to the new parent:
             * this assumes nright is a NODE NODE*/
            for(i = 0; i < 8; i++) {
                int child = nleft->s.suns[i];
                Nodes[child].father = this_left;
            }
            /* Make sure final child points to the parent's sibling.*/
#ifdef DEBUG
            int oldsib = Nodes[nleft->s.suns[7]].sibling;
#endif
            /* Walk downwards making sure all the children point to the new sibling.
             * Note also changes last particle node child. */
            int nn = this_left;
            while(Nodes[nn].f.ChildType == NODE_NODE_TYPE) {
                nn = Nodes[nn].s.suns[7];
#ifdef DEBUG
                if(Nodes[nn].sibling != oldsib)
                    endrun(20, "Not the expected sibling %d != %d\n",Nodes[nn].sibling, oldsib);
#endif
                Nodes[nn].sibling = nleft->sibling;
            }
            /* Mark the right node as now invalid*/
            nright->father = -5;
            /* Next iteration is going to sibling*/
            this_left = nleft->sibling;
            this_right = nright->sibling;
            continue;
        }
        else
            endrun(6, "Nodes %d %d have unexpected type %d %d\n", this_left, this_right, nleft->f.ChildType, nright->f.ChildType);
    }
    return 0;
}

/*! Does initial creation of the nodes for the gravitational oct-tree.
 * mask is a bitfield: Only types whose bit is set are added.
 **/
template <typename DerivedTree>
void
ForceTree<DerivedTree>::force_tree_create_nodes(const ActiveParticles * act, DomainDecomp * ddecomp)
{
    int64_t nnext = numnodes;

    /* Set up thread-local copies of the topnodes to anchor the subtrees. */
    const int StartLeaf = ddecomp->Tasks[ThisTask].StartLeaf;
    const int EndLeaf = ddecomp->Tasks[ThisTask].EndLeaf;
    const int nthr = omp_get_max_threads();
    int * topnodes = ta_malloc("topnodes", int, (EndLeaf - StartLeaf) * nthr);
    /* Topnodes for each thread. For tid 0, just use the real tree. Saves copying the tree back.*/
    for(int j = 0; j < EndLeaf - StartLeaf; j++)
        topnodes[j] = ddecomp->TopLeaves[j + StartLeaf].treenode;
    /* Other threads need a copy*/
    for(int t = 1; t < nthr; t++) {
        for(int j = 0; j < EndLeaf - StartLeaf; j++) {
            /* Make a local copy*/
            topnodes[j + t * (EndLeaf - StartLeaf)] = nnext;
            memmove(&Nodes[nnext], &Nodes[topnodes[j]], sizeof(struct NODE));
            nnext++;
        }
    }

/*     double tstart = second(); */
    const int first_free = nnext;
    /* Increment nnext for the threads we are about to initialise.*/
    nnext += NODECACHE_SIZE * nthr;
    /* now we insert all particles */
    int numparticles=0;

    #pragma omp parallel
    {
        /* Local topnodes*/
        int tid = omp_get_thread_num();
        const int * const local_topnodes = topnodes + tid * (EndLeaf - StartLeaf);

        /* This implements a small thread-local free Node cache.
         * The cache ensures that Nodes from the same (or close) particles
         * are created close to each other on the Node list and thus
         * helps cache locality. I tried each thread getting a separate
         * part of the tree, and it wasted too much memory. */
        struct NodeCache nc;
        nc.nnext_thread = first_free + tid * NODECACHE_SIZE;
        nc.nrem_thread = NODECACHE_SIZE;

        /* Stores the last-seen node on this thread.
         * Since most particles are close to each other, this should save a number of tree walks.*/
        int this_acc = local_topnodes[0];
        // message(1, "Topnodes %d real %d\n", local_topnodes[0], topnodes[0]);

        /* The default schedule is static with a chunk 1/4 the total.
         * However, particles are sorted by type and then by peano order.
         * Since we need to merge trees, it is advantageous to have all particles
         * spatially close be processed by the same thread. This means that the threads should
         * process particles at a constant offset from the start of the type.
         * We do this with a static schedule. */
        int chnksz = PartManager->NumPart/nthr;
        if(SlotsManager->info[0].enabled && SlotsManager->info[0].size > 0)
            chnksz = SlotsManager->info[0].size/nthr;
        if(chnksz < 1000)
            chnksz = 1000;
        #pragma omp for schedule(static, chnksz) reduction(+: numparticles)
        for(int j = 0; j < act->NumActiveParticle; j++)
        {
            /*Can't break from openmp for*/
            if(nc.nnext_thread >= lastnode)
                continue;

            /* Pick the next particle from the active list if there is one*/
            const int i = act->ActiveParticle ? act->ActiveParticle[j] : j;

            /* Do not add types that do not have their mask bit set.*/
            if(!((1<<Part[i].Type) & mask)) {
                continue;
            }
            /* Do not add garbage/swallowed particles to the tree*/
            if(Part[i].IsGarbage || (Part[i].Swallowed && Part[i].Type==5))
                continue;

            if(Part[i].Mass <= 0)
                endrun(12, "Zero mass particle %d m %g type %d id %ld pos %g %g %g\n", i, Part[i].Mass, Part[i].Type, Part[i].ID, Part[i].Pos[0], Part[i].Pos[1], Part[i].Pos[2]);
            /*First find the Node for the TopLeaf */
            int cur;
            if(inside_node(&Nodes[this_acc], Part[i].Pos, 0)) {
                cur = this_acc;
            } else {
                /* Get the topnode to which a particle belongs. Each local tree
                 * has a local set of treenodes copying the global topnodes, except tid 0
                 * which has the real topnodes.*/
                const int topleaf = Part[i].TopLeaf;
                if(topleaf < StartLeaf || topleaf >= EndLeaf)
                    endrun(5, "Bad topleaf %d start %d end %d type %d ID %ld\n", topleaf, StartLeaf, EndLeaf, Part[i].Type, Part[i].ID);
                //int treenode = ddecomp->TopLeaves[topleaf].treenode;
                cur = local_topnodes[topleaf - StartLeaf];
#ifdef DEBUG
                if(!inside_node(&Nodes[cur], Part[i].Pos, 1e-7))
                    endrun(13, "Particle %d at %g %g %g not inside topnode %d center %g %g %g len %g\n", i,
                        Part[i].Pos[0], Part[i].Pos[1], Part[i].Pos[2], cur, Nodes[cur].center[0], Nodes[cur].center[1], Nodes[cur].center[2], Nodes[cur].len);
#endif
            }
            numparticles++;
            this_acc = add_particle_to_tree(i, cur, &nc, &nnext);
        }
        /* The implicit omp-barrier is important here!*/
/*         double tend = second(); */
/*         message(0, "Initial insertion: %.3g ms. First node %d\n", (tend - tstart)*1000, local_topnodes[0]); */

        /* Merge each topnode separately, using a for loop.
         * This wastes threads if NTHREAD > NTOPNODES, but it
         * means only one merge is done per subtree and
         * it requires no locking.*/
        #pragma omp for schedule(static, 1)
        for(int j = 0; j < EndLeaf - StartLeaf; j++) {
            /* These are the addresses of the real topnodes*/
            const int target = topnodes[j];
            if(nc.nnext_thread >= lastnode)
                continue;
            for(int t = 1; t < nthr; t++) {
                const int righttop = topnodes[j + t * (EndLeaf - StartLeaf)];
//                  message(1, "tid = %d i = %d t = %d Merging %d to %d addresses are %lx - %lx end is %lx\n", omp_get_thread_num(), i, t, righttop, target, &Nodes[righttop], &Nodes[target], &Nodes[nnext]);
                if(merge_partial_force_trees(target, righttop, &nc, &nnext))
                    break;
            }
        }
    }
    NumParticles = numparticles;
    numnodes = nnext;
    ta_free(topnodes);
    return;
}

/*Get the sibling of a node, using the suns array. Only to be used in the tree build, before update_node_recursive is called.*/
static int
force_get_sibling(const int sib, const int j, const int * suns)
{
    /* check if we have a sibling on the same level */
    int jj;
    int nextsib = sib;
    for(jj = j + 1; jj < 8; jj++) {
        if(suns[jj] >= 0) {
            nextsib = suns[jj];
            break;
        }
    }
    return nextsib;
}

/* Set the center of mass of the current node*/
void
force_update_particle_node(struct NODE& curnode)
{
#ifdef DEBUG
    if(curnode.f.ChildType != PARTICLE_NODE_TYPE)
        endrun(3, "force_update_particle_node called on node of wrong type %d!\n", curnode.f.ChildType);
#endif
    int j;
    /*Set the center of mass moments*/
    const double mass = curnode.mom.mass;
    /* Be careful about empty nodes*/
    if(mass > 0) {
        for(j = 0; j < 3; j++)
            curnode.mom.cofm[j] /= mass;
    }
    else {
        for(j = 0; j < 3; j++)
            curnode.mom.cofm[j] = curnode.center[j];
    }
}

/*! this routine determines the multipole moments for a given internal node
 *  and all its subnodes using a recursive computation.  The result is
 *  stored in Nodes in the sequence of this tree-walk.
 *
 *  The function also computes the NextNode and sibling linked lists.
 *  The return value is the current tail of the NextNode linked list.
 *
 *  This function is called recursively using openmp tasks.
 *  We spawn a new task for a fixed number of levels of the tree.
 *
 */
/* Explicit instantiation of the CRTP base for each concrete tree, so that the
 * out-of-line members above are emitted in this translation unit.*/
template class ForceTree<ForceTreeHmax>;
template class ForceTree<ForceTreeMoments>;

int
ForceTreeMoments::force_update_node_recursive(const int no, const int sib, const int level)
{
#ifdef DEBUG
    if(Nodes[no].f.ChildType != NODE_NODE_TYPE)
        endrun(3, "force_update_node_recursive called on node %d of type %d != %d!\n", no, Nodes[no].f.ChildType, NODE_NODE_TYPE);
#endif
    int j;
    int * const suns = Nodes[no].s.suns;

    int childcnt = 0;
    /* Remove any empty children, moving the suns array around
     * so non-empty entries are contiguous at the beginning of the array.
     * This sharply reduces the size of the tree.
     * Also count the node children for thread balancing.*/
    int jj = 0;
    for(j=0; j < 8; j++, jj++) {
        /* Never remove empty top-level nodes so we don't
         * mess up the pseudo-data exchange.
         * This may happen for a pseudo particle host or, in very rare cases,
         * when one of the local domains is empty. */
        while(jj < 8 && !Nodes[suns[jj]].f.TopLevel &&
            Nodes[suns[jj]].f.ChildType == PARTICLE_NODE_TYPE &&
            Nodes[suns[jj]].s.noccupied == 0) {
                    jj++;
        }
        if(jj < 8)
            suns[j] = suns[jj];
        else
            suns[j] = -1;
        if(suns[j] >= 0 && Nodes[suns[j]].f.ChildType == NODE_NODE_TYPE)
            childcnt++;
    }

    /*First do the children*/
    for(j = 0; j < 8; j++)
    {
        const int p = suns[j];
        /*Empty slot*/
        if(p < 0)
            continue;
        const int nextsib = force_get_sibling(sib, j, suns);
        /* This is set in create_nodes but needed because we may remove empty nodes above.*/
        Nodes[p].sibling = nextsib;
        /* Nodes containing particles or pseudo-particles*/
        if(Nodes[p].f.ChildType == PARTICLE_NODE_TYPE)
            force_update_particle_node(Nodes[p]);
        if(Nodes[p].f.ChildType == NODE_NODE_TYPE) {
            /* Don't spawn a new task if we are deep enough that we already spawned a lot.*/
            if(childcnt > 1 && level < 512) {
                const int newlevel = level * childcnt;
                /* Firstprivate for const variables should be optimised out*/
                #pragma omp task default(none) firstprivate(nextsib, p, newlevel)
                force_update_node_recursive(p, nextsib, newlevel);
            }
            else
                force_update_node_recursive(p, nextsib, level);
        }
    }

    /*Make sure all child nodes are done*/
    #pragma omp taskwait

    /*Now we do the moments*/
    for(j = 0; j < 8; j++)
    {
        const int p = suns[j];
        if(p < 0)
            continue;
        Nodes[no].mom.mass += (Nodes[p].mom.mass);
        Nodes[no].mom.cofm[0] += (Nodes[p].mom.mass * Nodes[p].mom.cofm[0]);
        Nodes[no].mom.cofm[1] += (Nodes[p].mom.mass * Nodes[p].mom.cofm[1]);
        Nodes[no].mom.cofm[2] += (Nodes[p].mom.mass * Nodes[p].mom.cofm[2]);
        if(Nodes[p].mom.hmax > Nodes[no].mom.hmax)
            Nodes[no].mom.hmax = Nodes[p].mom.hmax;
    }

    /*Set the center of mass moments*/
    const double mass = Nodes[no].mom.mass;
    /* In principle all the children could be pseudo-particles*/
    if(mass > 0) {
        Nodes[no].mom.cofm[0] /= mass;
        Nodes[no].mom.cofm[1] /= mass;
        Nodes[no].mom.cofm[2] /= mass;
    }

    return -1;
}
