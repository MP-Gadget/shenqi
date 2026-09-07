#ifndef FORCETREE2_H
#define FORCETREE2_H

#include "types.h"
#include "domain.h"
#include "timestep.h"
#include "checkpoint.h"
#include "walltime.h"
#include "utils/mymalloc.h"
/*
 * Variables for Tree
 * ------------------
 */

/* Total allowed number of particle children for a node*/
#define NMAXCHILD 8
#define NODEFULL (1<<16)

/* Defines for the type of node, classified by type of children.*/
#define PARTICLE_NODE_TYPE 0
#define NODE_NODE_TYPE 1
#define PSEUDO_NODE_TYPE 2

/* Define to build a tree containing all types of particles*/
#define ALLMASK (1<<6)-1
#define GASMASK (1)
#define DMMASK (2)
#define NUMASK (1<<2)
#define STARMASK (1<<4)
#define BHMASK (1<<5)

/* Size of the free Node thread cache.
 * 12 8-node rows (works out at 8kB) was found
 * to be optimal for an Intel skylake and
 * an AMD Zen2 with 12 threads.*/
#define NODECACHE_SIZE (8*12)

/*Structure containing thread-local parameters of the tree build*/
class NodeCache {
public:
    int nnext_thread;
    int nrem_thread;

    NodeCache(int nnext): nnext_thread(nnext), nrem_thread(NODECACHE_SIZE)
    {}

    /*Get a pointer to memory for 8 free nodes, from our node cache. */
    int get_freenode(int64_t * nnext)
    {
        /*Get memory for an extra node from our cache.*/
        if(nrem_thread < 8) {
            nnext_thread = atomic_fetch_and_add_64(nnext, NODECACHE_SIZE);
            nrem_thread = NODECACHE_SIZE;
        }
        const int ninsert = nnext_thread;
        nnext_thread += 8;
        nrem_thread -= 8;
        return ninsert;
    }
};


struct NodeChild
{
    /*!< pointers to daughter nodes or daughter particles. */
    int suns[NMAXCHILD];
    /* Number of daughter particles if node contains particles.
     * During treebuild >= (1<<16) if node contains nodes.*/
    int noccupied;
};

class NODE
{
public:
    int sibling;		/*!< this gives the next node in the walk in case the current node can be used */
    int father;		/*!< this gives the parent node of each node (or -1 if we have the root node) */
    MyFloat len;			/*!< sidelength of treenode */
    MyFloat center[3];		/*!< geometrical center of node */

    struct {
        MyFloat cofm[3];		/*!< center of mass of node */
        MyFloat mass;		/*!< mass of node */
        MyFloat hmax;           /*!< maximum amount by which Pos + Hsml of all gas particles in the node exceeds len for this node. */
    } mom;

    /* If the current node needs to be opened, go to the first element of this array.
     * In principle storing the full array wastes memory, because we only use it for the leaf nodes.
     * However, in practice the wasted memory is fairly small: there are sum(1/8^n) ~ 0.15 internal nodes
     * for each leaf node, and we are losing 30% of the memory per node, so the total lost is 5%.
     * Any attempt to get it back by using a separate allocation means we lost the ability to resize
     * the Nodes array and that is always worse.*/
    struct NodeChild s;
    struct {
        unsigned int InternalTopLevel :1; /* TopLevel and has a child which is also TopLevel*/
        unsigned int TopLevel :1; /* Node corresponding to a toplevel node */
        unsigned int DependsOnLocalMass :1;  /* Intersects with local mass */
        unsigned int ChildType :2; /* Specify the type of children this node has: particles, other nodes, or pseudo-particles.
                                    * (should be an enum, but not standard in C).*/
        unsigned int unused : 3; /* Spare bits*/
    } f;

    /* Initialise an internal node. The parent is assumed to be locked, and
     * we have assured that nothing else will change the node while we are here.*/
    NODE(struct NODE *parent, int subnode) : sibling(-10), father(-10), len(0.5 * parent->len)
    {
        const MyFloat lenhalf = 0.25 * parent->len;
        f.TopLevel = 0;
        f.InternalTopLevel = 0;
        f.DependsOnLocalMass = 0;
        f.ChildType = PARTICLE_NODE_TYPE;
        f.unused = 0;

        for(int j = 0; j < 3; j++) {
            /* Detect which quadrant we are in by testing the bits of subnode:
            * if (subnode & [1,2,4]) is true we add lenhalf, otherwise subtract lenhalf*/
            const int sign = (subnode & (1 << j)) ? 1 : -1;
            center[j] = parent->center[j] + sign*lenhalf;
        }
        for(int j = 0; j < NMAXCHILD; j++)
            s.suns[j] = -1;
        s.noccupied = 0;
        memset(&(mom.cofm),0,3*sizeof(MyFloat));
        mom.mass = 0;
        mom.hmax = 0;
    }

    /* Get the subnode for a given particle and parent node.
    * This splits a parent node into 8 subregions depending on the particle position.
    * node is the parent node to split, p_i is the index of the particle we
    * are currently inserting
    * Returns a value between 0 and 7.
    * */
    int get_subnode(const double Pos[3])
    {
        /*Loop is unrolled to help out the compiler,which normally only manages it at -O3*/
        return (Pos[0] > center[0]) +
                ((Pos[1] > center[1]) << 1) +
                ((Pos[2] > center[2]) << 2);
    }

    /*Check whether a particle is inside the volume covered by a node,
    * by checking whether each dimension is close enough to center (L1 metric).
    * 'Nugget' is the allowed floating point error.*/
    int inside_node(const double Pos[3], const double nugget)
    {
        /*One can also use a loop, but the compiler unrolls it only at -O3,
        *so this is a little faster*/
        int inside =
            (fabs(2*(Pos[0] - center[0])) <= len + nugget) *
            (fabs(2*(Pos[1] - center[1])) <= len + nugget) *
            (fabs(2*(Pos[2] - center[2])) <= len + nugget);
        return inside;
    }

    /* Set the center of mass of the current node*/
    void force_update_particle_node(void)
    {
    #ifdef DEBUG
        if(f.ChildType != PARTICLE_NODE_TYPE)
            endrun(3, "force_update_particle_node called on node of wrong type %d!\n", f.ChildType);
    #endif
        int j;
        /*Set the center of mass moments*/
        const double mass = mom.mass;
        /* Be careful about empty nodes*/
        if(mass > 0) {
            for(j = 0; j < 3; j++)
                mom.cofm[j] /= mass;
        }
        else {
            for(j = 0; j < 3; j++)
                mom.cofm[j] = center[j];
        }
    }
};

/*Structure that contains the top tree only, and various Tree metadata.
 * No particle data is entered here: it simply mirrors the domain decomposition.*/
class TopTree {
protected:
    /*Index of first pseudo-particle node*/
    int64_t lastnode;
    /* Number of actually allocated nodes*/
    int64_t numnodes = 0;
    /*Pointer to the tree nodes. If a negative index is given, a particle is accessed.*/
    struct NODE *Nodes = NULL;
    /*Pointer to the TopLeaves struct imported from Domain. Sets up the pseudo particles.*/
    struct topleaf_data * TopLeaves = NULL;
    /*Number of TopLeaves*/
    int NTopLeaves = 0;
    /* Index of current task*/
    int ThisTask;

    /*! This function recursively creates a set of empty tree nodes which
    *  corresponds to the top-level tree for the ddecomp grid. This is done to
    *  ensure that this top-level tree is always "complete" so that we can easily
    *  associate the pseudo-particles of other CPUs with tree-nodes at a given
    *  level in the tree, even when the particle population is so sparse that
    *  some of these nodes are actually empty.
    */
    void force_create_node_for_topnode(int no, int topnode, const DomainDecomp * ddecomp, const int bits, const int x, const int y, const int z)
    {
        /*We reached the leaf of the toptree*/
        const int curdaughter = ddecomp->TopNodes[topnode].Daughter;
        if(curdaughter < 0)
            return;

        for(int i = 0; i < 2; i++)
            for(int j = 0; j < 2; j++)
                for(int k = 0; k < 2; k++)
                {
                    int sub = 7 & peano_hilbert_key((x << 1) + i, (y << 1) + j, (z << 1) + k, bits);

                    int count = i + 2 * j + 4 * k;

                    Nodes[no].s.suns[count] = numnodes;
                    /*We are an internal top level node as we now have a child top level.*/
                    Nodes[no].f.InternalTopLevel = 1;
                    Nodes[no].f.ChildType = NODE_NODE_TYPE;
                    Nodes[no].s.noccupied = NODEFULL;

                    /* We create a new leaf node.*/
                    new (&Nodes[numnodes]) NODE(&Nodes[no], count);
                    /*Set father of new node*/
                    Nodes[numnodes].father = no;
                    /*All nodes here are top level nodes*/
                    Nodes[numnodes].f.TopLevel = 1;

                    if(curdaughter + sub >= ddecomp->NTopNodes)
                        endrun(5, "Invalid topnode: daughter %d sub %d > topnodes %d\n", curdaughter, sub, ddecomp->NTopNodes);
                    const struct topnode_data curtopnode = ddecomp->TopNodes[curdaughter + sub];
                    if(curtopnode.Daughter == -1) {
                        int ThisTask;
                        MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
                        TopLeaves[curtopnode.Leaf].treenode = numnodes;
                        /* We set the first child as a pointer to the topleaf, essentially constructing the pseudoparticles early.
                        * We do not set nocc, so this first child will be over-written on local nodes when we construct the full tree.*/
                        Nodes[numnodes].s.suns[0] = curtopnode.Leaf + lastnode;
                        if(TopLeaves[curtopnode.Leaf].Task != ThisTask)
                            Nodes[numnodes].f.ChildType = PSEUDO_NODE_TYPE;
                    }

                    numnodes++;

                    if(numnodes >= lastnode)
                        endrun(11, "Not enough force nodes to topnode grid: need %ld\n", lastnode);
                }
        /* Set sibling on the child*/
        for(int j=0; j<7; j++) {
            int chld = Nodes[no].s.suns[j];
            Nodes[chld].sibling = Nodes[no].s.suns[j+1];
        }
        Nodes[Nodes[no].s.suns[7]].sibling = Nodes[no].sibling;
        for(int i = 0; i < 2; i++)
            for(int j = 0; j < 2; j++)
                for(int k = 0; k < 2; k++)
                {
                    int sub = 7 & peano_hilbert_key((x << 1) + i, (y << 1) + j, (z << 1) + k, bits);
                    int count = i + 2 * j + 4 * k;
                    force_create_node_for_topnode(Nodes[no].s.suns[count], ddecomp->TopNodes[topnode].Daughter + sub, ddecomp,
                            bits + 1, 2 * x + i, 2 * y + j, 2 * z + k);
                }
    }

public:
    double BoxSize;

    /* A toptree for domain exchange. This creates only the topnodes, which shadow elements of the domain decomposition.*/
    TopTree(DomainDecomp * ddecomp, const double i_BoxSize) : TopLeaves(ddecomp->TopLeaves), NTopLeaves(ddecomp->NTopLeaves), BoxSize(i_BoxSize)
    {
        /* Allocate memory. Two extra for the first node and for a sentinel*/
        Nodes = mymanagedmalloc("Nodes_base", struct NODE, ddecomp->NTopNodes+2);
        lastnode = ddecomp->NTopNodes+2;
        MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);

        // message(1, "Building toptree last %d, topnodes %d\n", lastnode, ddecomp->NTopNodes);
        numnodes = 0;       /* index of first free node */
        struct NODE *nfreep = &Nodes[numnodes];
        nfreep->len = BoxSize*1.001;
        for(int i = 0; i < 3; i++)
            nfreep->center[i] = BoxSize/2.;
        for(int i = 0; i < NMAXCHILD; i++)
            nfreep->s.suns[i] = -1;
        nfreep->s.noccupied = 0;
        nfreep->father = -1;
        nfreep->sibling = -1;
        nfreep->f.TopLevel = 1;
        nfreep->f.InternalTopLevel = 0;
        nfreep->f.DependsOnLocalMass = 0;
        nfreep->f.ChildType = PARTICLE_NODE_TYPE;
        nfreep->f.unused = 0;
        memset(&(nfreep->mom.cofm),0,3*sizeof(MyFloat));
        nfreep->mom.mass = 0;
        nfreep->mom.hmax = 0;
        /* Set the treenode for this node*/
        ddecomp->TopLeaves[0].treenode = numnodes;
        /* create a set of empty nodes corresponding to the top-level ddecomp
         * grid. We need to generate these nodes first to make sure that we have a
         * complete top-level tree which allows the easy insertion of the
         * pseudo-particles in the right place */
        force_create_node_for_topnode(0, 0, ddecomp, 1, 0, 0, 0);
    }

    ~TopTree(void)
    {
        myfree(Nodes);
    }

};

/*Structure containing the Node pointer, and various Tree metadata.
 * This stores the oct-tree structure.*/
template <typename DerivedTree> class ForceTree : public TopTree {
protected:
    /* Each processor allocates a number of nodes which is TreeAllocFactor times
       the maximum(!) number of particles.  Note: A typical local tree for N
       particles needs usually about ~0.65*N nodes.
       If the allocated memory is not sufficient, this parameter will be increased.*/
    double TreeAllocFactor = 0.9;
    /* Types which are included have their bits set to 1*/
    int mask = ALLMASK;
    /* Number of particles stored in this tree*/
    int64_t NumParticles;
    int64_t MaxPart;

    /* Constructor for derived classes: sets up the toptree and the mask, but
     * does not build the local tree. The derived class is responsible for
     * calling ForceTree_internal itself.*/
    ForceTree(DomainDecomp * ddecomp, const double BoxSize, const int i_mask):
    TopTree(ddecomp, BoxSize), mask(i_mask)
    {  }

    void ForceTree_internal(part_manager_type& PartManager, DomainDecomp * ddecomp, ActiveParticles * act, const std::string EmergencyOutputDir)
    {
        int64_t maxnodes = TreeAllocFactor * PartManager.NumPart + ddecomp->NTopNodes;
        /* int64_t maxmaxnodes;
        MPI_Reduce(&maxnodes, &maxmaxnodes, 1, MPI_INT64, MPI_MAX,0, MPI_COMM_WORLD);
        message(0, "Treebuild: Largest is %g MByte for %ld tree nodes. (presently allocated %g MB)\n",
            maxmaxnodes * sizeof(struct NODE) / (1024.0 * 1024.0), maxmaxnodes, PartManager->MaxPart,
            mymalloc_usedbytes() / (1024.0 * 1024.0));*/

        /* Make a copy of the node pointer from the toptree*/
        struct NODE * TopNodes = Nodes;

        do
        {
            /* Allocate memory: note that because node numbers are passed around between ranks,
            * this has to be something which is the same on all ranks. */
            Nodes = mymanagedmalloc("Nodes_base", struct NODE, (maxnodes + 1));
            memcpy(Nodes, TopNodes, numnodes * sizeof(struct NODE));
            lastnode = maxnodes;

            force_tree_create_nodes(act, ddecomp);

            if(numnodes >= maxnodes)
            {
                message(1, "Not enough tree nodes (%ld) for %ld particles. Created %ld\n", maxnodes, act->NumActiveParticle, numnodes);
                myfree(Nodes);
                TreeAllocFactor *= 1.15;
                if(TreeAllocFactor > 3.0) {
    #ifndef DEBUG
                    endrun(2, "TreeAllocFactor is %g nodes, which is too large!\n", TreeAllocFactor);
    #else
                    break;
    #endif
                }
                maxnodes = TreeAllocFactor * PartManager.NumPart + ddecomp->NTopNodes;
                message(1, "TreeAllocFactor from %g to %g now %ld tree nodes\n", TreeAllocFactor, TreeAllocFactor*1.15, maxnodes);
            }
        }
        while(numnodes >= lastnode);

    #ifdef DEBUG
        if(MPIU_Any(TreeAllocFactor > 3.0, MPI_COMM_WORLD)) {
            /* Assume scale factor = 1 for dump as position is not affected.*/
            if(EmergencyOutputDir.size() > 0) {
                Cosmology CP = {0};
                CP.Omega0 = 0.3;
                CP.OmegaLambda = 0.7;
                CP.HubbleParam = 0.7;
                dump_snapshot("FORCETREE-DUMP", 1, &CP, EmergencyOutputDir);
            }
            endrun(2, "Required too many nodes, snapshot dumped\n");
        }
    #endif
        myfree(TopNodes);
        report_memory_usage("FORCETREE");

        int64_t allact = NumParticles;
        int64_t maxnumnodes = numnodes;
    #ifdef DEBUG
        force_validate_nextlist();
        MPI_Reduce(&NumParticles, &allact, 1, MPI_INT64, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&numnodes, &maxnumnodes, 1, MPI_INT64, MPI_MAX, 0, MPI_COMM_WORLD);
    #endif
        message(0, "Tree constructed (type mask: %d) with %ld particles. Num nodes %ld, first pseudo %ld. NTopLeaves %d\n",
                mask, allact, maxnumnodes, lastnode, NTopLeaves);
    }

public:
    /* Main constructor with a mask argument.
    * Mask is a bitfield, specified as 1 for each type that should be included. Use ALLMASK for all particle types.
    * This is much faster than _full: because the particles are sorted by type the merge step is much faster than
    * with all particle types, and of course the tree is smaller.*/
    ForceTree(part_manager_type& PartManager, DomainDecomp * ddecomp, const int i_mask, const std::string EmergencyOutputDir):
    TopTree(ddecomp, PartManager.BoxSize), mask(i_mask)
    {
        message(0, "Tree construction for types: %d.\n", mask);
        /* Build for all particles*/
        ActiveParticles act = init_empty_active_particles(&PartManager);
        ForceTree_internal(PartManager, ddecomp, &act, EmergencyOutputDir);
    }

    bool node_is_pseudo_particle(const int no) const
    {
        return no >= lastnode;
    }

    bool node_is_particle(const int no) const
    {
        return no < 0 && no >= -1 * MaxPart;
    }

    bool
    node_is_node(const int no) const
    {
        return (no >= 0) && (no < numnodes);
    }

    /* Add a particle to a node in a known empty location.
     * Parent is assumed to be locked.*/
    void
    modify_internal_node(int parent, int subnode, int p_toplace)
    {
        Nodes[parent].s.suns[subnode] = p_toplace;
    }

    /* Create a new layer of nodes beneath the current node, and place the particle.
    * Must have node lock.*/
    int create_new_node_layer(int firstparent, int p_toplace, int64_t *nnext, struct NodeCache *nc)
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
            newsuns[0] = nc->get_freenode(nnext);
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
                new (nfreep) NODE(nprnt, 1);
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
                int subnode = nprnt->get_subnode(Part[oldsuns[i]].Pos);
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
            int subnode = nprnt->get_subnode(Part[p_toplace].Pos);
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

    /*! Does initial creation of the nodes for the gravitational oct-tree.
    * mask is a bitfield: Only types whose bit is set are added.
    **/
    void force_tree_create_nodes(const ActiveParticles * act, DomainDecomp * ddecomp)
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
            struct NodeCache nc(first_free + tid * NODECACHE_SIZE);

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
                if(Nodes[this_acc].inside_node(Part[i].Pos, 0)) {
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
                    if(!Nodes[cur].inside_node(Part[i].Pos, 1e-7))
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
    int
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
    int force_update_node_recursive(const int no, const int sib, const int level)
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
                Nodes[p].force_update_particle_node();
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

    /* Add a particle to the tree, extending the tree as necessary. Locking is done,
    * so may be called from a threaded context*/
    int add_particle_to_tree(int i, int cur_start, struct NodeCache *nc, int64_t* nnext)
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
            int subnode = Nodes[cur].get_subnode(Part[i].Pos);
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
    int merge_partial_force_trees(int left, int right, struct NodeCache * nc, int64_t * nnext)
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

    /*! This routine determines the multipole moments for a given internal node
    *  and all its subnodes in parallel, assigning the recursive algorithm to different threads using openmp's task api.
    *  The result is stored in tb.Nodes in the sequence of this tree-walk.
    *
    * - A new task is spawned from each  down from each local topleaf. Local topleaves are used
    * so that we do not waste time trying moment calculation with pseudoparticles.
    * - Each internal node found at that level is added to a list, together with its sibling.
    * - Each node in this list then has the recursive moment calculation called on it.
    * Note: If the tree is very unbalanced and one branch much deeper than the others, this will not be efficient.
    * - Once each tree's recursive moment is generated in parallel, the tail value from each recursion is stored, and the node marked as done.
    * - A final recursive moment calculation is run in serial for the top 3 levels of the tree. When it encounters one of the pre-computed nodes, it
    * searches the list of pre-computed tail values to set the next node as if it had recursed and continues.
    */
    void force_update_node_parallel(const DomainDecomp * const ddecomp)
    {
    #pragma omp parallel
    #pragma omp single nowait
        {
            for(int i = ddecomp->Tasks[ThisTask].StartLeaf; i < ddecomp->Tasks[ThisTask].EndLeaf; i ++) {
                const int no = ddecomp->TopLeaves[i].treenode;
                /* Set local mass dependence*/
                Nodes[no].f.DependsOnLocalMass = 1;
                /* Nodes containing other nodes: the overwhelmingly likely case.*/
                if(Nodes[no].f.ChildType == NODE_NODE_TYPE) {
                    #pragma omp task default(none) firstprivate(no)
                    force_update_node_recursive(no, Nodes[no].sibling, 1);
                }
                else if(Nodes[no].f.ChildType == PARTICLE_NODE_TYPE)
                    Nodes[no].force_update_particle_node();
                else if(Nodes[no].f.ChildType == PSEUDO_NODE_TYPE)
                    endrun(5, "Error, found pseudo node %d but domain entry %d says on task %d\n", no, i, ThisTask);
            }
        }
    }

    /* In the child classes, this does the update of the pseudo-node moments.
     * Here we just do the taskwait.*/
    void modify_pseudo_node(struct NODE * node)
    {
        /*Make sure all child nodes are done*/
        #pragma omp taskwait
    }

    /*! This function updates the top-level tree after the multipole moments of
    *  the pseudo-particles have been updated.
    */
    void force_treeupdate_pseudos(const int no, const int level)
    {
        /* This happens if we have a trivial domain with only one entry*/
        if(!Nodes[no].f.InternalTopLevel)
            return;

        int j;

        /* since we are dealing with top-level nodes, we know that there are 8 consecutive daughter nodes */
        for(j = 0; j < 8; j++)
        {
            const int p = Nodes[no].s.suns[j];

            /*This may not happen as we are an internal top level node*/
            if(p < 0 || p >= lastnode)
                endrun(6767, "Updating pseudos: %d -> %d which is not an internal node between 0 and %ld\n",no, p, lastnode);
    #ifdef DEBUG
            /* Check we don't move to another part of the tree*/
            if(Nodes[p].father != no)
                endrun(6767, "Tried to update toplevel node %d with parent %d != expected %d\n", p, Nodes[p].father, no);
    #endif

            if(Nodes[p].f.InternalTopLevel) {
                if(level < 512) {
                    #pragma omp task default(none) firstprivate(p, level)
                    force_treeupdate_pseudos(p, level*8);
                }
                else {
                    force_treeupdate_pseudos(p, level);
                }
            }
        }
        static_cast<DerivedTree*>(this)->modify_pseudo_node(&Nodes[no]);
    }

#ifdef DEBUG
    /* Walk the constructed tree, validating sibling and nextnode as we go*/
    void force_validate_nextlist(void)
    {
        int no = 0;
        while(no != -1)
        {
            struct NODE * current = &Nodes[no];
            if(current->sibling != -1 && !node_is_node(current->sibling))
                endrun(5, "Node %d (type %d) has sibling %d next %d father %d final %ld last %ld ntop %d\n", no, current->f.ChildType, current->sibling, current->s.suns[0], current->father, numnodes, lastnode, NTopLeaves);

            if(current->f.ChildType == PSEUDO_NODE_TYPE) {
                /* pseudo particle: nextnode should be a pseudo particle, sibling should be a node. */
                if(!node_is_pseudo_particle(current->s.suns[0]))
                    endrun(5, "Pseudo Node %d has next node %d sibling %d father %d final %ld last %ld ntop %d\n", no, current->s.suns[0], current->sibling, current->father, numnodes, lastnode, NTopLeaves);
            }
            else if(current->f.ChildType == NODE_NODE_TYPE) {
                /* Next node should be another node */
                if(!node_is_node(current->s.suns[0]))
                    endrun(5, "Node Node %d has next node which is particle %d sibling %d father %d final %ld last %ld ntop %d\n", no, current->s.suns[0], current->sibling, current->father, numnodes, lastnode, NTopLeaves);
                no = current->s.suns[0];
                continue;
            }
            no = current->sibling;
        }
        /* Every node should have a valid father: collect those that do not.*/
        for(no = 0; no < numnodes; no++)
        {
            if(!node_is_node(Nodes[no].father) && Nodes[no].father >= 0) {
                struct NODE *current = &Nodes[no];
                message(1, "Danger! no %d has father %d, next %d sib %d, (ptype = %d) len %g center (%g %g %g) mass %g cofm %g %g %g TL %d DLM %d ITL %d nocc %d suns %d %d %d %d\n", no, current->father, current->s.suns[0], current->sibling, current->f.ChildType,
                    current->len, current->center[0], current->center[1], current->center[2],
                    current->mom.mass, current->mom.cofm[0], current->mom.cofm[1], current->mom.cofm[2],
                    current->f.TopLevel, current->f.DependsOnLocalMass, current->f.InternalTopLevel, current->s.noccupied,
                    current->s.suns[0], current->s.suns[1], current->s.suns[2], current->s.suns[3]);
            }
        }
        walltime_measure("/Tree/Build/Validate");
    }
#endif
};

/* ForceTree variant which contains hmax.*/
class ForceTreeHmax : public ForceTree<ForceTreeHmax> {
    /* The CRTP base dispatches to our modify_internal_node, which is private.*/
    friend class ForceTree<ForceTreeHmax>;
public:
    /* Flags that hmax has been computed for this tree*/
    bool hmax_computed_flag = false;
    /*!< gives parent node in tree for every particle */
    int *Father = NULL;
    int64_t nfather = 0;

    /* Get the father of a node*/
    int
    get_father_node(const int no) const
    {
        return Nodes[no].father;
    }

    /* Get the father of a particle*/
    int
    get_father_particle(const int no) const
    {
        return Father[no];
    }

    /* Build a tree structure using all particles, compute moments and allocate a father array.
     * This is the fattest tree constructor, allows moments and walking up and down.*/
    ForceTreeHmax(part_manager_type& PartManager, DomainDecomp * ddecomp, int i_mask, const std::string EmergencyOutputDir):
    ForceTree(ddecomp, PartManager.BoxSize, i_mask)
    {
        /* Allocate the Father array before building: modify_internal_node
         * fills it in as each particle is attached to a node.*/
        Father = mymalloc("Father", int, PartManager.MaxPart);
        nfather = PartManager.MaxPart;
#ifdef DEBUG
        memset(Father, -1, PartManager.MaxPart * sizeof(int));
#endif
        ActiveParticles act = init_empty_active_particles(&PartManager);
        ForceTree_internal(PartManager, ddecomp, &act, EmergencyOutputDir);
    }

    /* Update the hmax in the parent node of the particle p_i*/
    void
    update_tree_hmax_father(const int p_i, const double Pos[3], const double Hsml)
    {
        if(!Father)
            endrun(4, "Father not allocated in tree_hmax_father\n");
        const int no = Father[p_i];
    #ifdef DEBUG
        if(no < 0)
            endrun(5, "Father for particle %d pos %g %g %g hsml %g not initialised, likely not in tree\n", p_i, Pos[0], Pos[1], Pos[2], Hsml);
    #endif
        struct NODE * node = &Nodes[no];
        /* How much does this particle peek beyond this node?
            * Note len does not change so we can read it without a lock or atomic. */

        MyFloat newhmax = 0;
        int j;
        for(j = 0; j < 3; j++)
            newhmax = DMAX(newhmax, fabs(Pos[j] - node->center[j]) + Hsml - node->len/2.);

        MyFloat readhmax;
        #pragma omp atomic read
        readhmax = node->mom.hmax;

        do {
            if (newhmax <= readhmax)
                break;
            /* Swap in the new hmax only if the old one hasn't changed. */
        } while(!__atomic_compare_exchange(&(node->mom.hmax), &readhmax, &newhmax, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED));
    }

    /* Update hmax recursively for all internal nodes*/
    void
    force_tree_calc_hmax(DomainDecomp * ddecomp)
    {
        force_update_node_parallel(ddecomp);
        /* Exchange the pseudo-data*/
        force_exchange_pseudodata(ddecomp);
        #pragma omp parallel
        #pragma omp single nowait
        {
            force_treeupdate_pseudos(0, 1);
        }
        hmax_computed_flag = true;
    }

private:
    /* Add a particle to a node in a known empty location.
     * Parent is assumed to be locked.*/
    void modify_internal_node(int parent, int subnode, int p_toplace)
    {
        ForceTree::modify_internal_node(parent, subnode, p_toplace);
        Father[p_toplace] = parent;

        const auto& part = Part[p_toplace];
        auto& pnode = Nodes[parent];

        /* We do not add active particles to the hmax here.
        * The active particles will have hsml updated in density_postprocess instead, often to a smaller value.*/
        if((part.Type == 0 || part.Type == 5 )&& !is_timebin_active(part.TimeBinHydro, part.Ti_drift))
        {
            /* Maximal distance any of the member particles peek out from the side of the node.
            * May be at most hsml, as |Pos - Center| < len/2.*/
            for(int j = 0; j < 3; j++) {
                pnode.mom.hmax = DMAX(pnode.mom.hmax, fabs(part.Pos[j] - pnode.center[j]) + part.Hsml - pnode.len/2.);
            }
        }
    }

    /* Update the pseudo-node hmax.*/
    void modify_pseudo_node(struct NODE * node)
    {
        /* Zero the moments*/
        node->mom.hmax = 0;

        /*Make sure all child nodes are done*/
        #pragma omp taskwait

        for(int j = 0; j < 8; j++)
        {
            const int p = node->s.suns[j];
            if(Nodes[p].mom.hmax > node->mom.hmax)
                node->mom.hmax = Nodes[p].mom.hmax;
            if(Nodes[p].f.DependsOnLocalMass)
                node->f.DependsOnLocalMass = 1;
        }
    }

        /*! This function communicates the values of the multipole moments of the
    *  top-level tree-nodes of the ddecomp grid.  This data can then be used to
    *  update the pseudo-particles on each CPU accordingly.
    */
    void force_exchange_pseudodata(const DomainDecomp * const ddecomp)
    {
        MyFloat * TopLeafMoments = mymalloc("TopLeafMoments", MyFloat, ddecomp->NTopLeaves);

        #pragma omp parallel for
        for(int i = ddecomp->Tasks[ThisTask].StartLeaf; i < ddecomp->Tasks[ThisTask].EndLeaf; i ++) {
            int no = ddecomp->TopLeaves[i].treenode;
            if(ddecomp->TopLeaves[i].Task != ThisTask)
                endrun(131231231, "TopLeaf %d Task table is corrupted: task is %d\n", i, ddecomp->TopLeaves[i].Task);
            /* read out the hmax from the local base cells */
            TopLeafMoments[i] = Nodes[no].mom.hmax;
        }

        /* share the pseudo-particle data across CPUs */
        int NTask;
        MPI_Comm_size(MPI_COMM_WORLD, &NTask);

        int * recvcounts = mymalloc("recvcounts", int, NTask);
        int * recvoffset = mymalloc("recvoffset", int, NTask);

        for(int recvTask = 0; recvTask < NTask; recvTask++)
        {
            recvoffset[recvTask] = ddecomp->Tasks[recvTask].StartLeaf * sizeof(TopLeafMoments[0]);
            recvcounts[recvTask] = (ddecomp->Tasks[recvTask].EndLeaf - ddecomp->Tasks[recvTask].StartLeaf) * sizeof(TopLeafMoments[0]);
        }

        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                &TopLeafMoments[0], recvcounts, recvoffset,
                MPI_BYTE, MPI_COMM_WORLD);

        myfree(recvoffset);
        myfree(recvcounts);

        for(int ta = 0; ta < NTask; ta++) {
            if(ta == ThisTask)
                continue; /* bypass ThisTask since it is already up to date */
            #pragma omp parallel for
            for(int i = ddecomp->Tasks[ta].StartLeaf; i < ddecomp->Tasks[ta].EndLeaf; i ++) {
                const int no = ddecomp->TopLeaves[i].treenode;
                Nodes[no].mom.hmax = TopLeafMoments[i];
            }
        }
        myfree(TopLeafMoments);
    }
};

/*Structure containing the Node pointer, and various Tree metadata.*/
/*The node index is an integer. For nodes which have type NODE_NODE_TYPE
 * it points to another node. For nodes which have type PARTICLE_NODE_TYPE
 * if points to a particle. For nodes which have type PSEUDO_NODE_TYPE
 * it points to a pseudo particle on another processor.*/
class ForceTreeMoments : public ForceTree <ForceTreeMoments> {
    /* The CRTP base dispatches to our modify_internal_node, which is private.*/
    friend class ForceTree<ForceTreeMoments>;
public:
    /* Flags that the tree contains all particles*/
    bool full_particle_tree_flag;

    /* Build a tree structure using all particles, compute moments and allocate a father array.
     * This is the fattest tree constructor, allows moments and walking up and down.*/
    ForceTreeMoments(part_manager_type& PartManager, DomainDecomp * ddecomp, ActiveParticles * act, bool HybridNuTracer, const std::string EmergencyOutputDir):
    ForceTree(ddecomp, PartManager.BoxSize, ALLMASK), full_particle_tree_flag(act == NULL)
    {
        /* Build for all particles by default, but skip neutrinos if they are passive.*/
        if(HybridNuTracer)
            mask = GASMASK + DMMASK + STARMASK + BHMASK;
        ForceTree_internal(PartManager, ddecomp, act, EmergencyOutputDir);
        walltime_measure("/Tree/Build/Nodes");
        /* Compute moments of the force tree, recursively.*/
        force_update_node_parallel(ddecomp);
        /* Exchange the pseudo-data*/
        force_exchange_pseudodata(ddecomp);
        #pragma omp parallel
        #pragma omp single nowait
        {
            force_treeupdate_pseudos(0, 1);
        }
        walltime_measure("/Tree/Build/Moments");
    }

    ~ForceTreeMoments(void){ }

private:
    /* Add a particle to a node in a known empty location.
     * Parent is assumed to be locked.*/
    void
    modify_internal_node(int parent, int subnode, int p_toplace)
    {
        ForceTree::modify_internal_node(parent, subnode, p_toplace);
        const struct particle_data& part = Part[p_toplace];
        auto& pnode = Nodes[parent];
        pnode.mom.mass += (part.Mass);
        for(int k=0; k<3; k++)
            pnode.mom.cofm[k] += (part.Mass * part.Pos[k]);
    }

    /* In the child classes, this does the update of the pseudo-node moments.
     * Here we just do the taskwait.*/
    void modify_pseudo_node(struct NODE * node)
    {
        /* Zero the moments*/
        node->mom.mass = 0;
        node->mom.cofm[0] = 0;
        node->mom.cofm[1] = 0;
        node->mom.cofm[2] = 0;

        /*Make sure all child nodes are done*/
        #pragma omp taskwait

        for(int j = 0; j < 8; j++)
        {
            const int p = node->s.suns[j];

            node->mom.mass += (Nodes[p].mom.mass);
            node->mom.cofm[0] += (Nodes[p].mom.mass * Nodes[p].mom.cofm[0]);
            node->mom.cofm[1] += (Nodes[p].mom.mass * Nodes[p].mom.cofm[1]);
            node->mom.cofm[2] += (Nodes[p].mom.mass * Nodes[p].mom.cofm[2]);

            if(Nodes[p].f.DependsOnLocalMass)
                node->f.DependsOnLocalMass = 1;
        }

        if(node->mom.mass)
        {
            node->mom.cofm[0] /= node->mom.mass;
            node->mom.cofm[1] /= node->mom.mass;
            node->mom.cofm[2] /= node->mom.mass;
        }
        else
        {
            node->mom.cofm[0] = node->center[0];
            node->mom.cofm[1] = node->center[1];
            node->mom.cofm[2] = node->center[2];
        }
    }
        /*! This function communicates the values of the multipole moments of the
    *  top-level tree-nodes of the ddecomp grid.  This data can then be used to
    *  update the pseudo-particles on each CPU accordingly.
    */
    void force_exchange_pseudodata(const DomainDecomp * const ddecomp)
    {
        struct topleaf_momentsdata {
            MyFloat s[3];
            MyFloat mass;
        };

        struct topleaf_momentsdata * TopLeafMoments = mymalloc("TopLeafMoments", struct topleaf_momentsdata, ddecomp->NTopLeaves);

        #pragma omp parallel for
        for(int i = ddecomp->Tasks[ThisTask].StartLeaf; i < ddecomp->Tasks[ThisTask].EndLeaf; i ++) {
            int no = ddecomp->TopLeaves[i].treenode;
            if(ddecomp->TopLeaves[i].Task != ThisTask)
                endrun(131231231, "TopLeaf %d Task table is corrupted: task is %d\n", i, ddecomp->TopLeaves[i].Task);
            /* read out the multipole moments from the local base cells */
            TopLeafMoments[i].s[0] = Nodes[no].mom.cofm[0];
            TopLeafMoments[i].s[1] = Nodes[no].mom.cofm[1];
            TopLeafMoments[i].s[2] = Nodes[no].mom.cofm[2];
            TopLeafMoments[i].mass = Nodes[no].mom.mass;
        }

        /* share the pseudo-particle data across CPUs */
        int NTask;
        MPI_Comm_size(MPI_COMM_WORLD, &NTask);

        int * recvcounts = mymalloc("recvcounts", int, NTask);
        int * recvoffset = mymalloc("recvoffset", int, NTask);

        for(int recvTask = 0; recvTask < NTask; recvTask++)
        {
            recvoffset[recvTask] = ddecomp->Tasks[recvTask].StartLeaf * sizeof(TopLeafMoments[0]);
            recvcounts[recvTask] = (ddecomp->Tasks[recvTask].EndLeaf - ddecomp->Tasks[recvTask].StartLeaf) * sizeof(TopLeafMoments[0]);
        }

        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                &TopLeafMoments[0], recvcounts, recvoffset,
                MPI_BYTE, MPI_COMM_WORLD);

        myfree(recvoffset);
        myfree(recvcounts);

        for(int ta = 0; ta < NTask; ta++) {
            if(ta == ThisTask)
                continue; /* bypass ThisTask since it is already up to date */
            #pragma omp parallel for
            for(int i = ddecomp->Tasks[ta].StartLeaf; i < ddecomp->Tasks[ta].EndLeaf; i ++) {
                const int no = ddecomp->TopLeaves[i].treenode;
                Nodes[no].mom.cofm[0] = TopLeafMoments[i].s[0];
                Nodes[no].mom.cofm[1] = TopLeafMoments[i].s[1];
                Nodes[no].mom.cofm[2] = TopLeafMoments[i].s[2];
                Nodes[no].mom.mass = TopLeafMoments[i].mass;
            }
        }
        myfree(TopLeafMoments);
    }

};


#endif
