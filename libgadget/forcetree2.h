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

struct NodeChild
{
    /*!< pointers to daughter nodes or daughter particles. */
    int suns[NMAXCHILD];
    /* Number of daughter particles if node contains particles.
     * During treebuild >= (1<<16) if node contains nodes.*/
    int noccupied;
};

struct NODE
{
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
};

/*Initialise an internal node at nfreep. The parent is assumed to be locked, and
 * we have assured that nothing else will change nfreep while we are here.*/
void init_internal_node(struct NODE *nfreep, struct NODE *parent, int subnode);
void force_update_particle_node(struct NODE& curnode);

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
                    init_internal_node(&Nodes[numnodes], &Nodes[no], count);
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

    int create_new_node_layer(int firstparent, int p_toplace, int64_t *nnext, struct NodeCache *nc);

    int add_particle_to_tree(int i, int cur_start, struct NodeCache *nc, int64_t* nnext);
    int merge_partial_force_trees(int left, int right, struct NodeCache * nc, int64_t * nnext);
    void force_tree_create_nodes(const ActiveParticles * act, DomainDecomp * ddecomp);

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

    int force_update_node_recursive(const int no, const int sib, const int level);

    /*! This function communicates the values of the multipole moments of the
    *  top-level tree-nodes of the ddecomp grid.  This data can then be used to
    *  update the pseudo-particles on each CPU accordingly.
    */
    void force_exchange_pseudodata(const DomainDecomp * const ddecomp)
    {
        struct topleaf_momentsdata {
            MyFloat s[3];
            MyFloat mass;
            MyFloat hmax;
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
            TopLeafMoments[i].hmax = Nodes[no].mom.hmax;
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
                Nodes[no].mom.hmax = TopLeafMoments[i].hmax;
            }
        }
        myfree(TopLeafMoments);
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
                    force_update_particle_node(Nodes[no]);
                else if(Nodes[no].f.ChildType == PSEUDO_NODE_TYPE)
                    endrun(5, "Error, found pseudo node %d but domain entry %d says on task %d\n", no, i, ThisTask);
            }
        }
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
        /* Zero the moments*/
        Nodes[no].mom.mass = 0;
        Nodes[no].mom.cofm[0] = 0;
        Nodes[no].mom.cofm[1] = 0;
        Nodes[no].mom.cofm[2] = 0;
        Nodes[no].mom.hmax = 0;

        /*Make sure all child nodes are done*/
        #pragma omp taskwait

        for(j = 0; j < 8; j++)
        {
            const int p = Nodes[no].s.suns[j];

            Nodes[no].mom.mass += (Nodes[p].mom.mass);
            Nodes[no].mom.cofm[0] += (Nodes[p].mom.mass * Nodes[p].mom.cofm[0]);
            Nodes[no].mom.cofm[1] += (Nodes[p].mom.mass * Nodes[p].mom.cofm[1]);
            Nodes[no].mom.cofm[2] += (Nodes[p].mom.mass * Nodes[p].mom.cofm[2]);

            if(Nodes[p].mom.hmax > Nodes[no].mom.hmax)
                Nodes[no].mom.hmax = Nodes[p].mom.hmax;
            if(Nodes[p].f.DependsOnLocalMass)
                Nodes[no].f.DependsOnLocalMass = 1;
        }

        if(Nodes[no].mom.mass)
        {
            Nodes[no].mom.cofm[0] /= Nodes[no].mom.mass;
            Nodes[no].mom.cofm[1] /= Nodes[no].mom.mass;
            Nodes[no].mom.cofm[2] /= Nodes[no].mom.mass;
        }
        else
        {
            Nodes[no].mom.cofm[0] = Nodes[no].center[0];
            Nodes[no].mom.cofm[1] = Nodes[no].center[1];
            Nodes[no].mom.cofm[2] = Nodes[no].center[2];
        }
    }
};


#endif
