#ifndef _LOCALEVALUATOR_H_
#define _LOCALEVALUATOR_H_
/* This header file contains the base classes for the treewalk.
 * Each class should be subclassed in each user of the treewalk, so that, for example,
 * GravTreeQuery derives from QueryBase.
 *
 * Everything here will be constructed and evaluated in the context of a single thread.
 * The thread may be in OpenMP, for the CPU evaluator, or on GPU, for the CUDA evaluator.
 * This imposes several restrictions:
 * - Nothing here may refer to global memory, as this is a data race on OpenMP and a segfault on GPU.
 * - GPU code may not call __host__ functions. This includes endrun and message.
 * Any uses of these functions here should be guarded by ifdef __CUDACC__
 * (the exception is the validate_tree() method), which is declared static, and used for checking on the host.
 * - Nothing here should call into MPI, as the GPU compiler is not an MPI compiler.
 *
 * Specific classes:
 * - ParamType: This contains input parameters of the treewalk, and should not be changed by the treewalk.
 * It should be a heap pointer so it can be copied to GPU.
 * - OutputType: The contains the output arrays, for example the accelerations.
 * - QueryType: This is the information needed for the particle that is walking the tree.
 * This data is also sent by the MPI_ISend to other processors.
 * - ResultType: Contains the information coming from a treewalk. This is received from other processes.
 *
 * LocalTreeWalk: these are functions for evaluating QueryType and ResultType.
 * The main function is visit() which contains the main loop for walking the tree.
 * In practice there are only two classes of LocalTreeWalk: the Gravity tree and the neighbour tree.
 * Everything apart from gravity is a neighbour tree,
 * LocalTopTreeType: This is as the LocalTreeWalk, but for the toptree, since there are a bunch of different
 * variables needed only for this. The toptree is not yet on GPU and so may use endrun.
 * */
#include <stdint.h>
#include <omp.h>
#include "domain.h"
#include "utils/endrun.h"
#include "forcetree.h"
#include "partmanager.h"
#include "types.h"

#define TREEWALK_REDUCE(A, B) if constexpr(mode==TREEWALK_PRIMARY){ (A) = (B);} else {(A) = ((A) + (B));}

/* Use a low number here. Larger numbers decrease the size of the export table, up to a point.
 * The need for a large Nodelist in older versions
 * was because we were sorting the DIT, so had incentive to keep it small.*/
#define  NODELISTLENGTH 4

enum NgbTreeFindSymmetric {
    NGB_TREEFIND_SYMMETRIC,
    NGB_TREEFIND_ASYMMETRIC,
};

enum TreeWalkReduceMode {
    TREEWALK_PRIMARY,
    TREEWALK_GHOSTS,
    TREEWALK_TOPTREE,
};

/* This enum specifies which mode the toptree walk is in.
 * TOPTREE_COUNT counts the number of exports, without writing to memory.
 * TOPTREE_EXPORT writes the export memory at a known-sized buffer.*/
enum TopTreeMode {
    TOPTREE_COUNT,
    TOPTREE_EXPORT,
};

/* Should be subclassed to contain the treewalk parameters specific to each treewalk.
 * Used to be called XX_GET_PRIV(tw)->priv.
 */
class ParamTypeBase
{
    public:
        double BoxSize;
        MYCUDAFN ParamTypeBase(const double i_BoxSize) : BoxSize(i_BoxSize) {};
};

/* Base class for the TreeWalk queries. You should subclass this and subclass the constructor. */
template <typename ParamType=ParamTypeBase> class TreeWalkQueryBase
{
    public:
        double Pos[3];
        int NodeList[NODELISTLENGTH];
    #ifdef DEBUG
        MyIDType ID;
    #endif

    /**
    * Check if a particle should be processed in this tree walk. Excludes garbage
    * or swallowed particles.
    * Child classes should subclass this to filter particles based on type or flags.
    * Subclasses must call the parent class function to ensure garbage is filtered!
    *
    * @param i  Particle index
    * @return true if the particle should be processed
    */
    static MYCUDAFN bool haswork(const particle_data& part) {
        if(part.IsGarbage || part.Swallowed)
            return false;
        return true;
    }

    /* Constructor:
    * particle_data: particle that is walking the tree.
    * i_NodeList: list of topnodes to start the treewalk from.
    * firstnode is used only if i_NodeList is NULL, in practice this is for primary treewalks.
    * This should be subclassed: the new constructor was called 'fill' in treewalk v1. */
    MYCUDAFN TreeWalkQueryBase(const particle_data& particle, const int * const i_NodeList, const int firstnode, const ParamType& priv) :
    Pos{particle.Pos[0], particle.Pos[1], particle.Pos[2]}, NodeList{firstnode, -1, -1, -1} /* Nodelist is rootnode and terminate immediately */
    #ifdef DEBUG
       , ID(particle.ID)
    #endif
    {
        if(i_NodeList) {
            memcpy(NodeList, i_NodeList, sizeof(i_NodeList[0]) * NODELISTLENGTH);
        }
    }
};

template <typename QueryType, typename OutputType>
class TreeWalkResultBase
{
    public:
    #ifdef DEBUG
        MyIDType ID;
    #endif

        MYCUDAFN TreeWalkResultBase(const QueryType& query)
        #ifdef DEBUG
        : ID(query.ID)
        #endif
        { }

        /**
        * Reduce partial results back to the local particle.
        * Override to accumulate results from tree walk iterations.
        *
        * @param j      Particle index
        * @param mode   Whether this is primary, ghost, or toptree reduction
        */
        template<TreeWalkReduceMode mode>
        MYCUDAFN void reduce(const int j, const OutputType * priv, struct particle_data * const parts)
        {
            #if defined DEBUG && not defined __CUDACC__
                if(parts[j].ID != ID)
                    endrun(2, "Mismatched ID (%ld != %ld) for particle %d in treewalk reduction, mode %d\n", parts[j].ID, ID, j, mode);
            #endif
        }
};

/**
* Cull a node.
*
* Returns 1 if the node shall be opened;
* Returns 0 if the node has no business with this query.
*/
template <NgbTreeFindSymmetric symmetric>
MYCUDAFN int
cull_node(const double * const Pos, const double BoxSize, const MyFloat Hsml, const struct NODE * const current)
{
    double dist;
    if constexpr (symmetric == NGB_TREEFIND_SYMMETRIC) {
        dist = fmax(current->mom.hmax, Hsml) + 0.5 * current->len;
    } else {
        dist = Hsml + 0.5 * current->len;
    }

    double r2 = 0;
    double dx = 0;
    /* do each direction */
    for(int d = 0; d < 3; d ++) {
        dx = NEAREST(current->center[d] - Pos[d], BoxSize);
        if(dx > dist) return 0;
        if(dx < -dist) return 0;
        r2 += dx * dx;
    }
    /* now test against the minimal sphere enclosing everything */
    constexpr double FACT1  = 0.5 *  (1.7320508075688772 - 1.0); /* FACT1 = 0.5 * (sqrt(3)-1) ~ 0.366 */
    dist += FACT1 * current->len;

    if(r2 > dist * dist) {
        return 0;
    }
    return 1;
};

/*!< Thread-local list of the particles to be exported,
 * and the destination tasks. This table allows the
results to be disentangled again and to be
assigned to the correct particle.*/
struct data_index
{
    int Task;
    int Index;
    int NodeList[NODELISTLENGTH];
};

/* Class that stores thread-local information and walks the toptree, finding exports, for a particle. */
template <typename QueryType, typename ParamType, NgbTreeFindSymmetric symmetric>
class TopTreeWalk
{
public:
    /* Constructor from treewalk */
    MYCUDAFN TopTreeWalk(const NODE * const i_Node, const topleaf_data * const i_TopLeaves, const int i_NTopLeaves, const int i_lastnode):
    Nodes(i_Node), TopLeaves(i_TopLeaves), NTopLeaves(i_NTopLeaves), lastnode(i_lastnode), lasttask(0), nodelistindex(0)
    { }

    /* Wrapper of the regular particle visit with some extra cleanup of the particle export table for the toptree walk
     * @param input  Query data for the particle
     * @param output Result accumulator
     * @return the number of nodes used in the dataindex table on success, -1 if export buffer is full
     */
    template <enum TopTreeMode mode>
    MYCUDAFN int toptree_visit(const int target, const QueryType& input, const ParamType& priv, data_index * const DataIndexTable, const size_t BunchSize)
    {
        //message(1, "Starting toptree visit for target %d Nexport %ld\n", target, Nexport);
        /* The number of exports from this particle treewalk. If negative, signals the buffer filled up.*/
        int64_t NThisParticleExport = 0;

        /* Toptree walk always starts from the first node */
        int no = input.NodeList[0];
        const double BoxSize = priv.BoxSize;

        while(no >= 0)
        {
            const NODE * const current = &Nodes[no];
            /* Cull the node */
            if(0 == cull_node<symmetric>(input.Pos, BoxSize, input.Hsml, current)) {
                /* in case the node can be discarded */
                no = current->sibling;
                continue;
            }
            if(current->f.ChildType == PSEUDO_NODE_TYPE) {
                /* Export the pseudo particle*/
                if constexpr(mode == TOPTREE_COUNT)
                    NThisParticleExport = export_count(current->s.suns[0], NThisParticleExport);
                else {
                    NThisParticleExport = export_particle(current->s.suns[0], target, NThisParticleExport, DataIndexTable, BunchSize);
                    /* Exit the loop as we cannot export more particles.*/
                    if(NThisParticleExport < 0)
                        break;
                }
                /* Move sideways*/
                no = current->sibling;
                continue;
            }
            /* Only walk toptree nodes here*/
            if(current->f.TopLevel && !current->f.InternalTopLevel) {
                no = current->sibling;
                continue;
            }
            /* ok, we need to open the node */
            no = current->s.suns[0];
        }
#if defined DEBUG && not defined __CUDACC__
        if(NThisParticleExport > 1000)
            message(5, "%ld exports for particle %d! Odd.\n", NThisParticleExport, target);
#endif
        /* If we filled up, this partial toptree walk will be discarded and the toptree loop exited.*/
        //message(5, "Export buffer full for particle %d with %ld (%lu) exports\n", target, NThisParticleExport, Nexport);
        return NThisParticleExport;
    }

protected:
    /* Adds a remote tree node to the export list for this particle.
    returns -1 if the buffer is full. */
    /* export a particle at target and no, thread safely
     *
     * This can also be called from a nonthreaded code
     *
     * */
    MYCUDAFN int64_t export_particle(const int no, const int target, int64_t nexp, data_index * const DataIndexTable, const int64_t BunchSize)
    {
        //message(1, "Export_particle: no %d target %d exports %ld %lu nodelist %ld\n", no, target, NThisParticleExport, Nexport, nodelistindex);
    #if defined DEBUG && not defined __CUDACC__
        if(no - lastnode > NTopLeaves)
            endrun(1, "Bad export leaf: no = %d lastnode %d ntop %d target %d\n", no, lastnode, NTopLeaves, target);
    #endif
        const topleaf_data * const topleaf = &TopLeaves[no - lastnode];
        const int task = topleaf->Task;
        /* If the last export was to this task, we can perhaps just add this export to the existing NodeList. We can
         * be sure that all exports of this particle are contiguous.*/
        if(nexp >= 1 && lasttask == task) {
    #if defined DEBUG && not defined __CUDACC__
            /* This is just to be safe: only happens if our indices are off.*/
            if(DataIndexTable[nexp - 1].Index != target)
                endrun(1, "Previous of %ld exports is target %d not current %d\n", nexp, DataIndexTable[nexp-1].Index, target);
            if(nodelistindex < NODELISTLENGTH && DataIndexTable[nexp-1].NodeList[nodelistindex] != -1)
                endrun(1, "Current nodelist %ld entry (%d) not empty!\n", nodelistindex, DataIndexTable[nexp-1].NodeList[nodelistindex]);
    #endif
            if(nodelistindex < NODELISTLENGTH) {
                DataIndexTable[nexp-1].NodeList[nodelistindex] = topleaf->treenode;
                nodelistindex++;
                return nexp;
            }
        }
        /* out of buffer space. Need to interrupt. */
        if(nexp >= BunchSize) {
            return -1;
        }
        DataIndexTable[nexp].Task = task;
        DataIndexTable[nexp].Index = target;
        DataIndexTable[nexp].NodeList[0] = topleaf->treenode;
        for(int i = 1; i < NODELISTLENGTH; i++)
            DataIndexTable[nexp].NodeList[i] = -1;
        nodelistindex = 1;
        lasttask = task;
        nexp++;
        return nexp;
    }

    /* Returns 1 if the number of exports is incremented, zero otherwise. */
    MYCUDAFN int64_t export_count(const int no, int64_t nexp)
    {
        //message(1, "Export_particle: no %d target %d exports %ld %lu nodelist %ld\n", no, target, NThisParticleExport, Nexport, nodelistindex);
        const topleaf_data * const topleaf = &TopLeaves[no - lastnode];
        const int task = topleaf->Task;
        /* If the last export was to this task, we can perhaps just add this export to the existing NodeList. We can
         * be sure that all exports of this particle are contiguous.*/
        if(nexp >= 1 && lasttask == task && nodelistindex < NODELISTLENGTH) {
            nodelistindex++;
            return nexp;
        }
        lasttask = task;
        nodelistindex = 1;
        return nexp+1;
    }

    /* A pointer to the force tree structure to walk.*/
    const NODE * const Nodes;
    const topleaf_data * const TopLeaves;
    const int NTopLeaves;
    const int lastnode;
    int lasttask;
    /* Index to use in the current node list*/
    size_t nodelistindex;
};

/* Class that stores thread-local information and walks the local tree for a particle.
 * No exports are made and this should not rely on any external memory, as it may occur on a GPU.
 */
template <typename DerivedType, typename QueryType, typename ResultType, typename ParamType, NgbTreeFindSymmetric symmetric, int mask>
class LocalNgbTreeWalk
{
public:
    /* A pointer to the tree nodes to walk.*/
    const NODE * const Nodes;

    /* Constructor from treewalk */
    MYCUDAFN LocalNgbTreeWalk(const NODE * const Node, const QueryType& input):
     Nodes(Node)
     { }

    static void validate_tree(const ForceTree * const tree)
    {
        if(!force_tree_allocated(tree))
            endrun(0, "Tree has been freed before this treewalk.\n");
        /* Check whether the tree contains the particles we are looking for*/
        if((tree->mask & mask) != mask)
            endrun(5, "Treewalk for particles with mask %d but tree mask is only %d overlap %d.\n", mask, tree->mask, tree->mask & mask);
        /* If symmetric, make sure we did hmax first*/
        if constexpr(symmetric == NGB_TREEFIND_SYMMETRIC)
            if(!tree->hmax_computed_flag)
                endrun(3, "Tried to do a symmetric treewalk without computing hmax!\n");
    }
    /**
     * Visit function - called between a tree node and a particle.
     * Override this to implement custom tree traversal logic.
     *
     * This default is a variant of ngbiter that doesn't use the Ngblist.
     * The ngblist is generally preferred for memory locality reasons.
     * Use this variant if the evaluation
     * wants to change the search radius, such as for knn algorithms
     * or some density code. Don't use it if the treewalk modifies other particles.
     *
     * @param input  Query data for the particle
     * @param output Result accumulator
     * @return number of particle-particle interactions.
     */
     template<TreeWalkReduceMode mode>
     MYCUDAFN int64_t visit(const QueryType& input, ResultType * output, const ParamType& priv, const struct particle_data * const parts)
     {
         static_assert(mode != TREEWALK_TOPTREE, "Toptree should call toptree_visit, not visit.");
         int64_t ninteractions = 0;
         for(int inode = 0; inode < NODELISTLENGTH && input.NodeList[inode] >= 0; inode++)
         {
             int no = input.NodeList[inode];

             while(no >= 0)
             {
                 const struct NODE * const current = &Nodes[no];
                 /* When walking exported particles we start from the encompassing top-level node,
                 * so if we get back to a top-level node again we are done.*/
                 if constexpr(mode == TREEWALK_GHOSTS) {
                     /* The first node is always top-level*/
                     if(current->f.TopLevel && no != input.NodeList[inode]) {
                         /* we reached a top-level node again, which means that we are done with the branch */
                         break;
                     }
                 }

                /* Cull the node */
                if(0 == cull_node<symmetric>(input.Pos, priv.BoxSize, input.Hsml, current)) {
                     /* in case the node can be discarded */
                     no = current->sibling;
                     continue;
                }
                /* Node contains relevant particles, add them.*/
                if(current->f.ChildType == PARTICLE_NODE_TYPE) {
                    for (int i = 0; i < current->s.noccupied; i++) {
                        /* Now evaluate a particle for the list*/
                        const int other = current->s.suns[i];
                        /* Skip garbage*/
                        if(parts[other].IsGarbage)
                            continue;
                        /* In case the type of the particle has changed since the tree was built.
                        * Happens for wind treewalk for gas turned into stars on this timestep.*/
                        if(!((1<<parts[other].Type) & mask))
                            continue;
                        /* Call ngbiter for the child class */
                        static_cast<DerivedType*>(this)->ngbiter(input, parts[other], output, priv);
                        ninteractions++;
                    }
                    /* Move sideways*/
                    no = current->sibling;
                    continue;
                }
                else if(current->f.ChildType == PSEUDO_NODE_TYPE) {
                    /* pseudo particle: this has already been evaluated with the toptree.
                     * Move sideways.*/
                    no = current->sibling;
                    continue;
                }
                /* ok, we need to open the node */
                no = current->s.suns[0];
             }
         }
         return ninteractions;
     }
    /**
    * Neighbour iteration function - called for each particle pair.
    * Override when using ngbiter-based visits.
    *
    * @param input  Query data
    * @param
    * @param output Result accumulator
    * @param iter   Neighbour iterator with distance info
    * @param lv     Thread-local walk state
    */
    MYCUDAFN void ngbiter(const QueryType& input, const particle_data& particle, ResultType * output, const ParamType& priv) {};

    MYCUDAFN double get_distance(const QueryType& input, const particle_data& partother, const double BoxSize, double * dist)
    {
        double r2 = 0;
        for(int d = 0; d < 3; d ++) {
            /* the distance vector points to 'other' */
            dist[d] = NEAREST(input.Pos[d] - partother.Pos[d], BoxSize);
            r2 += dist[d] * dist[d];
        }
        return r2;
    }
};

/* Variant of the local tree walk that uses an Ngblist.
 */
template <typename DerivedType, typename QueryType, typename ResultType, typename ParamType, NgbTreeFindSymmetric symmetric, int mask>
class LocalNgbListTreeWalk : public LocalNgbTreeWalk<DerivedType, QueryType, ResultType, ParamType, symmetric, mask>
{
public:
    /* Constructor from treewalk */
    MYCUDAFN LocalNgbListTreeWalk(const NODE * const Nodes, int * i_ngblist, const QueryType& input):
    LocalNgbTreeWalk<DerivedType, QueryType, ResultType, ParamType, symmetric, mask>(Nodes, input), ngblist(i_ngblist)
    { }
    /**
     * Variant of ngbiter that uses an Ngblist: first it builds a list of
     * particles to evaluate, then it evaluates them.
     * The ngblist is generally preferred for memory locality reasons and
     * to avoid particles being partially evaluated
     * twice if the buffer fills up. Do not use this variant if the evaluation
     * wants to change the search radius, such as for density code.
     * Use this one if the treewalk modifies other particles.
     **/
    template<TreeWalkReduceMode mode>
    MYCUDAFN int64_t visit(const QueryType& input, ResultType * output, const ParamType& priv, const struct particle_data * const parts)
    {
        int64_t ninteractions = 0;
        int inode = 0;

        for(inode = 0; inode < NODELISTLENGTH && input->NodeList[inode] >= 0; inode++)
        {
            int numcand = ngb_treefind_threads<mode>(input, input->NodeList[inode]);
            /* If we are here, export is successful. Work on this particle -- first
             * filter out all of the candidates that are actually outside. */
            int numngb;
            for(numngb = 0; numngb < numcand; numngb ++) {
                const int other = ngblist[numngb];

                /* Skip garbage*/
                if(parts[other].IsGarbage)
                    continue;
                /* In case the type of the particle has changed since the tree was built.
                 * Happens for wind treewalk for gas turned into stars on this timestep.*/
                if(!((1<<parts[other].Type) & mask)) {
                    continue;
                }
                ngbiter(input, other, output, priv, parts);
            }
            ninteractions += numngb;
        }
        return ninteractions;
    }

protected:
    /* List of particles to evaluate */
    int * ngblist;
    /*****
    * This is the internal code that looks for particles in the ngb tree from
    * searchcenter upto hsml. if iter->symmetric is NGB_TREE_FIND_SYMMETRIC, then up to
    * max(part.Hsml, iter->Hsml).
    *
    * Particle that intersects with other domains are marked for export.
    * The hosting nodes (leaves of the global tree) are exported as well.
    *
    * For all 'other' particle within the neighbourhood and are local on this processor,
    * this function calls the ngbiter member of the TreeWalk object.
    * iter->base.other, iter->base.dist iter->base.r2, iter->base.r, are properly initialized.
    *
    * */
    template<TreeWalkReduceMode mode>
    MYCUDAFN int ngb_treefind_threads(const QueryType& input, int startnode)
    {
        int no;
        int numcand = 0;

        const double BoxSize = this->BoxSize;

        no = startnode;

        while(no >= 0)
        {
            struct NODE *current = &this->Nodes[no];
            /* When walking exported particles we start from the encompassing top-level node,
             * so if we get back to a top-level node again we are done.*/
            if constexpr(mode == TREEWALK_GHOSTS) {
                /* The first node is always top-level*/
                if(current->f.TopLevel && no != startnode) {
                    /* we reached a top-level node again, which means that we are done with the branch */
                    break;
                }
            }

            if(0 == cull_node<symmetric>(input.Pos, BoxSize, input.Hsml, current)) {
                /* in case the node can be discarded */
                no = current->sibling;
                continue;
            }

            /* Node contains relevant particles, add them.*/
            if(current->f.ChildType == PARTICLE_NODE_TYPE) {
                int i;
                int * suns = current->s.suns;
                for (i = 0; i < current->s.noccupied; i++) {
                    ngblist[numcand++] = suns[i];
                }
                /* Move sideways*/
                no = current->sibling;
                continue;
            }
            else if(current->f.ChildType == PSEUDO_NODE_TYPE) {
                /* pseudo particle: this has already been evaluated with the toptree. Move sideways.*/
                no = current->sibling;
                continue;
            }
            /* ok, we need to open the node */
            no = current->s.suns[0];
        }

        return numcand;
    }
};
#endif
