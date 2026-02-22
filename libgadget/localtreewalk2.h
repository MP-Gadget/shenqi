#ifndef _LOCALEVALUATOR_H_
#define _LOCALEVALUATOR_H_

#include <stdint.h>
#include <math.h>
#include <omp.h>
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
        MYCUDAFN void reduce(const int j, const OutputType& priv, struct particle_data * const parts)
        {
            #ifdef DEBUG
                if(parts[j].ID != ID)
                    endrun(2, "Mismatched ID (%ld != %ld) for particle %d in treewalk reduction, mode %d\n", parts[j].ID, ID, j, mode);
            #endif
        }
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
    /* Current number of exports from this chunk*/
    size_t Nexport;

    /* Constructor from treewalk */
    TopTreeWalk(const ForceTree * const i_tree, const size_t i_BunchSize, data_index * const ExportTable_thread):
    Nexport(0), tree(i_tree), BunchSize(i_BunchSize), NThisParticleExport(0), nodelistindex(0), DataIndexTable(ExportTable_thread)
    { }

    /* Wrapper of the regular particle visit with some extra cleanup of the particle export table for the toptree walk
     * @param input  Query data for the particle
     * @param output Result accumulator
     * @return 0 on success, -1 if export buffer is full
     */
    int toptree_visit(const int target, const QueryType& input, const ParamType& priv, const struct particle_data * const parts)
    {
        //message(1, "Starting toptree visit for target %d Nexport %ld\n", target, Nexport);
        /* Reset the number of exported particles.*/
        NThisParticleExport = 0;

        /* Flags if the particle export failed. */
        int export_failed = 0;
        /* Toptree walk always starts from the first node */
        int no = tree->firstnode;
        const double BoxSize = tree->BoxSize;

        while(no >= 0)
        {
            struct NODE *current = &tree->Nodes[no];
            /* Cull the node */
            if(0 == cull_node<symmetric>(input.Pos, BoxSize, input.Hsml, current)) {
                /* in case the node can be discarded */
                no = current->sibling;
                continue;
            }
            if(current->f.ChildType == PSEUDO_NODE_TYPE) {
                /* Export the pseudo particle*/
                export_failed = export_particle(current->s.suns[0], target);
                /* Exit the loop as we cannot export more particles.*/
                if(export_failed != 0)
                    break;
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
        if(NThisParticleExport > 1000)
            message(5, "%ld exports for particle %d! Odd.\n", NThisParticleExport, target);
        /* If we filled up, we need to remove the partially evaluated last particle from the export list,
        * save the partially evaluated chunk, and leave this loop.*/
        if(export_failed != 0) {
            //message(5, "Export buffer full for particle %d with %ld (%lu) exports\n", target, NThisParticleExport, Nexport);
            /* Drop partial exports on the current particle, whose toptree will be re-evaluated*/
            Nexport -= NThisParticleExport;
            /* Check that the final export in the list is indeed from a different particle*/
            if(NThisParticleExport > 0 && DataIndexTable[Nexport-1].Index >= target)
                endrun(5, "Something screwed up in export queue: nexp %ld (local %ld) last %d < index %d\n", Nexport,
                    NThisParticleExport, target, DataIndexTable[Nexport-1].Index);
            /* Check that the earliest dropped export in the list is from the same particle*/
            if(NThisParticleExport > 0 && DataIndexTable[Nexport].Index != target)
                endrun(5, "Something screwed up in export queue: nexp %ld (local %ld) last %d != index %d\n", Nexport,
                    NThisParticleExport, target, DataIndexTable[Nexport].Index);
        }
        return export_failed;
    }

protected:
    /* Adds a remote tree node to the export list for this particle.
    returns -1 if the buffer is full. */
    /* export a particle at target and no, thread safely
     *
     * This can also be called from a nonthreaded code
     *
     * */
    int export_particle(const int no, const int target)
    {
        //message(1, "Export_particle: no %d target %d exports %ld %lu nodelist %ld\n", no, target, NThisParticleExport, Nexport, nodelistindex);
        if(no < tree->lastnode) {
            endrun(1, "Called export on a non-pseudo node %d < %ld.\n", no, tree->lastnode);
        }
        if(!DataIndexTable)
            endrun(1, "DataIndexTable not allocated\n");
        if(no - tree->lastnode > tree->NTopLeaves)
            endrun(1, "Bad export leaf: no = %d lastnode %ld ntop %d target %d\n", no, tree->lastnode, tree->NTopLeaves, target);
        const int task = tree->TopLeaves[no - tree->lastnode].Task;
        /* This index is a unique entry in the global DataIndexTable.*/
        size_t nexp = Nexport;
        /* If the last export was to this task, we can perhaps just add this export to the existing NodeList. We can
         * be sure that all exports of this particle are contiguous.*/
        if(NThisParticleExport >= 1 && DataIndexTable[nexp-1].Task == task) {
    #ifdef DEBUG
            /* This is just to be safe: only happens if our indices are off.*/
            if(DataIndexTable[nexp - 1].Index != target)
                endrun(1, "Previous of %ld (%lu) exports is target %d not current %d\n", NThisParticleExport, nexp, DataIndexTable[nexp-1].Index, target);
    #endif
            if(nodelistindex < NODELISTLENGTH) {
    #ifdef DEBUG
                if(DataIndexTable[nexp-1].NodeList[nodelistindex] != -1)
                    endrun(1, "Current nodelist %ld entry (%d) not empty!\n", nodelistindex, DataIndexTable[nexp-1].NodeList[nodelistindex]);
    #endif
                DataIndexTable[nexp-1].NodeList[nodelistindex] = tree->TopLeaves[no - tree->lastnode].treenode;
                nodelistindex++;
                return 0;
            }
        }
        /* out of buffer space. Need to interrupt. */
        if(Nexport >= BunchSize) {
            return -1;
        }
        DataIndexTable[nexp].Task = task;
        DataIndexTable[nexp].Index = target;
        DataIndexTable[nexp].NodeList[0] = tree->TopLeaves[no - tree->lastnode].treenode;
        int i;
        for(i = 1; i < NODELISTLENGTH; i++)
            DataIndexTable[nexp].NodeList[i] = -1;
        Nexport++;
        nodelistindex = 1;
        NThisParticleExport++;
        return 0;
    }

    /* A pointer to the force tree structure to walk.*/
    const ForceTree * const tree;
    /* Number of particles we can fit into the export buffer*/
    const size_t BunchSize;
    /* Number of entries in the export table for this particle*/
    size_t NThisParticleExport;
    /* Index to use in the current node list*/
    size_t nodelistindex;
    /* Pointer to memory for exports*/
    data_index * const DataIndexTable;
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
        dist = DMAX(current->mom.hmax, Hsml) + 0.5 * current->len;
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

/* Class that stores thread-local information and walks the local tree for a particle.
 * No exports are made and this should not rely on any external memory, as it may occur on a GPU.
 */
template <typename DerivedType, typename QueryType, typename ResultType, typename ParamType, NgbTreeFindSymmetric symmetric, int mask>
class LocalNgbTreeWalk
{
public:
    /* A pointer to the force tree structure to walk.*/
    const ForceTree * const tree;
    double dist[3];
    double r2;
    /* Constructor from treewalk */
    MYCUDAFN LocalNgbTreeWalk(const ForceTree * const i_tree, const QueryType& input):
     tree(i_tree)
    { }
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
             const double BoxSize = tree->BoxSize;

             while(no >= 0)
             {
                 const struct NODE * const current = &tree->Nodes[no];

                 /* When walking exported particles we start from the encompassing top-level node,
                 * so if we get back to a top-level node again we are done.*/
                 if constexpr(mode == TREEWALK_GHOSTS) {
                     /* The first node is always top-level*/
                     if(no > tree->lastnode)
                         endrun(7, "Node is after lastnode. no %d lastnode %ld start %d first %ld\n", no, tree->lastnode, input.NodeList[inode], tree->firstnode);
                     if(current->f.TopLevel && no != input.NodeList[inode]) {
                         /* we reached a top-level node again, which means that we are done with the branch */
                         break;
                     }
                 }

                /* Cull the node */
                if(0 == cull_node<symmetric>(input.Pos, BoxSize, input.Hsml, current)) {
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
                        ngbiter(input, other, output, priv, parts);
                        ninteractions++;
                    }
                    /* Move sideways*/
                    no = current->sibling;
                    continue;
                }
                else if(current->f.ChildType == PSEUDO_NODE_TYPE) {
                    /* pseudo particle */
                    if constexpr(mode == TREEWALK_GHOSTS)
                        endrun(12312, "Secondary for particle from node %d found pseudo at %d.\n", input.NodeList[inode], no);
                    /* This has already been evaluated with the toptree. Move sideways.*/
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
    * @param output Result accumulator
    * @param iter   Neighbour iterator with distance info
    * @param lv     Thread-local walk state
    */
    MYCUDAFN void ngbiter(const QueryType& input, const int other, ResultType * output, const ParamType& priv, const struct particle_data * const parts)
    {
        const particle_data& particle = parts[other];
        double symHsml = input.Hsml;
        if constexpr(symmetric == NGB_TREEFIND_SYMMETRIC) {
            symHsml = DMAX(particle.Hsml, input.Hsml);
        }

        r2 = 0;
        int d;
        double h2 = symHsml * symHsml;
        for(d = 0; d < 3; d ++) {
            /* the distance vector points to 'other' */
            dist[d] = NEAREST(input.Pos[d] - particle.Pos[d], priv.BoxSize);
            r2 += dist[d] * dist[d];
            if(r2 > h2) break;
        }
        /* Call ngbiter for the child class */
        static_cast<DerivedType*>(this)->ngbiter(input, other, output, priv, parts);
    }
};

/* Variant of the local tree walk that uses an Ngblist.
 */
template <typename DerivedType, typename QueryType, typename ResultType, typename ParamType, NgbTreeFindSymmetric symmetric, int mask>
class LocalNgbListTreeWalk : public LocalNgbTreeWalk<DerivedType, QueryType, ResultType, ParamType, symmetric, mask>
{
public:
    /* Constructor from treewalk */
    MYCUDAFN LocalNgbListTreeWalk(const ForceTree * const i_tree, int * i_ngblist, const QueryType& input):
    LocalNgbTreeWalk<DerivedType, QueryType, ResultType, ParamType, symmetric, mask>(i_tree, input), ngblist(i_ngblist)
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
        /* Check whether the tree contains the particles we are looking for*/
        if((this->tree->mask & mask) != mask)
            endrun(5, "Treewalk for particles with mask %d but tree mask is only %d overlap %d.\n", mask, this->tree->mask, this->tree->mask & mask);
        /* If symmetric, make sure we did hmax first*/
        if constexpr(symmetric == NGB_TREEFIND_SYMMETRIC)
            if(!this->tree->hmax_computed_flag)
                endrun(3, "Tried to do a symmetric treewalk without computing hmax!\n");
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

        const double BoxSize = this->tree->BoxSize;

        no = startnode;

        while(no >= 0)
        {
            if(node_is_particle(no, this->tree)) {
                int fat = force_get_father(no, this->tree);
                endrun(12312, "Particles should be added before getting here! no = %d, father = %d (ptype = %d) start=%d mode = %d\n", no, fat, this->tree->Nodes[fat].f.ChildType, startnode, mode);
            }
            if(node_is_pseudo_particle(no, this->tree)) {
                int fat = force_get_father(no, this->tree);
                endrun(12312, "Pseudo-Particles should be added before getting here! no = %d, father = %d (ptype = %d)\n", no, fat, this->tree->Nodes[fat].f.ChildType);
            }

            struct NODE *current = &this->tree->Nodes[no];

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
                /* pseudo particle */
                if constexpr(mode == TREEWALK_GHOSTS) {
                    endrun(12312, "Secondary for nodelist %d found pseudo at %d.\n", startnode, no);
                } else {
                    /* This has already been evaluated with the toptree. Move sideways.*/
                    no = current->sibling;
                    continue;
                }
            }
            /* ok, we need to open the node */
            no = current->s.suns[0];
        }

        return numcand;
    }
};
#endif
