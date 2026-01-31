#ifndef _LOCALEVALUATOR_H_
#define _LOCALEVALUATOR_H_

#include <cstdint>
#include <cmath>
#include "forcetree.h"
#include "partmanager.h"

/*Initialise treewalk parameters on first run*/
void set_treewalk_params(ParameterSet * ps);

/* Change the size of the export buffer, for tests*/
void treewalk_set_max_export_buffer(const size_t maxbuf);

/* Compute the number of entries that can live in an export table */
size_t compute_bunchsize(const size_t query_type_elsize, const size_t result_type_elsize, const char * const ev_label);

#define TREEWALK_REDUCE(A, B) (A) = (mode==TREEWALK_PRIMARY)?(B):((A) + (B))

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
        const double BoxSize;
        ParamTypeBase(const double i_BoxSize) : BoxSize(i_BoxSize) {};
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
        TreeWalkQueryBase(const particle_data& particle, const int * const i_NodeList, const int firstnode, const ParamType& priv)
        {
        #ifdef DEBUG
            query->ID = particle.ID;
        #endif

            int d;
            for(d = 0; d < 3; d ++) {
                Pos[d] = particle.Pos[d];
            }

            if(i_NodeList) {
                memcpy(NodeList, i_NodeList, sizeof(i_NodeList[0]) * NODELISTLENGTH);
            } else {
                NodeList[0] = firstnode; /* root node */
                NodeList[1] = -1; /* terminate immediately */
            }
        }
};

template <typename ParamType=ParamTypeBase>
class TreeWalkResultBase
{
    public:
        #ifdef DEBUG
            MyIDType ID;
        #endif

        TreeWalkResultBase(const TreeWalkQueryBase<ParamType>& query)
        {
            memset(this, 0, sizeof(*this));
        #ifdef DEBUG
            ID = query.ID;
        #endif
        }

        /**
        * Reduce partial results back to the local particle.
        * Override to accumulate results from tree walk iterations.
        *
        * @param j      Particle index
        * @param mode   Whether this is primary, ghost, or toptree reduction
        */
        void reduce(const int j, const TreeWalkReduceMode mode, const ParamType& priv, struct particle_data * const parts)
        {
            #ifdef DEBUG
                if(parts[j].ID != ID)
                    endrun(2, "Mismatched ID (%ld != %ld) for particle %d in treewalk reduction, mode %d\n", parts[j].ID, ID, j, mode);
            #endif
        }

};

template <typename QueryType, typename ResultType, typename ParamType=ParamTypeBase>
class TreeWalkNgbIterBase {
    public:
        const int mask;
        const NgbTreeFindSymmetric symmetric;
        const double Hsml;
        double dist[3];
        double r2;
        double r;
        int other;

        TreeWalkNgbIterBase(const int i_mask, const NgbTreeFindSymmetric i_symmetric, const QueryType& input) :
        mask(i_mask), symmetric(i_symmetric), Hsml(input.Hsml) {};
        /**
         * Neighbour iteration function - called for each particle pair.
         * Override when using ngbiter-based visits.
         *
         * @param input  Query data
         * @param output Result accumulator
         * @param iter   Neighbour iterator with distance info
         * @param lv     Thread-local walk state
         */
        void ngbiter(const QueryType& input, const int i_other, ResultType * output, const ParamType& priv, const struct particle_data * const parts)
        {
            const particle_data& particle = parts[other];
            double symHsml = Hsml;
            if(symmetric == NGB_TREEFIND_SYMMETRIC) {
                symHsml = DMAX(particle.Hsml, Hsml);
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
            /* update the iter and call the iteration function*/
            r = sqrt(r2);
            other = i_other;
        };
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

template <typename NgbIterType, typename QueryType, typename ResultType, typename ParamType>
class LocalTreeWalk
{
public:

    const int mode; /* 0 for Primary, 1 for Secondary */
    int target; /* Current particle, defined only for primary and toptree walks (mode == TREEWALK_PRIMARY and TREEWALK_TOPTREE) */
    /* Interaction counters */
    int64_t maxNinteractions;
    int64_t minNinteractions;
    int64_t Ninteractions;
    /* Current number of exports from this chunk*/
    size_t Nexport;

    /* Constructor from treewalk */
    LocalTreeWalk(const int i_mode, const ForceTree * const i_tree, const char * const i_ev_label, int * Ngblist, data_index ** ExportTable_thread);

    /**
     * Visit function - called between a tree node and a particle.
     * Override this to implement custom tree traversal logic.
     * Default implementation calls treewalk_visit_ngbiter.
     *
     * @param input  Query data for the particle
     * @param output Result accumulator
     * @return 0 on success, -1 if export buffer is full
     */
     int visit(const QueryType& input, ResultType * output, const ParamType& priv, const struct particle_data * const parts);

    /* Wrapper of the regular particle visit with some extra cleanup of the particle export table for the toptree walk
     * @param input  Query data for the particle
     * @param output Result accumulator
     * @return 0 on success, -1 if export buffer is full
     */
    int toptree_visit(const QueryType& input, ResultType * output, const ParamType& priv, const struct particle_data * const parts);

    /*****
     * Variant of ngbiter that uses an Ngblist: first it builds a list of
     * particles to evaluate, then it evaluates them.
     * The ngblist is generally preferred for memory locality reasons and
     * to avoid particles being partially evaluated
     * twice if the buffer fills up. Do not use this variant if the evaluation
     * wants to change the search radius, such as for density code.
     * Use this one if the treewalk modifies other particles.
     * */
    int visit_ngblist(const QueryType& input, ResultType * output, const ParamType& priv, const struct particle_data * const parts);

protected:
    int ngb_treefind_threads(const QueryType& input, NgbIterType * iter, int startnode);

    /* Adds a remote tree node to the export list for this particle.
    returns -1 if the buffer is full. */
    int export_particle(const int no);

    void
    treewalk_add_counters(const int64_t ninteractions)
    {
        if(maxNinteractions < ninteractions)
            maxNinteractions = ninteractions;
        if(minNinteractions > ninteractions)
            minNinteractions = ninteractions;
        Ninteractions += ninteractions;
    }

    /* A pointer to the force tree structure to walk.*/
    const ForceTree * const tree;

    int * ngblist;

    /* name of the evaluator (used in printing messages) */
    const char * ev_label;

    /* Number of entries in the export table for this particle*/
    size_t NThisParticleExport;
    /* Index to use in the current node list*/
    size_t nodelistindex;
    /* Pointer to memory for exports*/
    data_index * DataIndexTable;
    /* Number of particles we can fit into the export buffer*/
    const size_t BunchSize;
};

#endif
