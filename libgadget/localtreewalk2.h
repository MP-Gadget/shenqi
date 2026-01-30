#ifndef _LOCALEVALUATOR_H_
#define _LOCALEVALUATOR_H_

#include <cstdint>
#include "forcetree.h"
#include "libgadget/partmanager.h"

/*Initialise treewalk parameters on first run*/
void set_treewalk_params(ParameterSet * ps);

/* Change the size of the export buffer, for tests*/
void treewalk_set_max_export_buffer(size_t maxbuf);

/* Compute the number of entries that can live in an export table */
size_t compute_bunchsize(const size_t query_type_elsize, const size_t result_type_elsize, const char * const ev_label);

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

/* Base class for the TreeWalk queries. You should subclass this and subclass the constructor. */
class TreeWalkQueryBase
{
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
    TreeWalkQueryBase(const particle_data& particle, const int * const i_NodeList, int firstnode)
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

struct TreeWalkResultBase {
#ifdef DEBUG
    MyIDType ID;
#endif
};

struct TreeWalkNgbIterBase {
    int mask;
    int other;
    double Hsml;
    double dist[3];
    double r2;
    double r;
    NgbTreeFindSymmetric symmetric;
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

template <typename NgbIterType, typename QueryType, typename ResultType>
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
     * Neighbour iteration function - called for each particle pair.
     * Override when using ngbiter-based visits.
     *
     * @param input  Query data
     * @param output Result accumulator
     * @param iter   Neighbour iterator with distance info
     * @param lv     Thread-local walk state
     */
    void ngbiter(QueryType * input, ResultType * output, NgbIterType iter) {};

    /**
     * Visit function - called between a tree node and a particle.
     * Override this to implement custom tree traversal logic.
     * Default implementation calls treewalk_visit_ngbiter.
     *
     * @param input  Query data for the particle
     * @param output Result accumulator
     * @param lv     Thread-local walk state
     * @return 0 on success, -1 if export buffer is full
     */
    int visit(QueryType * input, ResultType * output);

    /* Wrapper of the regular particle visit with some extra cleanup of the particle export table for the toptree walk */
    int toptree_visit(QueryType * input, ResultType * output);

    /*****
     * Variant of ngbiter that doesn't use the Ngblist.
     * The ngblist is generally preferred for memory locality reasons and
     * to avoid particles being partially evaluated
     * twice if the buffer fills up. Use this variant if the evaluation
     * wants to change the search radius, such as for knn algorithms
     * or some density code. Don't use it if the treewalk modifies other particles.
     * */
    int visit_nolist_ngbiter(QueryType * input, ResultType * output);

private:

    int ngb_treefind_threads(QueryType * I, NgbIterType * iter, int startnode);

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
