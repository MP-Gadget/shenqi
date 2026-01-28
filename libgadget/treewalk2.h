#ifndef _EVALUATOR_H_
#define _EVALUATOR_H_

#include <cstdint>
#include "utils/paramset.h"
#include "forcetree.h"

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

enum TreeWalkType {
    TREEWALK_ACTIVE = 0,
    TREEWALK_ALL,
    TREEWALK_SPLIT,
};

struct TreeWalkQueryBase {
    double Pos[3];
    int NodeList[NODELISTLENGTH];
#ifdef DEBUG
    MyIDType ID;
#endif
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

struct LocalTreeWalk {
    int mode; /* 0 for Primary, 1 for Secondary */
    int target; /* defined only for primary (mode == 0) */

    /* Thread local export variables*/
    size_t Nexport;
    /* Number of entries in the export table for this particle*/
    size_t NThisParticleExport;
    /* Index to use in the current node list*/
    size_t nodelistindex;
    /* Pointer to memory for exports*/
    data_index * DataIndexTable;

    int * ngblist;
    int64_t maxNinteractions;
    int64_t minNinteractions;
    int64_t Ninteractions;
};

/**
 * TreeWalk - Base class for tree-based particle interactions.
 *
 * This class provides the framework for walking a tree structure and
 * computing interactions between particles. Derived classes should override
 * the virtual methods to implement specific physics (e.g., gravity, SPH).
 *
 * Usage:
 *   1. Derive from TreeWalk and override the required virtual methods
 *   2. Set tree, ev_label, type, and element sizes in the constructor
 *   3. Call treewalk_run() to execute the tree walk
 */
class TreeWalk {
public:
    /* A pointer to the force tree structure to walk.*/
    const ForceTree * tree;

    /* name of the evaluator (used in printing messages) */
    const char * ev_label;

    TreeWalkType type;
    int NTask; /*Number of MPI tasks*/
    /* If this is true, the primary and secondary treewalks will be offloaded to an accelerator device (a GPU).
     * This imposes certain limitations, most notably atomics will be slow.*/
    int use_openmp_target;

    size_t query_type_elsize;
    size_t result_type_elsize;
    size_t ngbiter_type_elsize;

    /* Set to true if haswork() is overridden to do actual filtering.
     * Used to optimize queue building when haswork always returns true. */
    bool haswork_defined;

    int64_t NThread; /*Number of OpenMP threads*/

    /* performance metrics */
    /* Wait for remotes to finish.*/
    double timewait1;
    /* Time spent in the toptree*/
    double timecomp0;
    /* This is the time spent in ev_primary*/
    double timecomp1;
    /* This is the time spent in ev_secondary (which may overlap with primary time)*/
    double timecomp2;
    /* Time spent in post-processing and pre-processing*/
    double timecomp3;
    /* Time spent for the reductions.*/
    double timecommsumm;
    /* Number of particles in the Ngblist for the primary treewalk*/
    int64_t Nlistprimary;
    /* Total number of exported particles
     * (Nexport is only the exported particles in the current export buffer). */
    int64_t Nexport_sum;
    /* Number of times we filled up our export buffer*/
    int64_t Nexportfull;
    /* Number of MPI ranks we export to from this rank.*/
    int64_t NExportTargets;
    /* Number of times we needed to re-run the treewalk.
     * Convenience variable for density. */
    int64_t Niteration;
    /* Counters for imbalance diagnostics*/
    int64_t maxNinteractions;
    int64_t minNinteractions;
    int64_t Ninteractions;

    /* internal flags*/
    /* Export counters for each thread*/
    size_t * Nexport_thread;
    /* Information allowing the toptree walk to restart successfully after the export buffer fills up*/
    int * QueueChunkRestart;
    int64_t * QueueChunkEnd;
    /* Pointer to a particle export table for each thread.*/
    data_index ** ExportTable_thread;
    /* Flags that our export buffer is full*/
    int BufferFullFlag;
    /* Number of particles we can fit into the export buffer*/
    size_t BunchSize;
    /* List of neighbour candidates.*/
    int *Ngblist;
    /* Flag not allocating neighbour list*/
    int NoNgblist;
    /*Did we use the active_set array as the WorkSet?*/
    int work_set_stolen_from_active;
    /* Index into WorkSet to start iteration.
     * Will be !=0 if the export buffer fills up*/
    int64_t WorkSetStart;
    /* The list of particles to work on. May be NULL, in which case all particles are used.*/
    int * WorkSet;
    /* Size of the workset list*/
    int64_t WorkSetSize;
    /* Redo counters and queues*/
    size_t *NPLeft;
    int **NPRedo;
    size_t Redo_thread_alloc;
    /* Max and min arrays for each iteration of the count*/
    double * maxnumngb;
    double * minnumngb;

    /**
     * Constructor - initializes all members to safe defaults.
     */
    TreeWalk() :
        tree(nullptr),
        ev_label(nullptr),
        type(TREEWALK_ACTIVE),
        NTask(0),
        use_openmp_target(0),
        query_type_elsize(0),
        result_type_elsize(0),
        ngbiter_type_elsize(0),
        haswork_defined(false),
        NThread(0),
        timewait1(0), timecomp0(0), timecomp1(0), timecomp2(0), timecomp3(0), timecommsumm(0),
        Nlistprimary(0), Nexport_sum(0), Nexportfull(0), NExportTargets(0), Niteration(0),
        maxNinteractions(0), minNinteractions(0), Ninteractions(0),
        Nexport_thread(nullptr), QueueChunkRestart(nullptr), QueueChunkEnd(nullptr),
        ExportTable_thread(nullptr), BufferFullFlag(0), BunchSize(0),
        Ngblist(nullptr), NoNgblist(0), work_set_stolen_from_active(0),
        WorkSetStart(0), WorkSet(nullptr), WorkSetSize(0),
        NPLeft(nullptr), NPRedo(nullptr), Redo_thread_alloc(0),
        maxnumngb(nullptr), minnumngb(nullptr)
    {}

    /**
     * Virtual destructor for proper cleanup in derived classes.
     */
    virtual ~TreeWalk() = default;

    /* This function does treewalk_run in a loop, allocating a queue to allow some particles to be redone.
     * This loop is used primarily in density estimation.*/
    void do_hsml_loop(int * queue, int64_t queuesize, int update_hsml);
    /* Do the distributed tree walking. Warning: as this is a threaded treewalk,
     * it may call tw->visit on particles more than once and in a noneterministic order.
     * Your module should behave correctly in this case! */
    void run(int * active_set, size_t size);

    /* Build the queue from the haswork function, in case we want to reuse it.
     * Arguments:
     * active_set: these items have haswork called on them.
     * size: size of the active set.
     * may_have_garbage: flags whether the active set may contain garbage. If the haswork is trivial and this is not set,
     * we can just reuse the active set as the queue.*/
    void build_queue(int * active_set, const size_t size, int may_have_garbage);

    /* ===== Virtual callback methods ===== */

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
    virtual int visit(TreeWalkQueryBase * input, TreeWalkResultBase * output, LocalTreeWalk * lv);

    /*****
     * Variant of ngbiter that doesn't use the Ngblist.
     * The ngblist is generally preferred for memory locality reasons and
     * to avoid particles being partially evaluated
     * twice if the buffer fills up. Use this variant if the evaluation
     * wants to change the search radius, such as for knn algorithms
     * or some density code. Don't use it if the treewalk modifies other particles.
     * */
    int visit_nolist_ngbiter(TreeWalkQueryBase * I, TreeWalkResultBase * O, LocalTreeWalk * lv);

    /**
     * Check if a particle should be processed in this tree walk.
     * Override to filter particles based on type, flags, etc.
     *
     * @param i  Particle index
     * @return true if the particle should be processed
     */
    virtual bool haswork(const int i) { return true; }

    /**
     * Fill a query structure with particle data.
     * Override to copy relevant particle attributes to the query.
     * NOTE: May be called multiple times (including after reduce),
     * so MUST NOT copy attributes modified by reduce.
     *
     * @param j     Particle index
     * @param query Query structure to fill
     */
    virtual void fill(const int j, TreeWalkQueryBase * query) = 0;

    /**
     * Reduce partial results back to the local particle.
     * Override to accumulate results from tree walk iterations.
     *
     * @param j      Particle index
     * @param result Result data to reduce
     * @param mode   Whether this is primary, ghost, or toptree reduction
     */
    virtual void reduce(const int j, TreeWalkResultBase * result, const TreeWalkReduceMode mode) {}

    /**
     * Neighbour iteration function - called for each particle pair.
     * Override when using ngbiter-based visits.
     *
     * @param input  Query data
     * @param output Result accumulator
     * @param iter   Neighbour iterator with distance info
     * @param lv     Thread-local walk state
     */
    virtual void ngbiter(TreeWalkQueryBase * input, TreeWalkResultBase * output,
                         TreeWalkNgbIterBase * iter, LocalTreeWalk * lv) {}

    /**
     * Postprocess - finalize quantities after tree walk completes.
     * Override to normalize results, compute derived quantities, etc.
     *
     * @param i Particle index
     */
    virtual void postprocess(const int i) {}

    /**
     * Preprocess - initialize quantities before tree walk starts.
     * Override to set up accumulators, clear buffers, etc.
     *
     * @param i Particle index
     */
    virtual void preprocess(const int i) {}

    private:
        int ev_toptree(void);
        void ev_begin(int * active_set, const size_t size);
        void ev_finish(void);
        void ev_primary(void);
        struct CommBuffer ev_secondary(struct CommBuffer * imports, struct ImpExpCounts* counts);
        /*returns -1 if the buffer is full */
        int export_particle(LocalTreeWalk * lv, int no);
        struct ImpExpCounts ev_export_import_counts(MPI_Comm comm);
        void ev_send_recv_export_import(struct ImpExpCounts * counts, struct CommBuffer * exports, struct CommBuffer * imports);
        void ev_recv_export_result(struct CommBuffer * exportbuf, struct ImpExpCounts * counts);
        void ev_reduce_export_result(struct CommBuffer * exportbuf, struct ImpExpCounts * counts);
        int ev_ndone(MPI_Comm comm);
        void alloc_export_memory(void);
        void free_export_memory(void);
        void wait_commbuffer(struct CommBuffer * buffer);
        int ngb_treefind_threads(TreeWalkQueryBase * I,
                TreeWalkNgbIterBase * iter,
                int startnode,
                LocalTreeWalk * lv);


        /* Print some counters for a completed treewalk*/
        void print_stats();

        void init_query(TreeWalkQueryBase * query, int i, const int * const NodeList);
        void init_result(TreeWalkResultBase * result, TreeWalkQueryBase * query);
        void reduce_result(TreeWalkResultBase * result, int i, TreeWalkReduceMode mode);

};

/*Initialise treewalk parameters on first run*/
void set_treewalk_params(ParameterSet * ps);

#define TREEWALK_REDUCE(A, B) (A) = (mode==TREEWALK_PRIMARY)?(B):((A) + (B))

#define MAXITER 400

/* This function find the closest index in the multi-evaluation list of hsml and numNgb, update left and right bound, and return the new hsml */
double ngb_narrow_down(double *right, double *left, const double *radius, const double *numNgb, int maxcmpt, int desnumngb, int *closeidx, double BoxSize);

/* Increment some counters in the ngbiter function*/
void treewalk_add_counters(LocalTreeWalk * lv, const int64_t ninteractions);

/* Change the size of the export buffer, for tests*/
void treewalk_set_max_export_buffer(size_t maxbuf);

#endif
