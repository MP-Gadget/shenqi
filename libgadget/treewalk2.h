#ifndef _TREEWALK2_H_
#define _TREEWALK2_H_

#include <cstdint>
#include <omp.h>
#include "localtreewalk2.h"
#include "forcetree.h"
#include "utils/mymalloc.h"
#include "utils/endrun.h"
#include "utils/system.h"

#define MAXITER 400

struct ImpExpCounts
{
    int64_t * Export_count;
    int64_t * Import_count;
    int64_t * Export_offset;
    int64_t * Import_offset;
    MPI_Comm comm;
    int NTask;
    /* Number of particles exported to this processor*/
    size_t Nimport;
    /* Number of particles exported from this processor*/
    size_t Nexport;
};

void free_impexpcount(struct ImpExpCounts * count)
{
    ta_free(count->Export_count);
}

#define COMM_RECV 1
#define COMM_SEND 0

/* This class wraps MPI_requests for the distributed part of the treewalk.
 * It allocates an array of requests, one for each task, and an array with the task index.
 * The requests are populated from the export lists in MPI_fill and MPI_wait waits for
 * all requests to be completed.
 *
 * The destructor waits() and then frees the memory.
 */
class CommBuffer
{
    public:
    char * databuf;
    int * rqst_task;
    MPI_Request * rdata_all;
    int nrequest_all;
    CommBuffer(const int NTask, const int alloc_high): databuf(NULL), nrequest_all(0)
    {
        if(alloc_high) {
            rdata_all = ta_malloc2("requests", MPI_Request, NTask);
            rqst_task = ta_malloc2("rqst", int, NTask);
        }
        else {
            rdata_all = ta_malloc("requests", MPI_Request, NTask);
            rqst_task = ta_malloc("rqst", int, NTask);
        }
    }
    ~CommBuffer()
    {
        /* First wait until all comms are done */
        wait();
        if(databuf) {
            myfree(databuf);
            databuf = NULL;
        }
        ta_free(rqst_task);
        ta_free(rdata_all);
    }

    /* Routine to send data to all tasks async. If receive is set, the routine receives data. The structure stores the requests.
     Empty tasks are skipped. Must call alloc_commbuffer on the buffer first and buffer->databuf must be set.*/
    void MPI_fill(int64_t *cnts, int64_t *displs, MPI_Datatype type, int receive, int tag, MPI_Comm comm)
    {
        int ThisTask;
        int NTask;
        MPI_Comm_rank(comm, &ThisTask);
        MPI_Comm_size(comm, &NTask);
        ptrdiff_t lb, elsize;
        MPI_Type_get_extent(type, &lb, &elsize);
        int nrequests = 0;

        int i;
        /* Loop over all tasks, starting with the one just past this one*/
        for(i = 1; i < NTask; i++)
        {
            int target = (ThisTask + i) % NTask;
            if(cnts[target] == 0) continue;
            rqst_task[nrequests] = target;
            if(receive == COMM_RECV) {
                MPI_Irecv(((char*) databuf) + elsize * displs[target], cnts[target],
                    type, target, tag, comm, &rdata_all[nrequests++]);
            }
            else {
                MPI_Isend(((char*) databuf) + elsize * displs[target], cnts[target],
                    type, target, tag, comm, &rdata_all[nrequests++]);
            }
        }
        nrequest_all = nrequests;
    }

    /* Waits for all the requests in the commbuffer to be complete*/
    void wait(void)
    {
        MPI_Waitall(nrequest_all, rdata_all, MPI_STATUSES_IGNORE);
    }
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
template <typename QueryType, typename ResultType, typename LocalTreeWalkType, typename ParamType>
class TreeWalk {
public:
    /* A pointer to the force tree structure to walk.*/
    const ForceTree * const tree;

    /* name of the evaluator (used in printing messages) */
    const char * const ev_label;

    const ParamType priv;
    int NTask; /*Number of MPI tasks*/
    /* Set to true if haswork() is overridden to do actual filtering.
     * Used to optimize queue building when haswork always returns true. */
    bool should_rebuild_queue;
    const int64_t NThread; /*Number of OpenMP threads*/
    /* If this is true, the primary and secondary treewalks will be offloaded to an accelerator device (a GPU).
     * This imposes certain limitations, most notably atomics will be slow.*/
    const int use_openmp_target;

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
    TreeWalk(const char * const i_ev_label, const ForceTree * const i_tree, const ParamType& i_priv, bool i_should_rebuild_queue=true) :
        tree(i_tree), ev_label(i_ev_label),
        priv(i_priv),
        should_rebuild_queue(i_should_rebuild_queue),
        NThread(omp_get_max_threads()),
        use_openmp_target(0),
        timewait1(0), timecomp0(0), timecomp1(0), timecomp2(0), timecomp3(0), timecommsumm(0),
        Nlistprimary(0), Nexport_sum(0), Nexportfull(0), NExportTargets(0), Niteration(0),
        maxNinteractions(0), minNinteractions(0), Ninteractions(0),
        Nexport_thread(nullptr), QueueChunkRestart(nullptr), QueueChunkEnd(nullptr),
        ExportTable_thread(nullptr), BufferFullFlag(0),
        Ngblist(nullptr), NoNgblist(0), work_set_stolen_from_active(0),
        WorkSetStart(0), WorkSet(nullptr), WorkSetSize(0),
        NPLeft(nullptr), NPRedo(nullptr), Redo_thread_alloc(0),
        maxnumngb(nullptr), minnumngb(nullptr)
    {
        MPI_Comm_size(MPI_COMM_WORLD, &NTask);
    }

    /* Do the distributed tree walking. Warning: as this is a threaded treewalk,
     * it may call tw->visit on particles more than once and in a noneterministic order.
     * Your module should behave correctly in this case!
     *
     * active_set : a list of indices of particles that walk the tree. If active_set is NULL,
     *              all (NumPart) particles are used. This is not the list of particles
     * in the tree, but the particles that do the walking.
     * size: length of the active set
     * particle_data parts: list of particles to use
     */
    void run(int * active_set, size_t size, particle_data * const parts)
    {
        if(!force_tree_allocated(tree)) {
            endrun(0, "Tree has been freed before this treewalk.\n");
        }

        double tstart, tend;
    #ifdef DEBUG
        GDB_current_ev = tw;
    #endif

        tstart = second();
        ev_begin(active_set, size);

        int64_t i;
        #pragma omp parallel for
        for(i = 0; i < WorkSetSize; i ++) {
            const int p_i = WorkSet ? WorkSet[i] : i;
            preprocess(p_i, parts);
        }

        tend = second();
        timecomp3 += timediff(tstart, tend);

        Nexportfull = 0;
        Nexport_sum = 0;
        Ninteractions = 0;
        int Ndone = 0;
        /* Needs to be outside loop because it allocates restart information*/
        alloc_export_memory();
        do
        {
            tstart = second();
            /* First do the toptree and export particles for sending.*/
            ev_toptree();
            /* All processes sync via alltoall.*/
            struct ImpExpCounts counts = ev_export_import_counts(MPI_COMM_WORLD);
            Ndone = ev_ndone(MPI_COMM_WORLD);
            /* Send the exported particle data */
            /* exports is allocated first, then imports*/
            CommBuffer exports(counts.NTask, 0);
            auto imports = new CommBuffer(counts.NTask, 0);

            ev_send_recv_export_import(&counts, &exports, &imports);
            tend = second();
            timecomp0 += timediff(tstart, tend);
            /* Only do this on the first iteration, as we only need to do it once.*/
            tstart = second();
            if(Nexportfull == 0)
                ev_primary(parts); /* do local particles and prepare export list */
            tend = second();
            timecomp1 += timediff(tstart, tend);
            /* Do processing of received particles. We implement a queue that
                * checks each incoming task in turn and processes them as they arrive.*/
            tstart = second();
            /* Posts recvs to get the export results (which are sent in ev_secondary).*/
            CommBuffer res_exports(counts.NTask, 1);
            ev_recv_export_result(&res_exports, &counts);
            CommBuffer res_imports(counts.NTask, 1);
            ev_secondary(&res_imports, &imports, &counts);
            // report_memory_usage(ev_label);
            // Want to explicitly run the destructor for this one so we free memory early.
            delete imports;
            tend = second();
            timecomp2 += timediff(tstart, tend);
            /* Now clear the sent data buffer, waiting for the send to complete.
                * This needs to be after the other end has called recv.*/
            tstart = second();
            res_exports.wait();
            tend = second();
            timewait1 += timediff(tstart, tend);
            tstart = second();
            ev_reduce_export_result(&res_exports, &counts);
            tend = second();
            timecommsumm += timediff(tstart, tend);
            /* Free export memory*/
            Nexportfull++;
            /* The destructors for the CommBuffers will fire at this point,
             * which means there is an implicit wait() */
        } while(Ndone < NTask);
        free_export_memory();

        tstart = second();
        #pragma omp parallel for
        for(i = 0; i < WorkSetSize; i ++) {
            const int p_i = WorkSet ? WorkSet[i] : i;
            postprocess(p_i, parts);
        }
        tend = second();
        timecomp3 += timediff(tstart, tend);
        ev_finish();
        Niteration++;
    }

    /* This function does treewalk_run in a loop, allocating a queue to allow some particles to be redone.
     * This loop is used primarily in density estimation.*/
    void do_hsml_loop(int * queue, int64_t queuesize, const int update_hsml, particle_data * parts)
    {
        int NumThreads = omp_get_max_threads();
        maxnumngb = ta_malloc("numngb", double, NumThreads);
        minnumngb = ta_malloc("numngb2", double, NumThreads);

        /* Build the first queue */
        double tstart = second();
        build_queue(queue, queuesize, 0, parts);
        double tend = second();

        /* Next call to treewalk_run will over-write these pointers*/
        int64_t size = WorkSetSize;
        int * ReDoQueue = WorkSet;
        /* First queue is allocated low*/
        int alloc_high = 0;
        /* We don't need to redo the queue generation
         * but need to keep track of allocated memory.*/
        bool orig_build_queue = should_rebuild_queue;
        should_rebuild_queue = false;
        timecomp3 += timediff(tstart, tend);
        /* we will repeat the whole thing for those particles where we didn't find enough neighbours */
        do {
            /* The RedoQueue needs enough memory to store every workset particle on every thread, because
             * we cannot guarantee that the sph particles are evenly spread across threads!*/
            int * CurQueue = ReDoQueue;
            int i;
            for(i = 0; i < NumThreads; i++) {
                maxnumngb[i] = 0;
                minnumngb[i] = 1e50;
            }
            /* The ReDoQueue swaps between high and low allocations so we can have two allocated alternately*/
            if(!alloc_high)
                alloc_high = 1;
            else
                alloc_high = 0;
            gadget_thread_arrays loop = gadget_setup_thread_arrays("ReDoQueue", alloc_high, size);
            NPRedo = loop.srcs;
            NPLeft = loop.sizes;
            Redo_thread_alloc = loop.total_size;
            run(CurQueue, size, parts);

            /* Now done with the current queue*/
            if(orig_build_queue || Niteration > 1)
                myfree(CurQueue);

            size = gadget_compact_thread_arrays(&ReDoQueue, &loop);
            /* We can stop if we are not updating hsml or if we are done.*/
            if(!update_hsml || !MPIU_Any(size > 0, MPI_COMM_WORLD)) {
                myfree(ReDoQueue);
                break;
            }
            for(i = 1; i < NumThreads; i++) {
                if(maxnumngb[0] < maxnumngb[i])
                    maxnumngb[0] = maxnumngb[i];
                if(minnumngb[0] > minnumngb[i])
                    minnumngb[0] = minnumngb[i];
            }
            double minngb, maxngb;
            MPI_Reduce(&maxnumngb[0], &maxngb, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            MPI_Reduce(&minnumngb[0], &minngb, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
            message(0, "Max ngb=%g, min ngb=%g\n", maxngb, minngb);
    #ifdef DEBUG
            print_stats();
    #endif

            /*Shrink memory*/
            ReDoQueue = (int *) myrealloc(ReDoQueue, sizeof(int) * size);
    #ifdef DEBUG
            if(size < 10 && Niteration > 20 ) {
                int pp = ReDoQueue[0];
                message(1, "Remaining i=%d, t %d, pos %g %g %g, hsml: %g\n", pp, parts[pp].Type, parts[pp].Pos[0], parts[pp].Pos[1], parts[pp].Pos[2], parts[pp].Hsml);
            }
    #endif

            if(size > 0 && Niteration > MAXITER) {
                endrun(1155, "failed to converge density for %ld particles\n", size);
            }
        } while(1);
        ta_free(minnumngb);
        ta_free(maxnumngb);
    }

    /* Build the queue from the haswork function, in case we want to reuse it.
     * Arguments:
     * active_set: these items have haswork called on them.
     * size: size of the active set.
     * may_have_garbage: flags whether the active set may contain garbage. If the haswork is trivial and this is not set,
     * we can just reuse the active set as the queue.*/
    void build_queue(int * active_set, const size_t size, int may_have_garbage, const particle_data * const parts);

    /* Print some counters for a completed treewalk*/
    void print_stats(void)
    {
        int64_t o_NExportTargets;
        int64_t o_minNinteractions, o_maxNinteractions, o_Ninteractions, o_Nlistprimary, Nexport;
        MPI_Reduce(&minNinteractions, &o_minNinteractions, 1, MPI_INT64, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&maxNinteractions, &o_maxNinteractions, 1, MPI_INT64, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&Ninteractions, &o_Ninteractions, 1, MPI_INT64, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&Nlistprimary, &o_Nlistprimary, 1, MPI_INT64, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&Nexport_sum, &Nexport, 1, MPI_INT64, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&NExportTargets, &o_NExportTargets, 1, MPI_INT64, MPI_SUM, 0, MPI_COMM_WORLD);
        message(0, "%s Ngblist: min %ld max %ld avg %g average exports: %g avg target ranks: %g\n", ev_label, o_minNinteractions, o_maxNinteractions,
                (double) o_Ninteractions / o_Nlistprimary, ((double) Nexport)/ NTask, ((double) o_NExportTargets)/ NTask);
    }

    private:
        /**
        * Check if a particle should be processed in this tree walk.
        * Override to filter particles based on type, flags, etc.
        *
        * @param i  Particle index
        * @return true if the particle should be processed
        */
        bool haswork(const particle_data& part) { return true; }

        /**
        * Postprocess - finalize quantities after tree walk completes.
        * Override to normalize results, compute derived quantities, etc.
        *
        * @param i Particle index
        */
        void postprocess(const int i, particle_data * const part) {}

        /**
        * Preprocess - initialize quantities before tree walk starts.
        * Override to set up accumulators, clear buffers, etc.
        *
        * @param i Particle index
        */
        void preprocess(const int i, particle_data * const part) {}

        int ev_toptree(const particle_data * const parts);
        void ev_begin(int * active_set, const size_t size, particle_data * const parts);

        /* Cleans up and frees memory */
        void ev_finish(void)
        {
            if(Ngblist)
                myfree(Ngblist);
            if(!work_set_stolen_from_active)
                myfree(WorkSet);
        }

        void ev_primary(const particle_data * const parts);
        void ev_secondary(CommBuffer * res_imports, CommBuffer * imports, struct ImpExpCounts * counts, const struct particle_data * const parts);
        struct ImpExpCounts ev_export_import_counts(MPI_Comm comm);

        void ev_send_recv_export_import(struct ImpExpCounts * counts, CommBuffer * exports, CommBuffer * imports, const particle_data * const parts);

        /* Receive the export results */
        void ev_recv_export_result(CommBuffer * exportbuf, struct ImpExpCounts * counts)
        {
            MPI_Datatype type;
            MPI_Type_contiguous(sizeof(ResultType), MPI_BYTE, &type);
            MPI_Type_commit(&type);
            exportbuf->databuf = (char*) mymalloc2("ExportResult", counts->Nexport * sizeof(ResultType));
            /* Post the receives first so we can hit a zero-copy fastpath.*/
            exportbuf->MPI_fill(counts->Export_count, counts->Export_offset, type, COMM_RECV, 101923, counts->comm);
            // alloc_commbuffer(&res_imports, counts.NTask, 0);
            // MPI_fill_commbuffer(import, counts->Import_count, counts->Import_offset, type, COMM_SEND, 101923, counts->comm);
            MPI_Type_free(&type);
            return;
        }

        void ev_reduce_export_result(CommBuffer * exportbuf, struct ImpExpCounts * counts, const struct particle_data * const parts);

        /* Checks whether all tasks have finished iterating */
        int ev_ndone(MPI_Comm comm)
        {
            int ndone;
            int done = !(BufferFullFlag);
            MPI_Allreduce(&done, &ndone, 1, MPI_INT, MPI_SUM, comm);
            return ndone;
        }

        void alloc_export_memory(void);
        void free_export_memory(void);
};

/* This function find the closest index in the multi-evaluation list of hsml and numNgb, update left and right bound, and return the new hsml */
double ngb_narrow_down(double *right, double *left, const double *radius, const double *numNgb, int maxcmpt, int desnumngb, int *closeidx, double BoxSize);

#endif
