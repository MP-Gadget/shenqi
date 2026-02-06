#ifndef _TREEWALK2_H_
#define _TREEWALK2_H_

#include <cstdint>
#include <omp.h>
#include <cmath>
#include "localtreewalk2.h"
#include "forcetree.h"
#include "utils/mymalloc.h"
#include "utils/endrun.h"
#include "utils/system.h"

#define MAXITER 400

class ExportMemory {
    public:
    const int64_t NThread; /*Number of OpenMP threads*/
    const size_t BunchSize;
    /* Information allowing the toptree walk to restart successfully after the export buffer fills up*/
    /* Export counters for each thread*/
    size_t * const Nexport_thread;
    /* Pointer to a particle export table for each thread.*/
    data_index ** ExportTable_thread;
    int64_t * const QueueChunkEnd;
    int * const QueueChunkRestart;
    ExportMemory(const size_t i_BunchSize) :
    NThread(omp_get_max_threads()), BunchSize(i_BunchSize), Nexport_thread(ta_malloc2("localexports", size_t, NThread)),
    ExportTable_thread(ta_malloc2("localexports", data_index *, NThread)), QueueChunkEnd(ta_malloc2("queueend", int64_t, NThread)),
    QueueChunkRestart(ta_malloc2("queuerestart", int, NThread))
    {
        int i;
        for(i = 0; i < NThread; i++)
            ExportTable_thread[i] = (data_index*) mymalloc("DataIndexTable", sizeof(data_index) * BunchSize);
        for(i = 0; i < NThread; i++)
            QueueChunkEnd[i] = -1;
    }

    ~ExportMemory()
    {
        myfree(QueueChunkRestart);
        myfree(QueueChunkEnd);
        for(int i = NThread - 1; i >= 0; i--)
            myfree(ExportTable_thread[i]);
        myfree(ExportTable_thread);
        myfree(Nexport_thread);
    }
};

/* This stores counts of the particles to be transferred.
 * It is book-keeping for the distributed part of the treewalk.
 * The constructor does an MPI_Alltoall, and thus is a synchronisation point.*/
class ImpExpCounts
{
    public:
    int64_t * Export_count;
    int64_t * Import_count;
    int64_t * Export_offset;
    int64_t * Import_offset;
    const MPI_Comm comm;
    int NTask;
    /* Number of particles exported to this processor*/
    size_t Nimport;
    /* Number of particles exported from this processor*/
    size_t Nexport;
    /* Number of MPI ranks we export to from this rank.*/
    int64_t NExportTargets;

    ImpExpCounts(const MPI_Comm i_comm, const ExportMemory& exports): comm(i_comm)
    {
        MPI_Comm_size(comm, &NTask);
        Export_count = ta_malloc("Tree_counts", int64_t, 4*NTask);
        Export_offset = Export_count + NTask;
        Import_count = Export_offset + NTask;
        Import_offset = Import_count + NTask;
        memset(Export_count, 0, sizeof(int64_t)*4*NTask);

        Nexport=0;
        /* Calculate the amount of data to send. */
        for(int64_t i = 0; i < exports.NThread; i++)
        {
            for(size_t k = 0; k < exports.Nexport_thread[i]; k++)
                Export_count[exports.ExportTable_thread[i][k].Task]++;
            /* This is the export count*/
            Nexport += exports.Nexport_thread[i];
        }
        /* Exchange the counts. Note this is synchronous so we need to ensure the toptree walk, which happens before this, is balanced.*/
        MPI_Alltoall(Export_count, 1, MPI_INT64, Import_count, 1, MPI_INT64, comm);
        // message(1, "Exporting %ld particles. Thread 0 is %ld\n", counts.Nexport, Nexport_thread[0]);

        Nimport = Import_count[0];
        NExportTargets = (Export_count[0] > 0);
        for(int i = 1; i < NTask; i++)
        {
            Nimport += Import_count[i];
            Export_offset[i] = Export_offset[i - 1] + Export_count[i - 1];
            Import_offset[i] = Import_offset[i - 1] + Import_count[i - 1];
            NExportTargets += (Export_count[i] > 0);
        }
        return;
    };

    ~ImpExpCounts()
    {
        ta_free(Export_count);
    };
};

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

/* 7/9/24: The code segfaults if the send/recv buffer is larger than 4GB in size.
 * Likely a 32-bit variable is overflowing but it is hard to debug. Easier to enforce a maximum buffer size.*/
size_t compute_bunchsize(const size_t query_type_elsize, const size_t result_type_elsize, const size_t MaxExportBufferBytes)
{
   /*The amount of memory eventually allocated per tree buffer*/
   size_t bytesperbuffer = sizeof(struct data_index) + query_type_elsize + result_type_elsize;
   /*This memory scales like the number of imports. In principle this could be much larger than Nexport
    * if the tree is very imbalanced and many processors all need to export to this one. In practice I have
    * not seen this happen, but provide a parameter to boost the memory for Nimport just in case.*/
   const double ImportBufferBoost = 2;
   bytesperbuffer += ceil(ImportBufferBoost * (query_type_elsize + result_type_elsize));
   /*Use all free bytes for the tree buffer, as in exchange. Leave some free memory for array overhead.*/
   size_t freebytes = (size_t) mymalloc_freebytes();
   freebytes -= 4096 * 10 * bytesperbuffer;

   size_t BunchSize = (size_t) floor(((double)freebytes)/ bytesperbuffer);
   if(BunchSize * query_type_elsize > MaxExportBufferBytes)
       BunchSize = MaxExportBufferBytes / query_type_elsize;
   /* Per thread*/
   BunchSize /= omp_get_max_threads();

   if(freebytes <= 4096 * bytesperbuffer || BunchSize < 100) {
       endrun(1231245, "Not enough free memory to export particles: needed %ld bytes have %ld. Can export %ld \n", bytesperbuffer, freebytes, BunchSize);
   }
   return BunchSize;
}

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
    /* Number of times we needed to re-run the treewalk.
     * Convenience variable for density. */
    int64_t Niteration;
    size_t NExportTargets;
    /* Counters for imbalance diagnostics*/
    int64_t maxNinteractions;
    int64_t minNinteractions;
    int64_t Ninteractions;

    /* internal flags*/
    /* Flags that our export buffer is full*/
    int BufferFullFlag;
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
        use_openmp_target(0),
        timewait1(0), timecomp0(0), timecomp1(0), timecomp2(0), timecomp3(0), timecommsumm(0),
        Nlistprimary(0), Nexport_sum(0), Nexportfull(0), Niteration(0), NExportTargets(0),
        maxNinteractions(0), minNinteractions(0), Ninteractions(0), BufferFullFlag(0),
        work_set_stolen_from_active(0),
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
    void run(int * active_set, size_t size, particle_data * const parts, const size_t MaxExportBufferBytes = 3584*1024*1024L)
    {
        if(!force_tree_allocated(tree)) {
            endrun(0, "Tree has been freed before this treewalk.\n");
        }

        double tstart, tend;
    #ifdef DEBUG
        GDB_current_ev = tw;
    #endif

        tstart = second();
        ev_begin(active_set, size, parts);

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
        /* Needs to be outside loop because it allocates restart information.
         * Freed at the end of the treewalk. */
        const size_t BunchSize = compute_bunchsize(sizeof(QueryType), sizeof(ResultType), MaxExportBufferBytes);
        ExportMemory exportlist(BunchSize);
        /* Print some balance numbers*/
        int64_t nmin, nmax, total;
        MPI_Reduce(&WorkSetSize, &nmin, 1, MPI_INT64, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&WorkSetSize, &nmax, 1, MPI_INT64, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&WorkSetSize, &total, 1, MPI_INT64, MPI_SUM, 0, MPI_COMM_WORLD);
        message(0, "Treewalk %s iter %ld: total part %ld max/MPI: %ld min/MPI: %ld balance: %g query %ld result %ld BunchSize %ld.\n",
            ev_label, Niteration, total, nmax, nmin, (double)nmax/((total+0.001)/NTask), sizeof(QueryType), sizeof(ResultType), BunchSize);
        report_memory_usage(ev_label);
        do
        {
            tstart = second();
            /* First do the toptree and export particles for sending.*/
            ev_toptree(parts, &exportlist);
            /* All processes sync via alltoall.*/
            ImpExpCounts counts(MPI_COMM_WORLD, exportlist);
            NExportTargets = counts.NExportTargets;
            Nexport_sum += counts.Nexport;
            Ndone = ev_ndone(MPI_COMM_WORLD);
            /* Send the exported particle data */
            /* exports is allocated first, then imports*/
            CommBuffer exports(counts.NTask, 0);
            CommBuffer imports(counts.NTask, 0);
            ev_send_recv_export_import(&counts, &exportlist, &exports, &imports, parts);
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
            ev_secondary(&res_imports, &imports, &counts, parts);
            // report_memory_usage(ev_label);
            // Want to explicitly free the databuf early for this one so we free memory early.
            myfree(imports.databuf);
            imports.databuf = NULL;
            tend = second();
            timecomp2 += timediff(tstart, tend);
            /* Now clear the sent data buffer, waiting for the send to complete.
                * This needs to be after the other end has called recv.*/
            tstart = second();
            res_exports.wait();
            tend = second();
            timewait1 += timediff(tstart, tend);
            tstart = second();
            ev_reduce_export_result(&res_exports, &counts, &exportlist, parts);
            tend = second();
            timecommsumm += timediff(tstart, tend);
            /* Free export memory*/
            Nexportfull++;
            /* The destructors for the CommBuffers will fire at this point,
             * which means there is an implicit wait() */
        } while(Ndone < NTask);

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
    void build_queue(int * active_set, const size_t size, int may_have_garbage, const particle_data * const Parts)
    {
        if(!should_rebuild_queue && !may_have_garbage)
        {
            WorkSetSize = size;
            WorkSet = active_set;
            work_set_stolen_from_active = 1;
            return;
        }

        work_set_stolen_from_active = 0;
        /* Explicitly deal with the case where the queue is zero and there is nothing to do.
         * Some OpenMP compilers (nvcc) seem to still execute the below loop in that case*/
        if(size == 0) {
            WorkSet = (int *) mymalloc("ActiveQueue", sizeof(int));
            WorkSetSize = size;
            return;
        }

        /*We want a lockless algorithm which preserves the ordering of the particle list.*/
        gadget_thread_arrays gthread = gadget_setup_thread_arrays("ActiveQueue", 0, size);
        /* We enforce schedule static to ensure that each thread executes on contiguous particles.
         * Note static enforces the monotonic modifier but on OpenMP 5.0 nonmonotonic is the default.
         * static also ensures that no single thread gets more than tsize elements.*/
        #pragma omp parallel
        {
            size_t i;
            const int tid = omp_get_thread_num();
            size_t nqthrlocal = 0;
            int *thrqlocal = gthread.srcs[tid];
            #pragma omp for schedule(static, gthread.schedsz)
            for(i=0; i < size; i++)
            {
                /*Use raw particle number if active_set is null, otherwise use active_set*/
                const int p_i = active_set ? active_set[i] : (int) i;
                const particle_data& pp = Parts[p_i];
                /* Skip the garbage /swallowed particles */
                if(pp.IsGarbage || pp.Swallowed)
                    continue;

                if(!haswork(pp))
                    continue;
        #ifdef DEBUG
                if(nqthrlocal >= gthread.total_size)
                    endrun(5, "tid = %d nqthr = %ld, tsize = %ld size = %ld, Nthread = %ld i = %ld\n", tid, nqthrlocal, gthread.total_size, size, NThread, i);
        #endif
                thrqlocal[nqthrlocal] = p_i;
                nqthrlocal++;
            }
            gthread.sizes[tid] = nqthrlocal;
        }
        /*Merge step for the queue.*/
        size_t nqueue = gadget_compact_thread_arrays(&WorkSet, &gthread);
        /*Shrink memory*/
        WorkSet = (int *) myrealloc(WorkSet, sizeof(int) * nqueue);

    #if 0
        /* check the uniqueness of the active_set list. This is very slow. */
        qsort_openmp(WorkSet, nqueue, sizeof(int), cmpint);
        for(i = 0; i < nqueue - 1; i ++) {
            if(WorkSet[i] == WorkSet[i+1]) {
                endrun(8829, "A few particles are twicely active.\n");
            }
        }
    #endif
        WorkSetSize = nqueue;
    }

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

        void ev_begin(int * active_set, const size_t size, particle_data * const parts)
        {
            /* The last argument is may_have_garbage: in practice the only
             * trivial haswork is the gravtree. This has no (active) garbage because
             * the active list was just rebuilt, but on a PM step the active list is NULL
             * and we may still have swallowed BHs around. So in practice this avoids
             * computing gravtree for swallowed BHs on a PM step.*/
            int may_have_garbage = 0;
            /* Note this is not collective, but that should not matter.*/
            if(!active_set && SlotsManager->info[5].size > 0)
                may_have_garbage = 1;
            build_queue(active_set, size, may_have_garbage, parts);
            /* Start first iteration at the beginning*/
            WorkSetStart = 0;
        }

        /* returns struct containing export counts */
        void ev_primary(particle_data * const parts)
        {
            int64_t maxNinteractions = 0, minNinteractions = 1L << 45, Ninteractions=0;
        #pragma omp parallel reduction(min:minNinteractions) reduction(max:maxNinteractions) reduction(+: Ninteractions)
            {
                /* Note: exportflag is local to each thread */
                LocalTreeWalkType lv(TREEWALK_PRIMARY, tree, 0, NULL);

                /* We must schedule dynamically so that we have reduced imbalance.
                * We do not need to worry about the export buffer filling up.*/
                /* chunk size: 1 and 1000 were slightly (3 percent) slower than 8.
                * FoF treewalk needs a larger chnksz to avoid contention.*/
                int64_t chnksz = WorkSetSize / (4*omp_get_num_threads());
                if(chnksz < 1)
                    chnksz = 1;
                if(chnksz > 100)
                    chnksz = 100;
                int k;
                #pragma omp for schedule(dynamic, chnksz)
                for(k = 0; k < WorkSetSize; k++) {
                    const int i = WorkSet ? WorkSet[k] : k;
                    /* Primary never uses node list */
                    QueryType input(parts[i], NULL, tree->firstnode, priv);
                    ResultType output(input);
                    lv.target = i;
                    lv.visit(input, &output, priv, parts);
                    output.reduce(i, TREEWALK_PRIMARY, priv, parts);
                }
                if(maxNinteractions < lv.maxNinteractions)
                    maxNinteractions = lv.maxNinteractions;
                if(minNinteractions > lv.maxNinteractions)
                    minNinteractions = lv.minNinteractions;
                Ninteractions = lv.Ninteractions;
            }
            Ninteractions += Ninteractions;
            Nlistprimary += WorkSetSize;
        }

        int ev_toptree(const particle_data * const parts, ExportMemory * const exportlist)
        {
            BufferFullFlag = 0;
            int64_t currentIndex = WorkSetStart;
            int BufferFullFlag = 0;

            if(Nexportfull > 0)
                message(0, "Toptree %s, iter %ld. First particle %ld size %ld.\n", ev_label, Nexportfull, WorkSetStart, WorkSetSize);

        #pragma omp parallel reduction(+: BufferFullFlag)
            {
                LocalTreeWalkType lv(TREEWALK_TOPTREE, tree, exportlist->BunchSize, exportlist->ExportTable_thread);
                /* Signals a full export buffer on this thread*/
                int BufferFull_thread = 0;
                const int tid = omp_get_thread_num();

                /* We schedule dynamically so that we have reduced imbalance.
                 * We do not use the openmp dynamic scheduling, but roll our own
                 * so that we can break from the loop if needed.*/
                int64_t chnk = 0;
                /* chunk size: 1 and 1000 were slightly (3 percent) slower than 8.
                 * FoF treewalk needs a larger chnksz to avoid contention.*/
                int64_t chnksz = WorkSetSize / (4*exportlist->NThread);
                if(chnksz < 1)
                    chnksz = 1;
                if(chnksz > 1000)
                    chnksz = 1000;
                do {
                    int64_t end;
                    /* Restart a previously partially evaluated chunk if there is one*/
                    if(Nexportfull > 0 && exportlist->QueueChunkEnd[tid] > 0) {
                        chnk = exportlist->QueueChunkRestart[tid];
                        end = exportlist->QueueChunkEnd[tid];
                        exportlist->QueueChunkEnd[tid] = -1;
                        //message(1, "T%d Restarting chunk %ld -> %ld\n", tid, chnk, end);
                    }
                    else {
                        /* Get another chunk from the global queue*/
                        chnk = atomic_fetch_and_add_64(&currentIndex, chnksz);
                        /* This is a hand-rolled version of what openmp dynamic scheduling is doing.*/
                        end = chnk + chnksz;
                        /* Make sure we do not overflow the loop*/
                        if(end > WorkSetSize)
                            end = WorkSetSize;
                    }
                    /* Reduce the chunk size towards the end of the walk*/
                    if((WorkSetSize  < end + chnksz * exportlist->NThread) && chnksz >= 2)
                        chnksz /= 2;
                    int k;
                    for(k = chnk; k < end; k++) {
                        const int i = WorkSet ? WorkSet[k] : k;
                        /* Toptree never uses node list */
                        QueryType input(parts[i], NULL, tree->firstnode, priv);
                        lv.target = i;
                        /* Reset the number of exported particles.*/
                        ResultType output(input);
                        const int rt = lv.toptree_visit(input, &output, priv, parts);
                        /* If we filled up, we need to save the partially evaluated chunk, and leave this loop.*/
                        if(rt < 0) {
                            //message(5, "Export buffer full for particle %d chnk: %ld -> %ld on thread %d with %ld exports\n", i, chnk, end, tid, lv->NThisParticleExport);
                            /* export buffer has filled up, can't do more work.*/
                            BufferFull_thread = 1;
                            /* Store information for the current chunk, so we can resume successfully exactly where we left off.
                                Each thread stores chunk information */
                            exportlist->QueueChunkRestart[tid] = k;
                            exportlist->QueueChunkEnd[tid] = end;
                            break;
                        }
                    }
                } while(chnk < WorkSetSize && BufferFull_thread == 0);
                exportlist->Nexport_thread[tid] = lv.Nexport;
                BufferFullFlag += BufferFull_thread;
            }

            if(BufferFullFlag > 0) {
                size_t Nexport = 0;
                int i;
                for(i = 0; i < exportlist->NThread; i++)
                    Nexport += exportlist->Nexport_thread[i];
                message(1, "Tree export buffer full on %d of %ld threads with %lu exports (%lu Mbytes). First particle %ld new start: %ld size %ld.\n",
                                BufferFullFlag, exportlist->NThread, Nexport, Nexport*sizeof(QueryType)/1024/1024, WorkSetStart, currentIndex, WorkSetSize);
                if(currentIndex == WorkSetStart)
                    endrun(5, "Not enough export space to make progress! lastsuc %ld\n", currentIndex);
            }
            // else
                // message(1, "Finished toptree on %d threads. First particle %ld next start: %ld size %ld.\n", BufferFullFlag, WorkSetStart, currentIndex, WorkSetSize);
            /* Start again with the next chunk not yet evaluated*/
            WorkSetStart = currentIndex;
            return BufferFullFlag;
        }

        void ev_secondary(CommBuffer * res_imports, CommBuffer * imports, ImpExpCounts* counts, const struct particle_data * const parts)
        {
            res_imports->databuf = (char *) mymalloc2("ImportResult", counts->Nimport * sizeof(ResultType));

            MPI_Datatype type;
            MPI_Type_contiguous(sizeof(ResultType), MPI_BYTE, &type);
            MPI_Type_commit(&type);
            int * complete_array = ta_malloc("completes", int, imports->nrequest_all);

            int tot_completed = 0;
            /* Test each request in turn until it completes*/
            while(tot_completed < imports->nrequest_all) {
                int complete_cnt = MPI_UNDEFINED;
                /* Check for some completed requests: note that cleanup is performed if the requests are complete.
                 * There may be only 1 completed request, and we need to wait again until we have more.*/
                MPI_Waitsome(imports->nrequest_all, imports->rdata_all, &complete_cnt, complete_array, MPI_STATUSES_IGNORE);
                /* This happens if all requests are MPI_REQUEST_NULL. It should never be hit*/
                if (complete_cnt == MPI_UNDEFINED)
                    break;
                int j;
                for(j = 0; j < complete_cnt; j++) {
                    const int i = complete_array[j];
                    /* Note the task number index is not the index in the request array (some tasks were skipped because we have zero exports)! */
                    const int task = imports->rqst_task[i];
                    const int64_t nimports_task = counts->Import_count[task];
                    // message(1, "starting at %d with %d for iport %d task %d\n", counts->Import_offset[task], counts->Import_count[task], i, task);
                    char * databufstart = imports->databuf + counts->Import_offset[task] * sizeof(QueryType);
                    char * dataresultstart = res_imports->databuf + counts->Import_offset[task] * sizeof(ResultType);
                    /* This sends each set of imports to a parallel for loop. This may lead to suboptimal resource allocation if only a small number of imports come from a processor.
                    * If there are a large number of importing ranks each with a small number of imports, a better scheme could be to send each chunk to a separate openmp task.
                    * However, each openmp task by default only uses 1 thread. One may explicitly enable openmp nested parallelism, but I think that is not safe,
                    * or it would be enabled by default.*/
                    #pragma omp parallel
                        {
                            ResultType * results = (ResultType *) dataresultstart;
                            int64_t j;
                            LocalTreeWalkType lv(TREEWALK_GHOSTS, tree, 0, NULL);
                            #pragma omp for
                            for(j = 0; j < nimports_task; j++) {
                                QueryType * input = &((QueryType *) databufstart)[j];
                                ResultType * output = new (&results[j]) ResultType(*input);
                                lv.target = -1;
                                lv.visit(*input, output, priv, parts);
                            }
                        }
                    /* Send the completed data back*/
                    res_imports->rqst_task[res_imports->nrequest_all] = task;
                    MPI_Isend(dataresultstart, nimports_task, type, task, 101923, counts->comm, &res_imports->rdata_all[res_imports->nrequest_all++]);
                    tot_completed++;
                }
            };
            myfree(complete_array);
            MPI_Type_free(&type);
            return;
        }

        /* Cleans up and frees memory */
        void ev_finish(void)
        {
            if(!work_set_stolen_from_active)
                myfree(WorkSet);
        }

        /* Builds the list of exported particles and async sends the export queries. */
        void ev_send_recv_export_import(const ImpExpCounts * const counts, const ExportMemory * const exportlist, CommBuffer * exports, CommBuffer * imports, const particle_data * const parts)
        {
            exports->databuf = (char *) mymalloc("ExportQuery", counts->Nexport * sizeof(QueryType));
            imports->databuf = (char *) mymalloc("ImportQuery", counts->Nimport * sizeof(QueryType));

            MPI_Datatype type;
            MPI_Type_contiguous(sizeof(QueryType), MPI_BYTE, &type);
            MPI_Type_commit(&type);

            /* Post recvs before sends. This sometimes allows for a fastpath.*/
            imports->MPI_fill(counts->Import_count, counts->Import_offset, type, COMM_RECV, 101922, counts->comm);

            /* prepare particle data for export */
            int64_t * real_send_count = ta_malloc("tmp_send_count", int64_t, NTask);
            memset(real_send_count, 0, sizeof(int64_t)*NTask);
            int64_t i;
            QueryType * export_queries = reinterpret_cast<QueryType*>(exports->databuf);
            for(i = 0; i < exportlist->NThread; i++)
            {
                size_t k;
                for(k = 0; k < exportlist->Nexport_thread[i]; k++) {
                    const int place = exportlist->ExportTable_thread[i][k].Index;
                    const int task = exportlist->ExportTable_thread[i][k].Task;
                    const int64_t bufpos = real_send_count[task] + counts->Export_offset[task];
                    real_send_count[task]++;
                    /* Initialize the query in this memory */
                    new(&export_queries[bufpos]) QueryType(parts[place], exportlist->ExportTable_thread[i][k].NodeList, -1, priv);
                }
            }
        #ifdef DEBUG
        /* Checks!*/
            for(i = 0; i < NTask; i++)
                if(real_send_count[i] != counts->Export_count[i])
                    endrun(6, "Inconsistent export to task %ld of %d: %ld expected %ld\n", i, NTask, real_send_count[i], counts->Export_count[i]);
        #endif
            myfree(real_send_count);
            exports->MPI_fill(counts->Export_count, counts->Export_offset, type, COMM_SEND, 101922, counts->comm);
            MPI_Type_free(&type);
            return;
        }

        /* Receive the export results */
        void ev_recv_export_result(CommBuffer * exportbuf, ImpExpCounts * counts)
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

        void ev_reduce_export_result(CommBuffer * exportbuf, const ImpExpCounts * const counts, const ExportMemory * const exportlist, struct particle_data * const parts)
        {
            int64_t i;
            /* Notice that we build the dataindex table individually
             * on each thread, so we are ordered by particle and have memory locality.*/
            int * real_recv_count = ta_malloc("tmp_recv_count", int, NTask);
            memset(real_recv_count, 0, sizeof(int)*NTask);
            for(i = 0; i < exportlist->NThread; i++)
            {
                size_t k;
                for(k = 0; k < exportlist->Nexport_thread[i]; k++) {
                    const int place = exportlist->ExportTable_thread[i][k].Index;
                    const int task = exportlist->ExportTable_thread[i][k].Task;
                    const int64_t bufpos = real_recv_count[task] + counts->Export_offset[task];
                    real_recv_count[task]++;
                    ResultType * output = &((ResultType *) exportbuf->databuf)[bufpos];
                    output->reduce(place, TREEWALK_GHOSTS, priv, parts);
        #ifdef DEBUG
                    if(output->ID != parts[place].ID)
                        endrun(8, "Error in communication: IDs mismatch %ld %ld\n", output->ID, parts[place].ID);
        #endif
                }
            }
            myfree(real_recv_count);
        }

        /* Checks whether all tasks have finished iterating */
        int ev_ndone(MPI_Comm comm)
        {
            int ndone;
            int done = !(BufferFullFlag);
            MPI_Allreduce(&done, &ndone, 1, MPI_INT, MPI_SUM, comm);
            return ndone;
        }
};

/* This function find the closest index in the multi-evaluation list of hsml and numNgb, update left and right bound, and return the new hsml */
double ngb_narrow_down(double *right, double *left, const double *radius, const double *numNgb, int maxcmpt, int desnumngb, int *closeidx, double BoxSize);

#endif
