#ifndef _TREEWALK2_H_
#define _TREEWALK2_H_
/* This file contains the main header for the distributed treewalk.  */
#include <cstdint>
#include <string>
#include <omp.h>
#include <cmath>
#include "localtreewalk2.h"
#include "forcetree.h"
#include "mpi.h"
#include "walltime.h"
#include "utils/mymalloc.h"
#include "utils/endrun.h"
#include "utils/system.h"
#include <ranges>
/* For the parallel execution policy.*/
#include <execution>
#include <algorithm>
#include <numeric>

#define MAXITER 400

/* This struct is returned by the ev_toptree. */
template<typename QueryType> struct ExportMemory2 {
    /* Export counter, size of the export table*/
    size_t Nexport = 0;
    /* Pointer to a particle export table.*/
    data_index * ExportTable = NULL;
    QueryType * ExportQueries = NULL;
    ~ExportMemory2()
    {
        if(ExportQueries)
            myfree(ExportQueries);
        if(ExportTable)
            myfree(ExportTable);
    }
};

/* This stores counts of the particles to be transferred.
 * It is book-keeping for the distributed part of the treewalk.
 * The constructor does an MPI_Alltoall, and thus is a synchronisation point.*/
template<typename QueryType> class ImpExpCounts
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

    ImpExpCounts(const MPI_Comm i_comm, const ExportMemory2<QueryType>& exports): comm(i_comm)
    {
        MPI_Comm_size(comm, &NTask);
        Export_count = ta_malloc("Tree_counts", int64_t, 4*NTask);
        Export_offset = Export_count + NTask;
        Import_count = Export_offset + NTask;
        Import_offset = Import_count + NTask;
        memset(Export_count, 0, sizeof(int64_t)*4*NTask);

        Nexport=0;
        /* Calculate the amount of data to send. */
        for(size_t k = 0; k < exports.Nexport; k++) {
#ifdef DEBUG
            if(exports.ExportTable[k].Task < 0 || exports.ExportTable[k].Task > NTask)
                endrun(5, "export count at %lu of %lu is %d > NTask: %d\n", k, exports.Nexport, exports.ExportTable[k].Task, NTask);
#endif
            Export_count[exports.ExportTable[k].Task]++;
	    }
        /* This is the export count*/
        Nexport  = exports.Nexport;
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
    std::vector<int> rqst_task;
    /* Needs a C-style flat memory array */
    MPI_Request * rdata_all;
    CommBuffer(): databuf(NULL), rdata_all(NULL) {}
    ~CommBuffer()
    {
        /* First wait until all comms are done */
        wait();
        if(databuf) {
            myfree(databuf);
            databuf = NULL;
        }
        if(rdata_all)
            myfree(rdata_all);
    }
    /* Get the number of active requests. */
    size_t nrequest(void) const
    {
        return rqst_task.size();
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

        /* Loop over all tasks, counting the number of requests.*/
        for(int j = 1; j < NTask; j++)
        {
            int target = (ThisTask + j) % NTask;
            if(cnts[target] == 0) continue;
            rqst_task.push_back(target);
        }
        rdata_all = mymalloc("MPI_requests", MPI_Request, rqst_task.size());
        /* Do Send/Recv over all non-trivial tasks*/
        for(size_t i = 0; i < rqst_task.size(); i++)
        {
            int target = rqst_task[i];
            if(receive == COMM_RECV) {
                MPI_Irecv(((char*) databuf) + elsize * displs[target], cnts[target],
                    type, target, tag, comm, &rdata_all[i]);
            }
            else {
                MPI_Isend(((char*) databuf) + elsize * displs[target], cnts[target],
                    type, target, tag, comm, &rdata_all[i]);
            }
        }
    }

    /* Waits for all the requests in the commbuffer to be complete*/
    void wait(void)
    {
        if(rdata_all)
            MPI_Waitall(rqst_task.size(), rdata_all, MPI_STATUSES_IGNORE);
    }
};

struct Greater_than_BunchSize
{
    int BunchSize;
    MYCUDAFN bool operator() (int i) const {
        return i >= BunchSize;
    }
};


template <typename QueryType>
class HasWorkPredicate {
public:
     const particle_data * parts;
     MYCUDAFN bool operator()(int i) const {
         return QueryType::haswork(parts[i]);
     }
};

/**
 * TreeWalk - Base class for tree-based particle interactions.
 *
 * This class provides the framework for walking a tree structure and
 * computing interactions between particles. Derived classes should override
 * the methods haswork and postprocess to implement specific physics (e.g., gravity, SPH).
 *
 * Usage:
 *   1. Derive from TreeWalk and override the required virtual methods
 *   2. Set tree, ev_label, type, and element sizes in the constructor
 *   3. Call treewalk_run() to execute the tree walk
 */
template <typename DerivedType, typename QueryType, typename ResultType, typename LocalTreeWalkType, typename LocalTopTreeWalkType, typename ParamType, typename OutputType>
class TreeWalk {
public:
    /* A pointer to the force tree structure to walk.*/
    const ForceTree * const tree;

    /* name of the evaluator (used in printing messages) */
    const char * const ev_label;
    /* Note this is a reference so that the ParamType/OutputType destructor does not run during TreeWalk destruction,
     * which would free the underlying memory */
    const ParamType& priv;
    OutputType * output;
    /* Maximum size of the export buffer */
    size_t MaxExportBufferBytes = 3584*1024*1024L;
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
    /* Total number of exported particles
     * (Nexport is only the exported particles in the current export buffer). */
    int64_t Nexport_sum;
    /* Convenience variable for density. */
    size_t NExportTargets;
    /**
     * Constructor - initializes all members to safe defaults.
     */
    TreeWalk(const char * const i_ev_label, const ForceTree * const i_tree, const ParamType& i_priv, OutputType * i_out):
        tree(i_tree), ev_label(i_ev_label),
        priv(i_priv), output(i_out),
        timewait1(0), timecomp0(0), timecomp1(0), timecomp2(0), timecomp3(0), timecommsumm(0),
        Nexport_sum(0), NExportTargets(0)
    {    }

    /* Do the distributed tree walking. Warning: as this is a threaded treewalk,
     * it may call visit on particles more than once and in a noneterministic order.
     * Your module should behave correctly in this case!
     *
     * active_set : a list of indices of particles that walk the tree. If active_set is NULL,
     *              all (NumPart) particles are used. This is not the list of particles
     * in the tree, but the particles that do the walking.
     * size: length of the active set
     * particle_data parts: list of particles to use
     */
    void run(int * active_set, size_t size, particle_data * const parts, MPI_Comm comm)
    {
        double tstart, tend;
        tstart = second();
        LocalTreeWalkType::validate_tree(tree);

        int * WorkSet=NULL;
        int64_t WorkSetSize = build_queue(&WorkSet, active_set, size, parts);
        tend = second();
        timecomp3 += timediff(tstart, tend);
        run_on_queue(WorkSet, WorkSetSize, parts, comm);

        myfree(WorkSet);
    }

    /* Do the distributed tree walking. This assumes that the work queue has already been built.
     *
     * WorkSet : a list of indices of particles that walk the tree. If NULL,
     *              all (WorkSetSize) particles are used. This is not the list of particles
     * in the tree, but the particles that do the walking.
     * WorkSetSize: length of the active set
     * particle_data parts: list of particles to use
     */
    void run_on_queue(int * WorkSet, int64_t WorkSetSize, particle_data * const parts, MPI_Comm comm, bool postprocess=true)
    {
        LocalTreeWalkType::validate_tree(tree);
        Nexport_sum = 0;

        /* Build the queries from the particle table first, so we only need to read it once*/
        QueryType * queries = mymanagedmalloc("Queries", QueryType, WorkSetSize);
        #pragma omp parallel for
        for(int64_t k = 0 ; k < WorkSetSize; k++) {
                const int i = WorkSet ? WorkSet[k] : k;
                /* Toptree never uses node list */
                new(&queries[k]) QueryType(parts[i], NULL, tree->firstnode, priv);
        }

        const int BunchSize = compute_bunchsize();
        int64_t nmin, nmax, total;
        int NTask;
        MPI_Comm_size(comm, &NTask);
        MPI_Reduce(&WorkSetSize, &nmin, 1, MPI_INT64, MPI_MIN, 0, comm);
        MPI_Reduce(&WorkSetSize, &nmax, 1, MPI_INT64, MPI_MAX, 0, comm);
        MPI_Reduce(&WorkSetSize, &total, 1, MPI_INT64, MPI_SUM, 0, comm);
        message(0, "Treewalk %s: total part %ld max/MPI: %ld min/MPI: %ld balance: %g query %ld result %ld BunchSize %d.\n",
            ev_label, total, nmax, nmin, (double)nmax/((total+0.001)/NTask), sizeof(QueryType), sizeof(ResultType), BunchSize);
        /* Print some balance numbers*/
        report_memory_usage(ev_label);

        /* Number of times we filled up our export buffer*/
        int Nexportfull = 0;
        int Ndone = 0;
        /* Start first iteration at the beginning*/
        int64_t WorkSetStart = 0;

        /* We count all exports before the main export loop. */
        int * exportcounts = static_cast<DerivedType *>(this)->ev_count_exports(WorkSet, WorkSetSize, queries);

        /* Main loop that copies into the export table, then does primary and secondary evals. */
        do {
            double tstart, tend;
            tstart = second();

            if(Nexportfull > 0)
                message(0, "Toptree %s, iter %d. First particle %ld size %ld.\n", ev_label, Nexportfull, WorkSetStart, WorkSetSize);
            ExportMemory2<QueryType> exportlist;
            /* First do the toptree and export particles for sending.*/
            WorkSetStart = static_cast<DerivedType *>(this)->ev_toptree(WorkSet, WorkSetStart, WorkSetSize, exportcounts, &exportlist, queries);
            /* All processes sync via alltoall.*/
            ImpExpCounts<QueryType> counts(comm, exportlist);
            NExportTargets = counts.NExportTargets;
            Nexport_sum += counts.Nexport;
            Ndone = ev_ndone(WorkSetStart < WorkSetSize, comm);
            /* Send the exported particle data */
            /* exports is allocated first, then imports*/
            CommBuffer exports;
            CommBuffer imports;
            ev_send_recv_export_import(&counts, &exportlist, &exports, &imports);
            tend = second();
            timecomp0 += timediff(tstart, tend);
            /* Only do this on the first iteration, as we only need to do it once.*/
            tstart = second();
            if(Nexportfull == 0)
                static_cast<DerivedType *>(this)->ev_primary(WorkSet, WorkSetSize, queries, parts); /* do local particles and prepare export list */
            tend = second();
            timecomp1 += timediff(tstart, tend);
            /* Do processing of received particles. We implement a queue that
                * checks each incoming task in turn and processes them as they arrive.*/
            tstart = second();
            /* Posts recvs to get the export results (which are sent in ev_secondary).*/
            CommBuffer res_exports;
            ev_recv_export_result(&res_exports, &counts);
            CommBuffer res_imports;
            ev_wait_secondary(&res_imports, &imports, &counts, parts);
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
            Nexportfull++;
            /* The destructors for the CommBuffers will fire at this point,
            * which means there is an implicit wait() */
            /* Free export memory in destructor of exportlist.*/
        } while(Ndone < NTask);

        /* GPU code needs to use cudaFree here. */
        static_cast<DerivedType *>(this)->ev_free_exports(exportcounts);

        myfree(queries);

        if(postprocess) {
            double tstart = second();
            static_cast<DerivedType *>(this)->ev_postprocess(WorkSet, WorkSetSize, parts);
            double tend = second();
            timecomp3 += timediff(tstart, tend);
        }
    }

    void ev_postprocess(int * WorkSet, int64_t WorkSetSize, particle_data * const parts)
    {
        #pragma omp parallel for
        for(int i = 0; i < WorkSetSize; i ++) {
            const int p_i = WorkSet ? WorkSet[i] : i;
            output->postprocess(p_i, parts, &priv);
        }
    }

    /* Build the queue by calling the haswork function on each particle in the active_set.
     * Arguments:
     * active_set: these items have haswork called on them.
     * size: size of the active set.*/
    int64_t build_queue(int ** WorkSet, int * active_set, const size_t size, const particle_data * const Parts)
    {
        /* Explicitly deal with the case where the queue is zero and there is nothing to do.
         * Some OpenMP compilers (nvcc) seem to still execute the below loop in that case*/
        if(size == 0) {
            *WorkSet = mymanagedmalloc("ActiveQueue", int, 1);
            return size;
        }

        *WorkSet = mymanagedmalloc("ActiveQueue", int, size);
        HasWorkPredicate<QueryType> haswork{Parts};
        /* This is a standard stream compaction algorithm. It evaluates the haswork function
         * for every particle in the active set, stores the results in an array of flags, counts the non-zero flags,
         * and then scatters each particle integer to the right index in the final array. All is parallelized. */
        if(active_set) {
            auto end = std::copy_if(std::execution::par,
                active_set, active_set + size, *WorkSet, haswork);
            return end - *WorkSet;
        }
        else { // Need to handle this separately.
#ifndef __CUDACC__
            /* The GPU code has a counting_iterator from thrust which avoids allocating the memory. This is the C++20 equivalent.*/
            auto iota = std::views::iota(0, (int) size);
#else
            /* The CUDA compiler does not support std::views::iota at this time, and this header is included there.
             * The code is never executed on the GPU as the function is over-ridden.*/
            std::vector <int> iota(size);
            std::iota(iota.begin(), iota.end(), 0);
#endif
            auto end = std::copy_if(std::execution::par, iota.begin(), iota.end(), *WorkSet, haswork);

            return end - *WorkSet;
        }
    }

    /* Print some counters for a completed treewalk*/
    void print_stats(const std::string& walltimeprefix, MPI_Comm comm)
    {
        /* collect some timing information */
        double timeall = walltime_measure(WALLTIME_IGNORE);
        double timecomp = timecomp0 + timecomp1 + timecomp2 + timecomp3;

        walltime_add(walltimeprefix+"/WalkTop", timecomp0);
        walltime_add(walltimeprefix+"/WalkPrim", timecomp1);
        walltime_add(walltimeprefix+"/WalkSec", timecomp2);
        walltime_add(walltimeprefix+"/PostProc", timecomp3);
        walltime_add(walltimeprefix+"/Wait", timewait1);
        walltime_add(walltimeprefix+"/Reduce", timecommsumm);
        walltime_add(walltimeprefix+"/Misc", timeall - (timecomp + timewait1 + timecommsumm));

        int NTask;
        MPI_Comm_size(comm, &NTask);
        int64_t o_NExportTargets, Nexport;
        MPI_Reduce(&Nexport_sum, &Nexport, 1, MPI_INT64, MPI_SUM, 0, comm);
        MPI_Reduce(&NExportTargets, &o_NExportTargets, 1, MPI_INT64, MPI_SUM, 0, comm);
        message(0, "%s: average exports: %g avg target ranks: %g\n",
            ev_label, ((double) Nexport)/ NTask, ((double) o_NExportTargets)/ NTask);
        message(0, "%s: top: %g prim: %g sec: %g wait: %g postproc: %g reduce: %g\n", ev_label, timecomp0, timecomp1, timecomp2, timewait1, timecomp3, timecommsumm);
    }

    /* 7/9/24: The code segfaults if the send/recv buffer is larger than 4GB in size.
        * Likely a 32-bit variable is overflowing but it is hard to debug. Easier to enforce a maximum buffer size.*/
    size_t compute_bunchsize(void)
    {
        /*The amount of memory eventually allocated per tree buffer*/
        const size_t query_type_elsize = sizeof(QueryType);
        const size_t result_type_elsize = sizeof(ResultType);
        size_t bytesperbuffer = sizeof(struct data_index) + query_type_elsize + result_type_elsize;
        /*This memory scales like the number of imports. In principle this could be much larger than Nexport
        * if the tree is very imbalanced and many processors all need to export to this one. In practice I have
        * not seen this happen, but provide a parameter to boost the memory for Nimport just in case.*/
        const double ImportBufferBoost = 2;
        bytesperbuffer += ceil(ImportBufferBoost * (query_type_elsize + result_type_elsize));
        /*Use all free bytes for the tree buffer, as in exchange. Leave some free memory for array overhead.*/
        size_t freebytes = mymalloc_freebytes();
        freebytes -= 4096 * 10 * bytesperbuffer;

        size_t BunchSize = (size_t) floor(((double)freebytes)/ bytesperbuffer);
        if(BunchSize * query_type_elsize > MaxExportBufferBytes)
            BunchSize = MaxExportBufferBytes / query_type_elsize;

        if(freebytes <= 4096 * bytesperbuffer || BunchSize < 100) {
            endrun(1231245, "Not enough free memory to export particles: needed %ld bytes have %ld. Can export %ld \n", bytesperbuffer, freebytes, BunchSize);
        }
        return BunchSize;
    }

        /* This function does treewalk_run in a loop, allocating a queue to allow some particles to be redone.
    * This loop is used primarily in density estimation.*/
    void do_hsml_loop(int * queue, int64_t queuesize, const bool update_hsml, particle_data * parts, MPI_Comm comm)
    {
        /* Build the first queue */
        double tstart = second();
        int * ReDoQueue = NULL;
        int64_t size = build_queue(&ReDoQueue, queue, queuesize, parts);
        double tend = second();
        this->timecomp3 += timediff(tstart, tend);
        /* Number of times the outer loop was run. */
        int Niteration = 0;
        /* we will repeat the whole thing for those particles where we didn't find enough neighbours */
        do {
            double maxnumngb = 0;
            double minnumngb = 1e60;

            int * CurQueue = ReDoQueue;
            /* ev_postprocess is not done in run_on_queue, instead, done in this loop*/
            run_on_queue(CurQueue, size, parts, comm, false);

            tstart = second();
            output->verbose = (Niteration >= MAXITER - 5);
            /* Check which particles we need to repeat for. */
            int * todo = mymalloc("Particle_todo", int, size);
            #pragma omp parallel for reduction(max: maxnumngb) reduction(min: minnumngb)
            for(int i = 0; i < size; i ++) {
                const int p_i = CurQueue ? CurQueue[i] : i;
                double numngb = output->GetNumNgb(p_i);
                if(maxnumngb < numngb)
                    maxnumngb = numngb;
                if(minnumngb > numngb)
                    minnumngb = numngb;
                todo[i] = -1;
                /* If we are done, postprocess returns 1, todo contains -1.
                 * If we need to repeat, postprocess returns 0, todo contains
                 * the new item to add to the redo queue*/
                if(0 == output->postprocess(p_i, parts, &priv))
                    todo[i] = p_i;
                /* If we are done repeating, update the hmax in the parent node,
                 * Only do this when gas particles are in the tree, since we only need
                 * it for a symmetric treewalk and that is just hydro and BH mergers.
                 * Do it only for types in the tree.*/
                else if constexpr(LocalTreeWalkType::tree_mask & GASMASK)
                    if(update_hsml && (LocalTreeWalkType::tree_mask & (1<<parts[p_i].Type)))
                        update_tree_hmax_father(tree, p_i, parts[p_i].Pos, parts[p_i].Hsml);
            }
            ReDoQueue = mymanagedmalloc("ReDoQueue", int, size);
            /* Compact the redo queue to remove done items with todo = -1*/
            auto end = std::copy_if(std::execution::par, todo, todo + size, ReDoQueue, [](int p_i){return p_i >= 0;});
            tend = second();
            timecomp3 += timediff(tstart, tend);

            Niteration++;
            /* Now done with the current queue and the flags*/
            myfree(todo);
            myfree(CurQueue);
            size = end - ReDoQueue;

            /* We can stop if we are not updating hsml or if we are done.*/
            if(!update_hsml || !MPIU_Any(size > 0, comm)) {
                myfree(ReDoQueue);
                break;
            }

            double minngb, maxngb;
            MPI_Reduce(&maxnumngb, &maxngb, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
            MPI_Reduce(&minnumngb, &minngb, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
            message(0, "Iter=%d Max ngb=%g, min ngb=%g\n", Niteration, maxngb, minngb);
    #ifdef DEBUG
            if(size > 0 && size < 10 && Niteration > 20 ) {
                int pp = ReDoQueue[0];
                message(1, "Remaining i=%d, t %d, pos %g %g %g, hsml: %g\n", pp, parts[pp].Type, parts[pp].Pos[0], parts[pp].Pos[1], parts[pp].Pos[2], parts[pp].Hsml);
            }
    #endif
            if(size > 0 && Niteration > MAXITER) {
                endrun(1155, "failed to converge density for %ld particles\n", size);
            }
        } while(1);
    };

private:
    void ev_free_exports(int * exportcounts)
    {
        myfree(exportcounts);
    }

    /* returns struct containing export counts */
    void ev_primary(int * WorkSet, const int64_t WorkSetSize, QueryType * queries, particle_data * const parts)
    {
    #pragma omp parallel
        {
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
            #pragma omp for schedule(dynamic, chnksz) nowait
            for(k = 0; k < WorkSetSize; k++) {
                const int i = WorkSet ? WorkSet[k] : k;
                /* Primary never uses node list */
                QueryType& input = queries[k];
                ResultType result(input);
                LocalTreeWalkType lv(tree->Nodes, input);
                lv.template visit<TREEWALK_PRIMARY>(input, &result, priv, parts);
                result.template reduce<TREEWALK_PRIMARY>(i, output, parts);
            }
        }
    }

    int * ev_count_exports(int * WorkSet, const int64_t WorkSetSize, QueryType * queries)
    {
        int * exportcounts = mymalloc("Export counts", int, WorkSetSize);
        /* Count all entries. */
        #pragma omp parallel
        {
            LocalTopTreeWalkType lv(tree->Nodes, tree->TopLeaves, tree->NTopLeaves, tree->lastnode);
            #pragma omp for
            for(int k = 0 ; k < WorkSetSize; k++) {
                const int i = WorkSet ? WorkSet[k] : k;
                /* Note index k is into the WorkSet and i is the particle.*/
                exportcounts[k] = lv.template toptree_visit<TOPTREE_COUNT>(i, queries[k], priv, NULL, NULL, 0);
            }
        }
        /* Parallel inclusive scan */
        std::inclusive_scan(std::execution::par, exportcounts, exportcounts+WorkSetSize, exportcounts);
        return exportcounts;
    }

    int64_t ev_toptree(int * WorkSet, const int64_t WorkSetStart, const int64_t WorkSetSize, int * exportcounts, ExportMemory2<QueryType> * const exportlist, QueryType * queries)
    {
        /* Adjust the indices for the restart */
        int64_t curSize = WorkSetSize - WorkSetStart;
        int exportoffset = 0;
        if(WorkSetStart > 0)
            exportoffset = exportcounts[WorkSetStart-1];
        /* Handle the no work case explicitly */
        if(curSize == 0) {
            exportlist->Nexport = 0;
            exportlist->ExportTable = mymalloc("DataIndexTable", data_index, 1);
            return WorkSetStart;
        }
        exportcounts = exportcounts + WorkSetStart;
        /* Note that the exportcount is built on the first iteration
         * and so the indices are off for the second one. */
        exportlist->Nexport = exportcounts[curSize-1] - exportoffset;
        int BunchSize = compute_bunchsize() + exportoffset;
        Greater_than_BunchSize gtrbunch{BunchSize};
        auto iter = std::find_if(std::execution::par, exportcounts, exportcounts+curSize,gtrbunch);
        if(iter != exportcounts + curSize) {
            curSize = iter - exportcounts;
            if(curSize == 0)
                endrun(5, "Not enough export space to make progress! lastsuc %ld\n", WorkSetStart);
            exportlist->Nexport = exportcounts[curSize-1] - exportoffset;
            message(1, "Tree export buffer full with %lu exports (%lu Mbytes). BunchSize %d. First particle %ld new start: %ld size %ld.\n",
                exportlist->Nexport, exportlist->Nexport*sizeof(QueryType)/1024/1024, BunchSize, WorkSetStart, WorkSetStart + curSize, WorkSetSize);
        }
        /* Note this is the sum including the current element. */
        exportlist->ExportTable = mymalloc("DataIndexTable", data_index, exportlist->Nexport);
        exportlist->ExportQueries = mymalloc("ExportQueries", QueryType, exportlist->Nexport);

        /* Now we run toptree_visit again with the export offsets to make the export table.
         * Likely most particles have zero exports, so this will be somewhat faster than the first run. */

    #pragma omp parallel
        {
            LocalTopTreeWalkType lv(tree->Nodes, tree->TopLeaves, tree->NTopLeaves, tree->lastnode);

            #pragma omp for
            for(int k = 0; k < curSize; k++) {
                int64_t nexport = exportcounts[k] - exportoffset;
                data_index * currentexport = exportlist->ExportTable;
                QueryType * currentexportquery = exportlist->ExportQueries;
                if(k > 0) {
                    currentexport = &exportlist->ExportTable[exportcounts[k-1] - exportoffset];
                    currentexportquery = &currentexportquery[exportcounts[k-1] - exportoffset];
                    nexport = exportcounts[k] - exportcounts[k-1];
                }
                /* With no exports we can skip evaluating this particle */
                if(nexport == 0)
                    continue;
                const int i = WorkSet ? WorkSet[k+WorkSetStart] : k + WorkSetStart;
                QueryType& input = queries[k + WorkSetStart];
                /* Indexing into the WorkSet, not the particle.*/
                lv.template toptree_visit<TOPTREE_EXPORT>(i, input, priv, currentexport, currentexportquery, nexport);
            }
        }

        /* Start again with the next chunk not yet evaluated*/
        return WorkSetStart + curSize;
    }

    /* Perform evaluation of a chunk of secondary particles from a single processor.
     *
     * Arguments:
     * - QueryType imports: an array of querys sent from another rank for evaluation on the local tree.
     * - ResultType results: an array of results generated by walking the local tree, for returning to the original rank.
     * - nimports_task: size of the query and result arrays.
     *
    Takes the data within imports, which should be a pointer to nimports_task values */
    void ev_secondary(ResultType * results, QueryType * imports, const int64_t nimports_task, struct particle_data * const parts)
    {
        #pragma omp parallel for
        for(int64_t j = 0; j < nimports_task; j++) {
            QueryType * input = &(imports[j]);
            ResultType * resoutput = new (&results[j]) ResultType(*input);
            LocalTreeWalkType lv(tree->Nodes, *input);
            lv.template visit<TREEWALK_GHOSTS>(*input, resoutput, priv, parts);
        }
    }

    void ev_wait_secondary(CommBuffer * res_imports, CommBuffer * imports, ImpExpCounts<QueryType>* counts, struct particle_data * const parts)
    {
        res_imports->databuf = mymanagedmalloc("ImportResult", char, counts->Nimport * sizeof(ResultType));

        MPI_Datatype type;
        MPI_Type_contiguous(sizeof(ResultType), MPI_BYTE, &type);
        MPI_Type_commit(&type);
        res_imports->rdata_all = mymalloc("Import Return Requests", MPI_Request, imports->nrequest());
        int * complete_array = mymalloc("completes", int, imports->nrequest());

        /* Test each request in turn until it completes*/
        while(res_imports->nrequest() < imports->nrequest()) {
            int complete_cnt = MPI_UNDEFINED;
            /* Check for some completed requests: note that cleanup is performed if the requests are complete.
                * There may be only 1 completed request, and we need to wait again until we have more.*/
            MPI_Waitsome(imports->nrequest(), imports->rdata_all, &complete_cnt, complete_array, MPI_STATUSES_IGNORE);
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
                static_cast<DerivedType *>(this)->ev_secondary((ResultType *) dataresultstart, (QueryType *) databufstart, nimports_task, parts);
                /* Send the completed data back*/
                MPI_Isend(dataresultstart, nimports_task, type, task, 101923, counts->comm, &res_imports->rdata_all[res_imports->nrequest()]);
                res_imports->rqst_task.push_back(task);
            }
        };
        myfree(complete_array);
        MPI_Type_free(&type);
        return;
    }

    /* Builds the list of exported particles and async sends the export queries. */
    void ev_send_recv_export_import(const ImpExpCounts<QueryType> * const counts, const ExportMemory2<QueryType> * const exportlist, CommBuffer * exports, CommBuffer * imports)
    {
        exports->databuf = mymalloc("ExportQuery", char, counts->Nexport * sizeof(QueryType));
        imports->databuf = mymanagedmalloc("ImportQuery", char, counts->Nimport * sizeof(QueryType));

        MPI_Datatype type;
        MPI_Type_contiguous(sizeof(QueryType), MPI_BYTE, &type);
        MPI_Type_commit(&type);

        /* Post recvs before sends. This sometimes allows for a fastpath.*/
        imports->MPI_fill(counts->Import_count, counts->Import_offset, type, COMM_RECV, 101922, counts->comm);

        /* Transpose the export query list from being ordered by particle to being ordered by task for export*/
        int64_t * real_send_count = mymalloc("tmp_send_count", int64_t, counts->NTask);
        memset(real_send_count, 0, sizeof(int64_t)*counts->NTask);
        QueryType * export_queries = reinterpret_cast<QueryType*>(exports->databuf);
        for(size_t k = 0; k < exportlist->Nexport; k++) {
            const int task = exportlist->ExportTable[k].Task;
            const int64_t bufpos = real_send_count[task] + counts->Export_offset[task];
            real_send_count[task]++;
            /* Copy the query to the export buffer. Note the index of the queries list is the same as the index of the ExportTable */
            export_queries[bufpos] = exportlist->ExportQueries[k];
        }
    #ifdef DEBUG
    /* Checks!*/
        for(int i = 0; i < counts->NTask; i++)
            if(real_send_count[i] != counts->Export_count[i])
                endrun(6, "Inconsistent export to task %d of %d: %ld expected %ld\n", i, counts->NTask, real_send_count[i], counts->Export_count[i]);
    #endif
        myfree(real_send_count);
        exports->MPI_fill(counts->Export_count, counts->Export_offset, type, COMM_SEND, 101922, counts->comm);
        MPI_Type_free(&type);
        return;
    }

    /* Receive the export results */
    void ev_recv_export_result(CommBuffer * exportbuf, ImpExpCounts<QueryType> * counts)
    {
        MPI_Datatype type;
        MPI_Type_contiguous(sizeof(ResultType), MPI_BYTE, &type);
        MPI_Type_commit(&type);
        exportbuf->databuf = mymalloc2("ExportResult", char, counts->Nexport * sizeof(ResultType));
        /* Post the receives first so we can hit a zero-copy fastpath.*/
        exportbuf->MPI_fill(counts->Export_count, counts->Export_offset, type, COMM_RECV, 101923, counts->comm);
        // alloc_commbuffer(&res_imports, counts.NTask, 0);
        // MPI_fill_commbuffer(import, counts->Import_count, counts->Import_offset, type, COMM_SEND, 101923, counts->comm);
        MPI_Type_free(&type);
        return;
    }

    void ev_reduce_export_result(CommBuffer * exportbuf, const ImpExpCounts<QueryType> * const counts, const ExportMemory2<QueryType> * const exportlist, struct particle_data * const parts)
    {
        /* Notice that we build the dataindex table individually
            * on each thread, so we are ordered by particle and have memory locality.*/
        int * real_recv_count = mymalloc("tmp_recv_count", int, counts->NTask);
        memset(real_recv_count, 0, sizeof(int)*counts->NTask);
        for(size_t k = 0; k < exportlist->Nexport; k++) {
            const int place = exportlist->ExportTable[k].Index;
            const int task = exportlist->ExportTable[k].Task;
            const int64_t bufpos = real_recv_count[task] + counts->Export_offset[task];
            real_recv_count[task]++;
            ResultType * result = &((ResultType *) exportbuf->databuf)[bufpos];
            result->template reduce<TREEWALK_GHOSTS>(place, output, parts);
#ifdef DEBUG
            if(result->ID != parts[place].ID)
                endrun(8, "Error in communication: IDs mismatch %ld %ld\n", result->ID, parts[place].ID);
#endif
        }
        myfree(real_recv_count);
    }

    /* Checks whether all tasks have finished iterating */
    int ev_ndone(const int BufferFullFlag, MPI_Comm comm)
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
