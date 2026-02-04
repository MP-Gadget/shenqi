#include <mpi.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
#include <alloca.h>

#include "treewalk2.h"

template <typename QueryType, typename ResultType, typename LocalTreeWalkType, typename ParamType>
void
TreeWalk<QueryType, ResultType, LocalTreeWalkType, ParamType>::ev_begin(int * active_set, const size_t size, particle_data * const parts)
{
    /* Needs to be 64-bit so that the multiplication in Ngblist malloc doesn't overflow*/
    const size_t NumThreads = omp_get_max_threads();
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

    if(!NoNgblist)
        Ngblist = (int*) mymalloc("Ngblist", tree->NumParticles * NumThreads * sizeof(int));
    else
        Ngblist = NULL;

    /* Print some balance numbers*/
    int64_t nmin, nmax, total;
    MPI_Reduce(&WorkSetSize, &nmin, 1, MPI_INT64, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&WorkSetSize, &nmax, 1, MPI_INT64, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&WorkSetSize, &total, 1, MPI_INT64, MPI_SUM, 0, MPI_COMM_WORLD);
    message(0, "Treewalk %s iter %ld: total part %ld max/MPI: %ld min/MPI: %ld balance: %g query %ld result %ld BunchSize %ld.\n",
        ev_label, Niteration, total, nmax, nmin, (double)nmax/((total+0.001)/NTask), sizeof(QueryType), sizeof(ResultType),
        compute_bunchsize(sizeof(QueryType), sizeof(ResultType), ev_label));

    report_memory_usage(ev_label);
}

template <typename QueryType, typename ResultType, typename LocalTreeWalkType, typename ParamType>
void
TreeWalk<QueryType, ResultType, LocalTreeWalkType, ParamType>::build_queue(int * active_set, const size_t size, int may_have_garbage, const particle_data * const Parts)
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

/* returns struct containing export counts */
template <typename QueryType, typename ResultType, typename LocalTreeWalkType, typename ParamType>
void
TreeWalk<QueryType, ResultType, LocalTreeWalkType, ParamType>::ev_primary(const particle_data * const parts)
{
    int64_t maxNinteractions = 0, minNinteractions = 1L << 45, Ninteractions=0;
#pragma omp parallel reduction(min:minNinteractions) reduction(max:maxNinteractions) reduction(+: Ninteractions)
    {
        /* Note: exportflag is local to each thread */
        LocalTreeWalkType lv(TREEWALK_PRIMARY, tree, ev_label, Ngblist, ExportTable_thread);

        /* We must schedule dynamically so that we have reduced imbalance.
        * We do not need to worry about the export buffer filling up.*/
        /* chunk size: 1 and 1000 were slightly (3 percent) slower than 8.
        * FoF treewalk needs a larger chnksz to avoid contention.*/
        int64_t chnksz = WorkSetSize / (4*NThread);
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
            lv.visit(input, output, priv, parts);
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

template <typename QueryType, typename ResultType, typename LocalTreeWalkType, typename ParamType>
void
TreeWalk<QueryType, ResultType, LocalTreeWalkType, ParamType>::alloc_export_memory()
{
    Nexport_thread = ta_malloc2("localexports", size_t, NThread);
    ExportTable_thread = ta_malloc2("localexports", data_index *, NThread);
    int i;
    for(i = 0; i < NThread; i++)
        ExportTable_thread[i] = (data_index*) mymalloc("DataIndexTable", sizeof(data_index) * compute_bunchsize(sizeof(QueryType), sizeof(ResultType), ev_label));
    QueueChunkEnd = ta_malloc2("queueend", int64_t, NThread);
    for(i = 0; i < NThread; i++)
        QueueChunkEnd[i] = -1;
    QueueChunkRestart = ta_malloc2("queuerestart", int, NThread);
}

template <typename QueryType, typename ResultType, typename LocalTreeWalkType, typename ParamType>
void
TreeWalk<QueryType, ResultType, LocalTreeWalkType, ParamType>::free_export_memory()
{
    myfree(QueueChunkRestart);
    myfree(QueueChunkEnd);
    int i;
    for(i = NThread - 1; i >= 0; i--)
        myfree(ExportTable_thread[i]);
    myfree(ExportTable_thread);
    myfree(Nexport_thread);
}

template <typename QueryType, typename ResultType, typename LocalTreeWalkType, typename ParamType>
int
TreeWalk<QueryType, ResultType, LocalTreeWalkType, ParamType>::ev_toptree(const particle_data * const parts)
{
    BufferFullFlag = 0;
    int64_t currentIndex = WorkSetStart;
    int BufferFullFlag = 0;

    if(Nexportfull > 0)
        message(0, "Toptree %s, iter %ld. First particle %ld size %ld.\n", ev_label, Nexportfull, WorkSetStart, WorkSetSize);

#pragma omp parallel reduction(+: BufferFullFlag)
    {
        LocalTreeWalkType lv(TREEWALK_TOPTREE, tree, ev_label, Ngblist, ExportTable_thread);
        /* Signals a full export buffer on this thread*/
        int BufferFull_thread = 0;
        const int tid = omp_get_thread_num();

        ResultType output;

        /* We schedule dynamically so that we have reduced imbalance.
         * We do not use the openmp dynamic scheduling, but roll our own
         * so that we can break from the loop if needed.*/
        int64_t chnk = 0;
        /* chunk size: 1 and 1000 were slightly (3 percent) slower than 8.
         * FoF treewalk needs a larger chnksz to avoid contention.*/
        int64_t chnksz = WorkSetSize / (4*NThread);
        if(chnksz < 1)
            chnksz = 1;
        if(chnksz > 1000)
            chnksz = 1000;
        do {
            int64_t end;
            /* Restart a previously partially evaluated chunk if there is one*/
            if(Nexportfull > 0 && QueueChunkEnd[tid] > 0) {
                chnk = QueueChunkRestart[tid];
                end = QueueChunkEnd[tid];
                QueueChunkEnd[tid] = -1;
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
            if((WorkSetSize  < end + chnksz * NThread) && chnksz >= 2)
                chnksz /= 2;
            int k;
            for(k = chnk; k < end; k++) {
                const int i = WorkSet ? WorkSet[k] : k;
                /* Toptree never uses node list */
                QueryType input(parts[i], NULL, tree->firstnode, priv);
                lv.target = i;
                /* Reset the number of exported particles.*/
                const int rt = lv.visit(input, output, priv, parts);
                /* If we filled up, we need to save the partially evaluated chunk, and leave this loop.*/
                if(rt < 0) {
                    //message(5, "Export buffer full for particle %d chnk: %ld -> %ld on thread %d with %ld exports\n", i, chnk, end, tid, lv->NThisParticleExport);
                    /* export buffer has filled up, can't do more work.*/
                    BufferFull_thread = 1;
                    /* Store information for the current chunk, so we can resume successfully exactly where we left off.
                        Each thread stores chunk information */
                    QueueChunkRestart[tid] = k;
                    QueueChunkEnd[tid] = end;
                    break;
                }
            }
        } while(chnk < WorkSetSize && BufferFull_thread == 0);
        Nexport_thread[tid] = lv.Nexport;
        BufferFullFlag += BufferFull_thread;
    }

    if(BufferFullFlag > 0) {
        size_t Nexport = 0;
        int i;
        for(i = 0; i < NThread; i++)
            Nexport += Nexport_thread[i];
        message(1, "Tree export buffer full on %d of %ld threads with %lu exports (%lu Mbytes). First particle %ld new start: %ld size %ld.\n",
                        BufferFullFlag, NThread, Nexport, Nexport*sizeof(QueryType)/1024/1024, WorkSetStart, currentIndex, WorkSetSize);
        if(currentIndex == WorkSetStart)
            endrun(5, "Not enough export space to make progress! lastsuc %ld\n", currentIndex);
    }
    // else
        // message(1, "Finished toptree on %d threads. First particle %ld next start: %ld size %ld.\n", BufferFullFlag, WorkSetStart, currentIndex, WorkSetSize);
    /* Start again with the next chunk not yet evaluated*/
    WorkSetStart = currentIndex;
    return BufferFullFlag;
}

template <typename QueryType, typename ResultType, typename LocalTreeWalkType, typename ParamType>
void
TreeWalk<QueryType, ResultType, LocalTreeWalkType, ParamType>::ev_secondary(CommBuffer * res_imports, CommBuffer * imports, struct ImpExpCounts* counts, const struct particle_data * const parts)
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
                    LocalTreeWalkType lv(TREEWALK_GHOSTS, tree, ev_label, Ngblist, ExportTable_thread);
                    #pragma omp for
                    for(j = 0; j < nimports_task; j++) {
                        QueryType * input = ((QueryType *) databufstart)[j];
                        ResultType * output = new (results[j]) ResultType(*input);
                        lv.target = -1;
                        lv.visit(input, output, priv, parts);
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

template <typename QueryType, typename ResultType, typename LocalTreeWalkType, typename ParamType>
struct ImpExpCounts
TreeWalk<QueryType, ResultType, LocalTreeWalkType, ParamType>::
ev_export_import_counts(MPI_Comm comm)
{
    int NTask;
    struct ImpExpCounts counts = {0};
    MPI_Comm_size(comm, &NTask);
    counts.NTask = NTask;
    counts.comm = comm;
    counts.Export_count = ta_malloc("Tree_counts", int64_t, 4*NTask);
    counts.Export_offset = counts.Export_count + NTask;
    counts.Import_count = counts.Export_offset + NTask;
    counts.Import_offset = counts.Import_count + NTask;
    memset(counts.Export_count, 0, sizeof(int64_t)*4*NTask);

    int64_t i;
    counts.Nexport=0;
    /* Calculate the amount of data to send. */
    for(i = 0; i < NThread; i++)
    {
        int64_t * exportcount = counts.Export_count;
        size_t k;
        #pragma omp parallel for reduction(+: exportcount[:NTask])
        for(k = 0; k < Nexport_thread[i]; k++)
            exportcount[ExportTable_thread[i][k].Task]++;
        /* This is over all full buffers.*/
        Nexport_sum += Nexport_thread[i];
        /* This is the export count*/
        counts.Nexport += Nexport_thread[i];
    }
    /* Exchange the counts. Note this is synchronous so we need to ensure the toptree walk, which happens before this, is balanced.*/
    MPI_Alltoall(counts.Export_count, 1, MPI_INT64, counts.Import_count, 1, MPI_INT64, counts.comm);
    // message(1, "Exporting %ld particles. Thread 0 is %ld\n", counts.Nexport, Nexport_thread[0]);

    counts.Nimport = counts.Import_count[0];
    NExportTargets = (counts.Export_count[0] > 0);
    for(i = 1; i < NTask; i++)
    {
        counts.Nimport += counts.Import_count[i];
        counts.Export_offset[i] = counts.Export_offset[i - 1] + counts.Export_count[i - 1];
        counts.Import_offset[i] = counts.Import_offset[i - 1] + counts.Import_count[i - 1];
        NExportTargets += (counts.Export_count[i] > 0);
    }
    return counts;
}

/* Builds the list of exported particles and async sends the export queries. */
template <typename QueryType, typename ResultType, typename LocalTreeWalkType, typename ParamType>
void
TreeWalk<QueryType, ResultType, LocalTreeWalkType, ParamType>::ev_send_recv_export_import(struct ImpExpCounts * counts, CommBuffer * exports, CommBuffer * imports, const particle_data * const parts)
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
    for(i = 0; i < NThread; i++)
    {
        size_t k;
        for(k = 0; k < Nexport_thread[i]; k++) {
            const int place = ExportTable_thread[i][k].Index;
            const int task = ExportTable_thread[i][k].Task;
            const int64_t bufpos = real_send_count[task] + counts->Export_offset[task];
            real_send_count[task]++;
            /* Initialize the query in this memory */
            QueryType * input = new(export_queries[bufpos]) QueryType(parts[place], ExportTable_thread[i][k].NodeList, -1, priv);
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

template <typename QueryType, typename ResultType, typename LocalTreeWalkType, typename ParamType>
void
TreeWalk<QueryType, ResultType, LocalTreeWalkType, ParamType>::ev_reduce_export_result(CommBuffer * exportbuf, struct ImpExpCounts * counts, const struct particle_data * const parts)
{
    int64_t i;
    /* Notice that we build the dataindex table individually
     * on each thread, so we are ordered by particle and have memory locality.*/
    int * real_recv_count = ta_malloc("tmp_recv_count", int, NTask);
    memset(real_recv_count, 0, sizeof(int)*NTask);
    for(i = 0; i < NThread; i++)
    {
        size_t k;
        for(k = 0; k < Nexport_thread[i]; k++) {
            const int place = ExportTable_thread[i][k].Index;
            const int task = ExportTable_thread[i][k].Task;
            const int64_t bufpos = real_recv_count[task] + counts->Export_offset[task];
            real_recv_count[task]++;
            ResultType * output = ((ResultType *) exportbuf->databuf)[bufpos];
            output->reduce(place, TREEWALK_GHOSTS, priv, parts);
#ifdef DEBUG
            if(output->ID != parts[place].ID)
                endrun(8, "Error in communication: IDs mismatch %ld %ld\n", output->ID, parts[place].ID);
#endif
        }
    }
    myfree(real_recv_count);
}

/* find the closest index from radius and numNgb list, update left and right bound, return new hsml */
double
ngb_narrow_down(double *right, double *left, const double *radius, const double *numNgb, int maxcmpt, int desnumngb, int *closeidx, double BoxSize)
{
    int j;
    int close = 0;
    double ngbdist = fabs(numNgb[0] - desnumngb);
    for(j = 1; j < maxcmpt; j++){
        double newdist = fabs(numNgb[j] - desnumngb);
        if(newdist < ngbdist){
            ngbdist = newdist;
            close = j;
        }
    }
    if(closeidx)
        *closeidx = close;

    for(j = 0; j < maxcmpt; j++){
        if(numNgb[j] < desnumngb)
            *left = radius[j];
        if(numNgb[j] > desnumngb){
            *right = radius[j];
            break;
        }
    }

    double hsml = radius[close];

    if(*right > 0.99 * BoxSize){
        double dngbdv = 0;
        if(maxcmpt > 1 && (radius[maxcmpt-1]>radius[maxcmpt-2]))
            dngbdv = (numNgb[maxcmpt-1]-numNgb[maxcmpt-2])/(pow(radius[maxcmpt-1],3) - pow(radius[maxcmpt-2],3));
        /* Increase hsml by a maximum factor to avoid madness. We can be fairly aggressive about this factor.*/
        double newhsml = 4 * hsml;
        if(dngbdv > 0) {
            double dngb = (desnumngb - numNgb[maxcmpt-1]);
            double newvolume = pow(hsml,3) + dngb / dngbdv;
            if(pow(newvolume, 1./3) < newhsml)
                newhsml = pow(newvolume, 1./3);
        }
        hsml = newhsml;
    }
    if(hsml > *right)
        hsml = *right;

    if(*left == 0) {
        /* Extrapolate using volume, ie locally constant density*/
        double dngbdv = 0;
        if(radius[1] > radius[0])
            dngbdv = (numNgb[1] - numNgb[0]) / (pow(radius[1],3) - pow(radius[0],3));
        /* Derivative is not defined for minimum, so use 0.*/
        if(maxcmpt == 1 && radius[0] > 0)
            dngbdv = numNgb[0] / pow(radius[0],3);

        if(dngbdv > 0) {
            double dngb = desnumngb - numNgb[0];
            double newvolume = pow(hsml,3) + dngb / dngbdv;
            hsml = pow(newvolume, 1./3);
        }
    }
    if(hsml < *left)
        hsml = *left;

    return hsml;
}
