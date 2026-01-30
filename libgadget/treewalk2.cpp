#include <mpi.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
#include <alloca.h>
#include "utils.h"

#include "treewalk2.h"
#include "partmanager.h"
#include "domain.h"
#include "forcetree.h"

void
TreeWalk::ev_begin(int * active_set, const size_t size)
{
    /* Needs to be 64-bit so that the multiplication in Ngblist malloc doesn't overflow*/
    const size_t NumThreads = omp_get_max_threads();
    MPI_Comm_size(MPI_COMM_WORLD, &NTask);
    /* The last argument is may_have_garbage: in practice the only
     * trivial haswork is the gravtree. This has no (active) garbage because
     * the active list was just rebuilt, but on a PM step the active list is NULL
     * and we may still have swallowed BHs around. So in practice this avoids
     * computing gravtree for swallowed BHs on a PM step.*/
    int may_have_garbage = 0;
    /* Note this is not collective, but that should not matter.*/
    if(!active_set && SlotsManager->info[5].size > 0)
        may_have_garbage = 1;
    build_queue(active_set, size, may_have_garbage);

    /* Start first iteration at the beginning*/
    WorkSetStart = 0;

    if(!NoNgblist)
        Ngblist = (int*) mymalloc("Ngblist", tree->NumParticles * NumThreads * sizeof(int));
    else
        Ngblist = NULL;

    /* Assert that the query and result structures are aligned to  64-bit boundary,
     * so that our MPI Send/Recv's happen from aligned memory.*/
    if(query_type_elsize % 8 != 0)
        endrun(0, "Query structure has size %ld, not aligned to 64-bit boundary.\n", query_type_elsize);
    if(result_type_elsize % 8 != 0)
        endrun(0, "Result structure has size %ld, not aligned to 64-bit boundary.\n", result_type_elsize);

    /* Print some balance numbers*/
    int64_t nmin, nmax, total;
    MPI_Reduce(&WorkSetSize, &nmin, 1, MPI_INT64, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&WorkSetSize, &nmax, 1, MPI_INT64, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&WorkSetSize, &total, 1, MPI_INT64, MPI_SUM, 0, MPI_COMM_WORLD);
    message(0, "Treewalk %s iter %ld: total part %ld max/MPI: %ld min/MPI: %ld balance: %g query %ld result %ld BunchSize %ld.\n",
        ev_label, Niteration, total, nmax, nmin, (double)nmax/((total+0.001)/NTask), query_type_elsize, result_type_elsize, compute_bunchsize(query_type_elsize, result_type_elsize, ev_label));

    report_memory_usage(ev_label);
}

void TreeWalk::ev_finish(void)
{
    if(Ngblist)
        myfree(Ngblist);
    if(!work_set_stolen_from_active)
        myfree(WorkSet);
}

void
TreeWalk::init_query(TreeWalkQueryBase * query, int i, const int * const NodeList)
{
#ifdef DEBUG
    query->ID = Part[i].ID;
#endif

    int d;
    for(d = 0; d < 3; d ++) {
        query->Pos[d] = Part[i].Pos[d];
    }

    if(NodeList) {
        memcpy(query->NodeList, NodeList, sizeof(query->NodeList[0]) * NODELISTLENGTH);
    } else {
        query->NodeList[0] = tree->firstnode; /* root node */
        query->NodeList[1] = -1; /* terminate immediately */
    }

    fill(i, query);
}
void
TreeWalk::init_result(TreeWalkResultBase * result, TreeWalkQueryBase * query)
{
    memset(result, 0, result_type_elsize);
#ifdef DEBUG
    result->ID = query->ID;
#endif
}

void
TreeWalk::reduce_result(TreeWalkResultBase * result, int i, TreeWalkReduceMode mode)
{
    reduce(i, result, mode);
#ifdef DEBUG
    if(Part[i].ID != result->ID)
        endrun(2, "Mismatched ID (%ld != %ld) for particle %d in treewalk reduction, mode %d\n", Part[i].ID, result->ID, i, mode);
#endif
}

void
TreeWalk::build_queue(int * active_set, const size_t size, int may_have_garbage)
{
    NThread = omp_get_max_threads();

    if(!haswork_defined && !may_have_garbage)
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

            /* Skip the garbage /swallowed particles */
            if(Part[p_i].IsGarbage || Part[p_i].Swallowed)
                continue;

            if(haswork_defined && !haswork(p_i))
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
void
TreeWalk::ev_primary(void)
{
    int64_t maxNinteractions = 0, minNinteractions = 1L << 45, Ninteractions=0;
#pragma omp parallel reduction(min:minNinteractions) reduction(max:maxNinteractions) reduction(+: Ninteractions)
    {
        /* Note: exportflag is local to each thread */
        LocalTreeWalk<> lv(TREEWALK_PRIMARY, tree, ev_label, Ngblist, ExportTable_thread);

        /* use old index to recover from a buffer overflow*/;
        TreeWalkQueryBase * input = (TreeWalkQueryBase *) alloca(query_type_elsize);
        TreeWalkResultBase * output = (TreeWalkResultBase *) alloca(result_type_elsize);
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
            init_query(input, i, NULL);
            init_result(output, input);
            lv.target = i;
            lv.visit(input, output);
            reduce_result(output, i, TREEWALK_PRIMARY);
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

int TreeWalk::ev_ndone(MPI_Comm comm)
{
    int ndone;
    int done = !(BufferFullFlag);
    MPI_Allreduce(&done, &ndone, 1, MPI_INT, MPI_SUM, comm);
    return ndone;
}

void
TreeWalk::alloc_export_memory()
{
    Nexport_thread = ta_malloc2("localexports", size_t, NThread);
    ExportTable_thread = ta_malloc2("localexports", data_index *, NThread);
    int i;
    for(i = 0; i < NThread; i++)
        ExportTable_thread[i] = (data_index*) mymalloc("DataIndexTable", sizeof(data_index) * compute_bunchsize(query_type_elsize, result_type_elsize, ev_label));
    QueueChunkEnd = ta_malloc2("queueend", int64_t, NThread);
    for(i = 0; i < NThread; i++)
        QueueChunkEnd[i] = -1;
    QueueChunkRestart = ta_malloc2("queuerestart", int, NThread);
}

void
TreeWalk::free_export_memory()
{
    myfree(QueueChunkRestart);
    myfree(QueueChunkEnd);
    int i;
    for(i = NThread - 1; i >= 0; i--)
        myfree(ExportTable_thread[i]);
    myfree(ExportTable_thread);
    myfree(Nexport_thread);
}

int
TreeWalk::ev_toptree()
{
    BufferFullFlag = 0;
    int64_t currentIndex = WorkSetStart;
    int BufferFullFlag = 0;

    if(Nexportfull > 0)
        message(0, "Toptree %s, iter %ld. First particle %ld size %ld.\n", ev_label, Nexportfull, WorkSetStart, WorkSetSize);

#pragma omp parallel reduction(+: BufferFullFlag)
    {
        LocalTreeWalk<> lv(TREEWALK_TOPTREE, tree, ev_label, Ngblist, ExportTable_thread);
        /* Signals a full export buffer on this thread*/
        int BufferFull_thread = 0;
        const int tid = omp_get_thread_num();

        TreeWalkQueryBase * input = (TreeWalkQueryBase *) alloca(query_type_elsize);
        TreeWalkResultBase * output = (TreeWalkResultBase *) alloca(result_type_elsize);

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
                init_query(input, i, NULL);
                lv.target = i;
                /* Reset the number of exported particles.*/
                const int rt = lv.visit(input, output);
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
                        BufferFullFlag, NThread, Nexport, Nexport*query_type_elsize/1024/1024, WorkSetStart, currentIndex, WorkSetSize);
        if(currentIndex == WorkSetStart)
            endrun(5, "Not enough export space to make progress! lastsuc %ld\n", currentIndex);
    }
    // else
        // message(1, "Finished toptree on %d threads. First particle %ld next start: %ld size %ld.\n", BufferFullFlag, WorkSetStart, currentIndex, WorkSetSize);
    /* Start again with the next chunk not yet evaluated*/
    WorkSetStart = currentIndex;
    return BufferFullFlag;
}

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

struct CommBuffer
{
    char * databuf;
    int * rqst_task;
    MPI_Request * rdata_all;
    int nrequest_all;
};

void alloc_commbuffer(struct CommBuffer * buffer, int NTask, int alloc_high)
{
    if(alloc_high) {
        buffer->rdata_all = ta_malloc2("requests", MPI_Request, NTask);
        buffer->rqst_task = ta_malloc2("rqst", int, NTask);
    }
    else {
        buffer->rdata_all = ta_malloc("requests", MPI_Request, NTask);
        buffer->rqst_task = ta_malloc("rqst", int, NTask);
    }
    buffer->nrequest_all = 0;
    buffer->databuf = NULL;
}

void free_impexpcount(struct ImpExpCounts * count)
{
    ta_free(count->Export_count);
}

void free_commbuffer(struct CommBuffer * buffer)
{
    if(buffer->databuf) {
        myfree(buffer->databuf);
        buffer->databuf = NULL;
    }
    ta_free(buffer->rqst_task);
    ta_free(buffer->rdata_all);
}

#define COMM_RECV 1
#define COMM_SEND 0

/* Routine to send data to all tasks async. If receive is set, the routine receives data. The structure stores the requests.
 Empty tasks are skipped. Must call alloc_commbuffer on the buffer first and buffer->databuf must be set.*/
void
MPI_fill_commbuffer(struct CommBuffer * buffer, int64_t *cnts, int64_t *displs, MPI_Datatype type, int receive, int tag, MPI_Comm comm)
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
        buffer->rqst_task[nrequests] = target;
        if(receive == COMM_RECV) {
            MPI_Irecv(((char*) buffer->databuf) + elsize * displs[target], cnts[target],
                type, target, tag, comm, &buffer->rdata_all[nrequests++]);
        }
        else {
            MPI_Isend(((char*) buffer->databuf) + elsize * displs[target], cnts[target],
                type, target, tag, comm, &buffer->rdata_all[nrequests++]);
        }
    }
    buffer->nrequest_all = nrequests;
}

/* Waits for all the requests in the bufferbuffer to be complete*/
void TreeWalk::wait_commbuffer(struct CommBuffer * buffer)
{
    MPI_Waitall(buffer->nrequest_all, buffer->rdata_all, MPI_STATUSES_IGNORE);
}

struct CommBuffer TreeWalk::ev_secondary(struct CommBuffer * imports, struct ImpExpCounts* counts)
{
    struct CommBuffer res_imports = {0};
    alloc_commbuffer(&res_imports, counts->NTask, 1);
    res_imports.databuf = (char *) mymalloc2("ImportResult", counts->Nimport * result_type_elsize);

    MPI_Datatype type;
    MPI_Type_contiguous(result_type_elsize, MPI_BYTE, &type);
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
            char * databufstart = imports->databuf + counts->Import_offset[task] * query_type_elsize;
            char * dataresultstart = res_imports.databuf + counts->Import_offset[task] * result_type_elsize;
            /* This sends each set of imports to a parallel for loop. This may lead to suboptimal resource allocation if only a small number of imports come from a processor.
            * If there are a large number of importing ranks each with a small number of imports, a better scheme could be to send each chunk to a separate openmp task.
            * However, each openmp task by default only uses 1 thread. One may explicitly enable openmp nested parallelism, but I think that is not safe,
            * or it would be enabled by default.*/
            #pragma omp parallel
                {
                    int64_t j;
                    LocalTreeWalk<> lv(TREEWALK_GHOSTS, tree, ev_label, Ngblist, ExportTable_thread);
                    #pragma omp for
                    for(j = 0; j < nimports_task; j++) {
                        TreeWalkQueryBase * input = (TreeWalkQueryBase *) (databufstart + j * query_type_elsize);
                        TreeWalkResultBase * output = (TreeWalkResultBase *) (dataresultstart + j * result_type_elsize);
                        init_result(output, input);
                        lv.target = -1;
                        lv.visit(input, output);
                    }
                }
            /* Send the completed data back*/
            res_imports.rqst_task[res_imports.nrequest_all] = task;
            MPI_Isend(dataresultstart, nimports_task, type, task, 101923, counts->comm, &res_imports.rdata_all[res_imports.nrequest_all++]);
            tot_completed++;
        }
    };
    myfree(complete_array);
    MPI_Type_free(&type);
    return res_imports;
}

struct ImpExpCounts
TreeWalk::ev_export_import_counts(MPI_Comm comm)
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
void TreeWalk::ev_send_recv_export_import(struct ImpExpCounts * counts, struct CommBuffer * exports, struct CommBuffer * imports)
{
    alloc_commbuffer(exports, counts->NTask, 0);
    exports->databuf = (char *) mymalloc("ExportQuery", counts->Nexport * query_type_elsize);

    alloc_commbuffer(imports, counts->NTask, 0);
    imports->databuf = (char *) mymalloc("ImportQuery", counts->Nimport * query_type_elsize);

    MPI_Datatype type;
    MPI_Type_contiguous(query_type_elsize, MPI_BYTE, &type);
    MPI_Type_commit(&type);

    /* Post recvs before sends. This sometimes allows for a fastpath.*/
    MPI_fill_commbuffer(imports, counts->Import_count, counts->Import_offset, type, COMM_RECV, 101922, counts->comm);

    /* prepare particle data for export */
    int64_t * real_send_count = ta_malloc("tmp_send_count", int64_t, NTask);
    memset(real_send_count, 0, sizeof(int64_t)*NTask);
    int64_t i;
    for(i = 0; i < NThread; i++)
    {
        size_t k;
        for(k = 0; k < Nexport_thread[i]; k++) {
            const int place = ExportTable_thread[i][k].Index;
            const int task = ExportTable_thread[i][k].Task;
            const int64_t bufpos = real_send_count[task] + counts->Export_offset[task];
            TreeWalkQueryBase * input = (TreeWalkQueryBase*) (exports->databuf + bufpos * query_type_elsize);
            real_send_count[task]++;
            init_query(input, place, ExportTable_thread[i][k].NodeList);
        }
    }
#ifdef DEBUG
/* Checks!*/
    for(i = 0; i < NTask; i++)
        if(real_send_count[i] != counts->Export_count[i])
            endrun(6, "Inconsistent export to task %ld of %d: %ld expected %ld\n", i, NTask, real_send_count[i], counts->Export_count[i]);
#endif
    myfree(real_send_count);
    MPI_fill_commbuffer(exports, counts->Export_count, counts->Export_offset, type, COMM_SEND, 101922, counts->comm);
    MPI_Type_free(&type);
    return;
}

void TreeWalk::ev_recv_export_result(struct CommBuffer * exportbuf, struct ImpExpCounts * counts)
{
    alloc_commbuffer(exportbuf, counts->NTask, 1);
    MPI_Datatype type;
    MPI_Type_contiguous(result_type_elsize, MPI_BYTE, &type);
    MPI_Type_commit(&type);
    exportbuf->databuf = (char*) mymalloc2("ExportResult", counts->Nexport * result_type_elsize);
    /* Post the receives first so we can hit a zero-copy fastpath.*/
    MPI_fill_commbuffer(exportbuf, counts->Export_count, counts->Export_offset, type, COMM_RECV, 101923, counts->comm);
    // alloc_commbuffer(&res_imports, counts.NTask, 0);
    // MPI_fill_commbuffer(import, counts->Import_count, counts->Import_offset, type, COMM_SEND, 101923, counts->comm);
    MPI_Type_free(&type);
}

void TreeWalk::ev_reduce_export_result(struct CommBuffer * exportbuf, struct ImpExpCounts * counts)
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
            TreeWalkResultBase * output = (TreeWalkResultBase*) (exportbuf->databuf + result_type_elsize * bufpos);
            reduce_result(output, place, TREEWALK_GHOSTS);
#ifdef DEBUG
            if(output->ID != Part[place].ID)
                endrun(8, "Error in communication: IDs mismatch %ld %ld\n", output->ID, Part[place].ID);
#endif
        }
    }
    myfree(real_recv_count);
}

/* run a treewalk on an active_set.
 *
 * active_set : a list of indices of particles. If active_set is NULL,
 *              all (NumPart) particles are used.
 *
 * */
void
TreeWalk::run(int * active_set, size_t size)
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

    {
        int64_t i;
        #pragma omp parallel for
        for(i = 0; i < WorkSetSize; i ++) {
            const int p_i = WorkSet ? WorkSet[i] : i;
            preprocess(p_i);
        }
    }

    tend = second();
    timecomp3 += timediff(tstart, tend);

    {
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
            struct CommBuffer exports = {0}, imports = {0};
            /* exports is allocated first, then imports*/
            ev_send_recv_export_import(&counts, &exports, &imports);
            tend = second();
            timecomp0 += timediff(tstart, tend);
            /* Only do this on the first iteration, as we only need to do it once.*/
            tstart = second();
            if(Nexportfull == 0)
                ev_primary(); /* do local particles and prepare export list */
            tend = second();
            timecomp1 += timediff(tstart, tend);
            /* Do processing of received particles. We implement a queue that
             * checks each incoming task in turn and processes them as they arrive.*/
            tstart = second();
            /* Posts recvs to get the export results (which are sent in ev_secondary).*/
            struct CommBuffer res_exports = {0};
            ev_recv_export_result(&res_exports, &counts);
            struct CommBuffer res_imports = ev_secondary(&imports, &counts);
            // report_memory_usage(ev_label);
            free_commbuffer(&imports);
            tend = second();
            timecomp2 += timediff(tstart, tend);
            /* Now clear the sent data buffer, waiting for the send to complete.
             * This needs to be after the other end has called recv.*/
            tstart = second();
            wait_commbuffer(&res_exports);
            tend = second();
            timewait1 += timediff(tstart, tend);
            tstart = second();
            ev_reduce_export_result(&res_exports, &counts);
            wait_commbuffer(&exports);
            free_commbuffer(&exports);
            wait_commbuffer(&res_imports);
            tend = second();
            timecommsumm += timediff(tstart, tend);
            free_commbuffer(&res_imports);
            free_commbuffer(&res_exports);
            free_impexpcount(&counts);
            /* Free export memory*/
            Nexportfull++;
            /* Note there is no sync at the end!*/
        } while(Ndone < NTask);
        free_export_memory();
    }

    tstart = second();
    {
        int64_t i;
        #pragma omp parallel for
        for(i = 0; i < WorkSetSize; i ++) {
            const int p_i = WorkSet ? WorkSet[i] : i;
            postprocess(p_i);
        }
    }
    tend = second();
    timecomp3 += timediff(tstart, tend);
    ev_finish();
    Niteration++;
}

/* This function does treewalk_run in a loop, allocating a queue to allow some particles to be redone.
 * This loop is used primarily in density estimation.*/
void
TreeWalk::do_hsml_loop(int * queue, int64_t queuesize, int update_hsml)
{
    int NumThreads = omp_get_max_threads();
    maxnumngb = ta_malloc("numngb", double, NumThreads);
    minnumngb = ta_malloc("numngb2", double, NumThreads);

    /* Build the first queue */
    double tstart = second();
    build_queue(queue, queuesize, 0);
    double tend = second();

    /* Next call to treewalk_run will over-write these pointers*/
    int64_t size = WorkSetSize;
    int * ReDoQueue = WorkSet;
    /* First queue is allocated low*/
    int alloc_high = 0;
    /* We don't need to redo the queue generation
     * but need to keep track of allocated memory.*/
    bool orig_haswork_defined = haswork_defined;
    haswork_defined = false;
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
        run(CurQueue, size);

        /* Now done with the current queue*/
        if(orig_haswork_defined || Niteration > 1)
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

        /*
        if(ntot < 1 ) {
            foreach(ActiveParticle)
            {
                if(density_haswork(i)) {
                    MyFloat Left = DENSITY_GET_PRIV(tw)->Left[i];
                    MyFloat Right = DENSITY_GET_PRIV(tw)->Right[i];
                    message (1, "i=%d task=%d ID=%llu type=%d, Hsml=%g Left=%g Right=%g Ngbs=%g Right-Left=%g\n   pos=(%g|%g|%g)\n",
                         i, ThisTask, Part[i].ID, Part[i].Type, Part[i].Hsml, Left, Right,
                         (float) Part[i].NumNgb, Right - Left, Part[i].Pos[0], Part[i].Pos[1], Part[i].Pos[2]);
                }
            }

        }
        */
#ifdef DEBUG
        if(size < 10 && Niteration > 20 ) {
            int pp = ReDoQueue[0];
            message(1, "Remaining i=%d, t %d, pos %g %g %g, hsml: %g\n", pp, Part[pp].Type, Part[pp].Pos[0], Part[pp].Pos[1], Part[pp].Pos[2], Part[pp].Hsml);
        }
#endif

        if(size > 0 && Niteration > MAXITER) {
            endrun(1155, "failed to converge density for %ld particles\n", size);
        }
    } while(1);
    ta_free(minnumngb);
    ta_free(maxnumngb);
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

void
TreeWalk::print_stats(void)
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
