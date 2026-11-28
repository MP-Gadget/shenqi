#ifndef TREEWALK2_CUH
#define TREEWALK2_CUH
/* This file contains the GPU specialisations for the treewalk, and launches the CUDA kernels.
 * The template parameter types are specialisations of the functions in localtreewalk2.h.
 *
 * Functions here are compiled as CUDA by nvcc, and launch __global__ kernels for the treewalk.
 * Functions here may not use global memory.
 *
 * Note that the NVCC compiler DOES NOT support OpenMP! Because the functions here
 * are implemented as a compile-time template, any treewalk function using OpenMP
 * and not over-ridden will not be parallelized.
 *
 * Functions here should call cudaDeviceSynchronize() and check the return code for the kernel error status,
 * so that we are sure the result is available.
 *
 * Kernels may not call endrun() or message(), but the template treewalk functions should if there is a CUDA error.
 * CUDA errors are generally not recoverable.
 */
#include "utils/mymalloc.h"
#include "treewalk2.h"

#include <cstring>
#include <cuda_runtime.h>           // For CUDA runtime API functions.
#include <device_launch_parameters.h>  // To support device-related parameters.
#include <thrust/copy.h>                        // thrust::copy_if
#include <thrust/execution_policy.h>            // thrust::device
#include <thrust/scan.h>            // thrust::exclusive_scan
#include <thrust/find.h>            // thrust::find_if
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <algorithm>
#include <vector>
#include <thrust/binary_search.h>   // thrust::upper_bound
#ifdef DEBUG
#include <thrust/functional.h>        // thrust::plus
#include <thrust/transform_reduce.h>
#include <vector>
#endif

/* Allocate bytes of plain device memory for name. If the device is full, fall back to managed memory,
 * which the driver can page out to the host: slower, but the run continues. cudaFree frees either kind.
 * CUDA-aware MPI libraries (e.g. via UCX) also accept managed memory.*/
inline void * treewalk_device_malloc(const size_t bytes, const char * const name)
{
    void * ptr = NULL;
    cudaError_t err = cudaMalloc(&ptr, bytes);
    if(err == cudaErrorMemoryAllocation) {
        /* Clear the error so later error checks do not report it.*/
        cudaGetLastError();
        message(1, "Device memory full when allocating %lu bytes for %s: using managed memory instead\n", bytes, name);
        err = cudaMallocManaged(&ptr, bytes, cudaMemAttachGlobal);
    }
    if(err != cudaSuccess)
        endrun(5, "Failed to allocate %lu bytes of device memory for %s: %s\n", bytes, name, cudaGetErrorString(err));
    return ptr;
}

/* Extract the destination task from an entry in the export table.*/
struct ExportTaskOf
{
    MYCUDAFN int operator()(const data_index& entry) const {
        return entry.Task;
    }
};

/* Sum the results returned for the exports of each particle into results.
 * ExportTable is ordered by Index (do_toptree_exports writes the exports of work set entry tid at
 * the inclusive scan exportcounts[tid-1]), so the exports of one particle are a contiguous run.
 * The first entry of each run sums the whole run, serially and in export table order.
 * This means there are no write races and the additions happen in the same order as the CPU loop
 * in TreeWalk::ev_reduce_export_result.*/
template <typename ResultType>
struct ReduceExportRun
{
    ResultType * results;
    const ResultType * recvbuf;
    const data_index * ExportTable;
    /* bufpos[k] is the position of the result for export k in recvbuf.*/
    const int * bufpos;
    size_t Nexport;

    MYCUDAFN void operator()(const size_t k) const
    {
        const int place = ExportTable[k].Index;
        /* Not the start of a run: the thread for the start of the run does this entry.*/
        if(k > 0 && ExportTable[k-1].Index == place)
            return;
        ResultType res = results[place];
        for(size_t kk = k; kk < Nexport && ExportTable[kk].Index == place; kk++)
            res += recvbuf[bufpos[kk]];
        results[place] = res;
    }
};

#ifdef DEBUG
/* Returns 1 if the result returned for export k is not for the particle it is reduced onto.
 * TreeWalkResultBase::operator+= does not check this when compiled by nvcc.*/
template <typename ResultType>
struct ExportIDMismatch
{
    const ResultType * results;
    const ResultType * recvbuf;
    const data_index * ExportTable;
    const int * bufpos;

    MYCUDAFN int64_t operator()(const size_t k) const {
        return results[ExportTable[k].Index].ID != recvbuf[bufpos[k]].ID;
    }
};

/* Returns 1 if export k is before export k-1 in the work set: the reduction needs ExportTable sorted by Index.*/
struct ExportIndexUnsorted
{
    const data_index * ExportTable;

    MYCUDAFN int64_t operator()(const size_t k) const {
        return k > 0 && ExportTable[k].Index < ExportTable[k-1].Index;
    }
};
#endif

/* Each thread counts the number of exports needed for each particle.
 * counts are stored in the exportcounts argument.
 * We then use these counts to allocate memory and fill the export table. */
template <typename QueryType, typename LocalTopTreeWalkType, typename ParamType>
__global__ void count_toptree_exports(
    // The memory here should be allocated cudaManagedMalloc
    QueryType * const queries,
    const typename LocalTopTreeWalkType::node_type * const Nodes,
    const topleaf_data * const TopLeaves,
    const int * const TopLeafNodes,
    const int NTopLeaves,
    const int64_t WorkSetSize,
    // by reference so the destructor does not run,
    // which means in managed memory, and with all sub-arrays in managed memory
    const ParamType * priv,
    int * exportcounts)   // output: flags[i] = is the number of exports needed for this particle.
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= WorkSetSize)
        return;

    LocalTopTreeWalkType lv(Nodes, TopLeaves, TopLeafNodes, NTopLeaves);
    const int rt = lv.template toptree_visit<TOPTREE_COUNT>(queries[tid], tid, *priv, NULL, NULL, 0);
    exportcounts[tid] = rt;
}

template <typename QueryType, typename LocalTopTreeWalkType, typename ParamType>
__global__
void do_toptree_exports(
    // The memory here should be allocated cudaManagedMalloc
    QueryType * const queries,
    const typename LocalTopTreeWalkType::node_type * const Nodes,
    const topleaf_data * const TopLeaves,
    const int * const TopLeafNodes,
    const int NTopLeaves,
    const int WorkSetStart,
    const int64_t WorkSetSize,
    // by reference so the destructor does not run,
    // which means in managed memory, and with all sub-arrays in managed memory
    const ParamType * priv,
    const int * exportcounts,
    const int exportoffset,
    data_index * ExportTable, // These are the main export table output.
    QueryType * ExportQueries
    )
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= WorkSetSize)
        return;

    LocalTopTreeWalkType lv(Nodes, TopLeaves, TopLeafNodes, NTopLeaves);

    int64_t nexport = exportcounts[tid] - exportoffset;
    data_index * currentexport = ExportTable;
    QueryType * currentexportquery = ExportQueries;
    if(tid > 0) {
        currentexport = &ExportTable[exportcounts[tid-1] - exportoffset];
        currentexportquery = &ExportQueries[exportcounts[tid-1] - exportoffset];
        nexport = exportcounts[tid] - exportcounts[tid-1];
    }
    /* With no exports we can skip evaluating this particle */
    if(nexport == 0)
        return;
    /* This will save exports to the memory in ExportTable[exportoffsets[tid-1]].
     * We ignore return as it is the same as exportcounts. We arranged never to overflow.*/
    lv.template toptree_visit<TOPTREE_EXPORT>(queries[tid + WorkSetStart], tid + WorkSetStart, *priv, currentexport, currentexportquery, nexport);
};

template <typename QueryType, typename ResultType, typename LocalTreeWalkType, typename ParamType, TreeWalkReduceMode mode>
__global__
void treewalk_visit_kernel(
    // The memory here should be allocated cudaManagedMalloc
    QueryType * const queries,
    ResultType * const results,
    const int64_t WorkSetSize,
    particle_data * const parts,
    const typename LocalTreeWalkType::node_type * const Nodes,
    // by reference so the destructor does not run,
    // which means in managed memory, and with all sub-arrays in managed memory
    const ParamType * priv)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= WorkSetSize)
        return;

    LocalTreeWalkType lv(Nodes, queries[tid]);
    results[tid] = lv.template visit<mode>(queries[tid], *priv, parts);
};

template <typename ResultType, typename ParamType, typename OutputType>
__global__
void treewalk_postprocess_kernel(
    // The memory here should be allocated cudaManagedMalloc
    particle_data * const parts,
    const int * WorkSet,
    const int64_t WorkSetSize,
    ResultType * results,
    const ParamType * priv,
    OutputType * output)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= WorkSetSize)
        return;

    const int p_i = WorkSet ? WorkSet[tid] : tid;
    output->postprocess(p_i, results[tid], parts, priv);
}

template <typename DerivedType, typename QueryType, typename ResultType, typename LocalTreeWalkType, typename LocalTopTreeWalkType, typename ParamType, typename OutputType, typename ForceTreeType>
class TreeWalkGPU: public TreeWalk<DerivedType, QueryType, ResultType, LocalTreeWalkType, LocalTopTreeWalkType, ParamType, OutputType, ForceTreeType>
{
    public:
    using Base = TreeWalk<DerivedType, QueryType, ResultType, LocalTreeWalkType, LocalTopTreeWalkType, ParamType, OutputType, ForceTreeType>;
    using Base::TreeWalk;
    using Base::tree;
    using Base::priv;
    using Base::output;

    /* The number of exports that fit in memory. The export, import and result buffers are on the device
     * (plain device memory with CUDA-aware MPI, otherwise managed memory written by kernels, which migrates there),
     * so they must fit in device memory as well as in the host memory used by the base class calculation.
     * Free device memory is read from CUDA, which (unlike mymalloc) sees every allocation on the device,
     * including those of other ranks sharing it.*/
    size_t compute_bunchsize()
    {
        const size_t hostbunch = Base::compute_bunchsize();
        /* Wait for queued prefetches (of Queries and Results), so that their pages are counted as used.*/
        cudaError_t status = cudaDeviceSynchronize();
        size_t freedev = 0, totaldev = 0;
        if(status == cudaSuccess)
            status = cudaMemGetInfo(&freedev, &totaldev);
        if(status != cudaSuccess) {
            message(1, "Could not read free device memory (%s): sizing the export buffer from host memory\n", cudaGetErrorString(status));
            cudaGetLastError();
            return hostbunch;
        }
        const size_t q = sizeof(QueryType), r = sizeof(ResultType);
        /* Per export, alive for the whole round: the export table (managed, written on the device),
         * the export queries and the send buffer.*/
        const size_t perexport_base = sizeof(data_index) + 2 * q;
        /* Peak during ev_transpose_exports: the sort by task, ie tasks, order and the radix sort double buffers of both.
         * These are freed when it returns, except tasks, which is kept as export_bufpos.*/
        const size_t perexport_sort = 4 * sizeof(int);
        /* Peak after the transpose: export_bufpos, the returned results, and the import queries and results,
         * with the same allowance for imbalance as the host calculation.*/
        const size_t perexport_comm = sizeof(int) + r + ceil(Base::ImportBufferBoost * (q + r));
        const size_t perexport = perexport_base + std::max(perexport_sort, perexport_comm);
        /* Leave slack for CUB scratch space, CUDA-aware MPI staging buffers and fragmentation.*/
        const size_t reserve = std::max<size_t>(totaldev / 20, 512L * 1024 * 1024);
        const size_t usable = freedev > reserve ? (freedev - reserve)  : 0;
        const size_t devbunch = usable / perexport;
        if(devbunch < 20)
            endrun(1231246, "Not enough free device memory to export particles: free %lu of %lu bytes, reserve %lu, %lu bytes per export\n",
                   freedev, totaldev, reserve, perexport);
        return std::min(hostbunch, devbunch);
    }

    /* Inverse of the export transpose, set by ev_transpose_exports and used and freed by ev_reduce_export_result:
     * export_bufpos[k] is the position of export k (in ExportTable order) in the send buffer,
     * and so of its result in the buffer of returned results.*/
    thrust::device_vector<int> export_bufpos;

    /* Transpose the export queries, from the order of the export table (ordered by particle)
     * into export_queries, ordered by destination task and fill Export_count.
     * Count the exports to each task on the device, and store them in exportlist->Export_count,
     * so that the export table (written by the toptree kernel) is never read on the host.
     * We sort the export table by task, which is also the first step of the transpose:
     * the CPU transpose scatters, walking the export table in order and placing export k at
     *     bufpos(k) = Exportoffset[task] + (number of earlier exports to the same task).
     * Here we compute the inverse map, order[bufpos] = k, with a stable sort.
     * - Sorting by task in increasing order puts the block for task t after all the exports to lower
     *   tasks, so it starts at sum_{s<t} Export_count[s].
     * - The sort is stable and order starts as the identity, so within each task the exports stay
     *   in increasing k, as with the per-task counter of the CPU version.
     * Because the tasks are then sorted, the exports to task t occupy [ends[t-1], ends[t]),
     * where ends[t] is the index of the first entry > t, which a vectorised binary search finds.
     * Only these NTask counts are copied to the host.
     * It stores the inverse permutation in export_bufpos, for use by ev_reduce_export_result.*/
    void ev_transpose_exports(ExportMemory2<QueryType> * const exportlist, QueryType * const export_queries)
    {
        /* export_order[j] is the export table index of the j-th query to send, and export_tasks[j] its task.*/
        /* Free any stale export buffer from an earlier round.*/
        thrust::device_vector<int>().swap(export_bufpos);
        const int64_t Nexport = exportlist->Nexport;
        if(Nexport == 0)
            return;

        /* tasks[k] is the destination task of export k: the sort key.*/
        thrust::device_vector<int> export_tasks(Nexport);
        thrust::transform(thrust::device, exportlist->ExportTable, exportlist->ExportTable + Nexport, export_tasks.begin(), ExportTaskOf());
        /* order[k] is the export table index: the sort value. int is enough because data_index::Index is also int.*/
        /* Start from the identity permutation, which is the order of the export table.*/
        thrust::device_vector<int> export_order(Nexport);
        thrust::sequence(thrust::device, export_order.begin(), export_order.end());
        /* After this, tasks is in increasing order and order[j] is the index in the export table
        * of the j-th query to send, ie, the CPU version would place export order[j] at bufpos j.*/
        thrust::stable_sort_by_key(thrust::device, export_tasks.begin(), export_tasks.end(), export_order.begin());

        const int NTask = exportlist->Export_count.size();
        std::vector<int64_t> ends(NTask);
        {
            thrust::device_vector<int64_t> dends(NTask);
            thrust::upper_bound(thrust::device, export_tasks.begin(), export_tasks.end(),
                                thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(NTask), dends.begin());
            thrust::copy(dends.begin(), dends.end(), ends.begin());
        }
        /* Tasks outside [0, NTask) would be silently miscounted: task numbers >= NTask sort after all valid tasks
         * and are missed by ends, negative ones are counted with task 0. These are cheap to check, so always do it.*/
        const int firsttask = export_tasks.front();
        if(firsttask < 0)
            endrun(6, "Device export table has negative task %d\n", firsttask);
        if(ends[NTask-1] != Nexport)
            endrun(6, "Device export table has %ld entries with task >= NTask %d\n", Nexport - ends[NTask-1], NTask);
        for(auto i = 0; i < exportlist->Export_count.size(); i++)
            exportlist->Export_count[i] = ends[i] - (i > 0 ? ends[i-1] : 0);
        /* export_queries[j] = ExportQueries[order[j]]: the queries are now grouped by task.*/
        thrust::gather(thrust::device, export_order.begin(), export_order.end(), exportlist->ExportQueries, export_queries);
        /* The sorted tasks are no longer needed, so reuse them for the inverse permutation:
         * export_bufpos[k] is the position of export k in the send buffer, and so of its result in the
         * buffer of returned results. This is the bufpos the CPU version computes with per-task counters.*/
        thrust::scatter(thrust::device, thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(Nexport),
                        export_order.begin(), export_tasks.begin());
        export_bufpos.swap(export_tasks);
    }

    /* Device version of TreeWalk::ev_reduce_export_result: sum the returned export results into results.
     * exportbuf->databuf may be plain device memory (CUDA-aware MPI) or managed memory;
     * results and ExportTable are managed. Uses (and then frees) the export_bufpos permutation
     * stored by ev_transpose_exports for the same export list.*/
    void ev_reduce_export_result(ResultType * results, CommBuffer * exportbuf, const ExportMemory2<QueryType> * const exportlist)
    {
        const size_t Nexport = exportlist->Nexport;
        if(Nexport == 0)
            return;
        if(export_bufpos.size() != Nexport)
            endrun(8, "Export permutation has %lu entries, expected %lu\n", export_bufpos.size(), Nexport);
        const ResultType * recvbuf = reinterpret_cast<const ResultType *>(exportbuf->databuf);
        const int * bufpos = thrust::raw_pointer_cast(export_bufpos.data());
        const thrust::counting_iterator<size_t> first(0), last(Nexport);
#ifdef DEBUG
        const int64_t nunsorted = thrust::transform_reduce(thrust::device, first, last,
            ExportIndexUnsorted{exportlist->ExportTable}, (int64_t) 0, thrust::plus<int64_t>());
        if(nunsorted)
            endrun(8, "Export table is not sorted by work set index at %ld places\n", nunsorted);
        const int64_t nbad = thrust::transform_reduce(thrust::device, first, last,
            ExportIDMismatch<ResultType>{results, recvbuf, exportlist->ExportTable, bufpos}, (int64_t) 0, thrust::plus<int64_t>());
        if(nbad)
            endrun(8, "Error in communication: %ld of %lu export results have mismatched IDs\n", nbad, Nexport);
#endif
        thrust::for_each(thrust::device, first, last,
            ReduceExportRun<ResultType>{results, recvbuf, exportlist->ExportTable, bufpos, Nexport});
        /* thrust::device is synchronous, but check for an asynchronous kernel error.*/
        cudaError_t status = cudaDeviceSynchronize();
        if (status != cudaSuccess)
            endrun(5, "ev_reduce_export_result kernel failed: %s\n", cudaGetErrorString(status));
        /* Free the permutation: clear() would keep the capacity.*/
        thrust::device_vector<int>().swap(export_bufpos);
    }

    /* Build the queue by calling the haswork function on each particle in the active_set.
     * Arguments:
     * active_set: these items have haswork called on them.
     * size: size of the active set.*/
    int64_t build_queue(int ** WorkSet, int * active_set, const size_t size, const particle_data * const parts)
    {
        /* Explicitly deal with the case where the queue is zero and there is nothing to do.
         * Some OpenMP compilers (nvcc) seem to still execute the below loop in that case*/
        if(size == 0) {
            *WorkSet = mymanagedmalloc("ActiveQueue", int, 1);
            return size;
        }

        cudaError_t err = cudaMalloc(WorkSet, size * sizeof(int));
        if(err != cudaSuccess)
            endrun(5, "Failed to allocate device memory for active set: %s\n", cudaGetErrorString(err));

        HasWorkPredicate<QueryType> haswork{parts};
        /* This is a standard stream compaction algorithm. It evaluates the haswork function
         * for every particle, stores the results in an array of flags, counts the non-zero flags,
         * and then scatters each particle integer to the right index in the final array. All is parallelized. */
        if(active_set) {
            auto end = thrust::copy_if(thrust::device,
                active_set, active_set + size, *WorkSet, haswork);
            return end - *WorkSet;
        }
        else { // Need to handle this separately
            auto end = thrust::copy_if(thrust::device,
                thrust::make_counting_iterator<int>(0),   // input: indices 0..size-1
                thrust::make_counting_iterator<int>(size),
                *WorkSet, haswork);
            return end - *WorkSet;
        }
    }

    int * ev_count_exports(QueryType * queries, const int64_t WorkSetSize)
    {
        /* Allocate at least 1 element so cudaFree is always valid. */
        int * exportcounts = (int *) treewalk_device_malloc(sizeof(int) * std::max(WorkSetSize, (int64_t)1), "ExportCounts");
        if(WorkSetSize == 0)
            return exportcounts;

        const int threadsPerBlock = 256;
        const int blocks = (WorkSetSize + threadsPerBlock - 1) / threadsPerBlock;

        /* First count the exports from each particle and store the counts in exportcounts.*/
        count_toptree_exports<QueryType, LocalTopTreeWalkType, ParamType>
        <<<blocks, threadsPerBlock>>>(queries, tree->get_nodes(), tree->get_topleaves(), tree->get_topleaf_nodes(), tree->get_ntopleaves(), WorkSetSize, &priv, exportcounts);

        /*  inclusive_scan is a partial sum:
         * it counts the total number of particle exports before and including the current point, so we know which
         * elements of the export table to slot this particle's exports into.
         * Particle j has exports from exportcounts[j-1] to exportcounts[j]
         * Final element is Nexports */
        /* Only count exports that we have not yet sent. */
        thrust::inclusive_scan(thrust::device, exportcounts, exportcounts+WorkSetSize, exportcounts);
        return exportcounts;
    }

    void ev_free_exports(int * exportcounts)
    {
        cudaFree(exportcounts);
    }

    /* Ask the driver to move the managed allocation [ptr, ptr+bytes) to this rank's device in bulk.
     * - For pages already on the host (the queries, built there) this migrates them in one go,
     *   instead of page by page as the first kernel faults on them.
     * - For pages not yet touched (Results) this creates them directly in device memory,
     *   so the first kernel write does not fault.
     * If prefer_device is set, the device is also made the preferred location, so the pages stay there
     * (a later host read migrates them back on x86, or reads them over NVLink-C2C on Grace Hopper).
     * The prefetch is asynchronous: it is queued on the default stream, so the kernels
     * launched afterwards on that stream run once it has finished.
     * This is only a hint, so a failure is reported but is not fatal.*/
    static void ev_prefetch_to_device(void * ptr, const size_t bytes, const char * const name, const bool prefer_device)
    {
        if(bytes == 0)
            return;
        int device = 0;
        cudaError_t status = cudaGetDevice(&device);
        if(status == cudaSuccess) {
#if CUDART_VERSION >= 13000
            /* CUDA 13 removed the legacy signatures which took a device id.*/
            struct cudaMemLocation location;
            location.type = cudaMemLocationTypeDevice;
            location.id = device;
            if(prefer_device)
                status = cudaMemAdvise(ptr, bytes, cudaMemAdviseSetPreferredLocation, location);
            if(status == cudaSuccess)
                status = cudaMemPrefetchAsync(ptr, bytes, location, 0, 0);
#else
            if(prefer_device)
                status = cudaMemAdvise(ptr, bytes, cudaMemAdviseSetPreferredLocation, device);
            if(status == cudaSuccess)
                status = cudaMemPrefetchAsync(ptr, bytes, device, 0);
#endif
        }
        if(status != cudaSuccess) {
            message(1, "Could not prefetch %lu bytes of %s to the device: %s\n", bytes, name, cudaGetErrorString(status));
            /* Clear the error, so it is not reported by a later, unrelated, error check.*/
            cudaGetLastError();
        }
    }

    /* The queries are built on the host in managed memory: migrate them to the device in bulk.*/
    void ev_prefetch_queries(QueryType * queries, const int64_t WorkSetSize)
    {
        ev_prefetch_to_device(queries, WorkSetSize * sizeof(QueryType), "Queries", false);
    }

    int64_t ev_toptree(const int64_t WorkSetStart, const int64_t WorkSetSize, int * exportcounts, ExportMemory2<QueryType> * const exportlist, QueryType * queries)
    {
        const int threadsPerBlock = 256;
        /* Adjust the indices for the restart */
        int64_t curSize = WorkSetSize - WorkSetStart;
        int exportoffset = 0;
        if(WorkSetStart > 0) {
            cudaMemcpy(&exportoffset, &exportcounts[WorkSetStart-1], sizeof(int), cudaMemcpyDeviceToHost);
        }
        /* Handle the no work case explicitly */
        if(curSize == 0) {
            exportlist->Nexport = 0;
            exportlist->ExportTable = mymalloc("DataIndexTable", data_index, 1);
            return WorkSetStart;
        }
        exportcounts = exportcounts + WorkSetStart;
        /* Here we check whether our export buffer will fill up.
         * If the export buffer filled up, find the first place where it did.
         * First element where the exportcount is larger than the bunch size */
        int BunchSize = compute_bunchsize() + exportoffset;
        Greater_than_BunchSize gtrbunch{BunchSize};

        bool BufferFull = false;
        auto iter = thrust::find_if(thrust::device, exportcounts, exportcounts+curSize,gtrbunch);
        if(iter != exportcounts + curSize) {
            curSize = iter - exportcounts;
            if(curSize == 0)
                endrun(5, "Not enough export space to make progress! lastsuc %ld\n", WorkSetStart);
            BufferFull = true;
        }
        /* Note this is the sum including the current element. */
        cudaMemcpy(&exportlist->Nexport, &exportcounts[curSize-1], sizeof(int), cudaMemcpyDeviceToHost);
        /* Note that the exportcount is built on the first iteration
         * and so the indices are off for the second one. */
        exportlist->Nexport -= exportoffset;
        exportlist->ExportTable = mymanagedmalloc("DataIndexTable", data_index, exportlist->Nexport);
        exportlist->ExportQueries = mymanagedmalloc("ExportQueries", QueryType, exportlist->Nexport);

        if(BufferFull)
            message(1, "Tree export buffer full with %lu exports (%lu Mbytes). BunchSize %d. First particle %ld new start: %ld size %ld.\n",
                exportlist->Nexport, exportlist->Nexport*sizeof(QueryType)/1024/1024, BunchSize, WorkSetStart, WorkSetStart + curSize, WorkSetSize);

        const int blocks = (curSize + threadsPerBlock - 1) / threadsPerBlock;
        /* Now we run toptree_visit again with the export offsets to make the export table.
         * Likely most particles have zero exports, so this will be somewhat faster than the first run. */
        do_toptree_exports<QueryType, LocalTopTreeWalkType, ParamType>
         <<<blocks, threadsPerBlock>>>(queries, tree->get_nodes(), tree->get_topleaves(), tree->get_topleaf_nodes(), tree->get_ntopleaves(),
             WorkSetStart, curSize, &priv, exportcounts, exportoffset, exportlist->ExportTable, exportlist->ExportQueries);

        cudaError_t status = cudaDeviceSynchronize();
        if (status != cudaSuccess)
            endrun(5, "ev_toptree kernel failed: %s\n", cudaGetErrorString(status));
        // else
            // message(1, "Finished toptree. First particle %ld next start: %ld size %ld.\n", BufferFullFlag, WorkSetStart, currentIndex, WorkSetSize);
        /* Start again with the next chunk not yet evaluated*/
        return WorkSetStart + curSize;
    }

    /* Perform evaluation of a chunk of queries and store the output in results.
     *
     * Arguments:
     * - QueryType queries: an array of querys sent from another rank for evaluation on the local tree.
     * - ResultType results: an array of results generated by walking the local tree, for returning to the original rank.
     * - WorkSetSize: number of queries to evaluate.
     * - parts: particle table to put at the roots of the tree.
     * */
    template <TreeWalkReduceMode mode>
    void ev_visit(QueryType * queries, ResultType * results, const int64_t WorkSetSize, particle_data * const parts) {
        if(WorkSetSize == 0)
            return;
        const int threadsPerBlock = 256;
        const int blocks = (WorkSetSize + threadsPerBlock - 1) / threadsPerBlock;
        /* All arrays need to be managed malloc or device:
         * particles, tree nodes,
         * priv should be heap-allocated as placement-new pointers in managed memory */
        treewalk_visit_kernel<QueryType, ResultType, LocalTreeWalkType, ParamType, mode>
        <<<blocks, threadsPerBlock>>>(queries, results, WorkSetSize, parts, tree->get_nodes(), &priv);
        cudaError_t status = cudaDeviceSynchronize();
        if (status != cudaSuccess)
            endrun(5, "ev_visit kernel failed: %s\n", cudaGetErrorString(status));
    };

    /* Do the postprocessing on the GPU. This simply evaluates the postprocess function for every particle. */
    void ev_postprocess(int * WorkSet, int64_t WorkSetSize, ResultType * results, particle_data * const particles)
    {
        if(WorkSetSize == 0)
            return;
        const int threadsPerBlock = 256;
        const int blocks = (WorkSetSize + threadsPerBlock - 1) / threadsPerBlock;
        treewalk_postprocess_kernel<ResultType, ParamType, OutputType>
        <<<blocks, threadsPerBlock>>>(particles, WorkSet, WorkSetSize, results, &priv, output);
        cudaError_t status = cudaDeviceSynchronize();
        if (status != cudaSuccess)
            endrun(5, "ev_postprocess kernel failed: %s\n", cudaGetErrorString(status));
    }
};

#endif
