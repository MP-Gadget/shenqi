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

#include <cuda_runtime.h>           // For CUDA runtime API functions.
#include <device_launch_parameters.h>  // To support device-related parameters.
#include <thrust/copy.h>                        // thrust::copy_if
#include <thrust/execution_policy.h>            // thrust::device
#include <thrust/scan.h>            // thrust::exclusive_scan
#include <thrust/find.h>            // thrust::find_if

/* Each thread counts the number of exports needed for each particle.
 * counts are stored in the exportcounts argument.
 * We then use these counts to allocate memory and fill the export table. */
template <typename QueryType, typename LocalTopTreeWalkType, typename ParamType>
__global__ void count_toptree_exports(
    // The memory here should be allocated cudaManagedMalloc
    QueryType * const queries,
    const NODE * const Nodes,
    const topleaf_data * const TopLeaves,
    const int NTopLeaves,
    const int lastnode,
    const int64_t WorkSetSize,
    // by reference so the destructor does not run,
    // which means in managed memory, and with all sub-arrays in managed memory
    const ParamType * priv,
    int * exportcounts)   // output: flags[i] = is the number of exports needed for this particle.
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= WorkSetSize)
        return;

    LocalTopTreeWalkType lv(Nodes, TopLeaves, NTopLeaves, lastnode);
    const int rt = lv.template toptree_visit<TOPTREE_COUNT>(queries[tid], tid, *priv, NULL, NULL, 0);
    exportcounts[tid] = rt;
}

template <typename QueryType, typename LocalTopTreeWalkType, typename ParamType>
__global__
void do_toptree_exports(
    // The memory here should be allocated cudaManagedMalloc
    QueryType * const queries,
    const NODE * const Nodes,
    const topleaf_data * const TopLeaves,
    const int NTopLeaves,
    const int lastnode,
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

    LocalTopTreeWalkType lv(Nodes, TopLeaves, NTopLeaves, lastnode);

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
    const NODE * const Nodes,
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

template <typename DerivedType, typename QueryType, typename ResultType, typename LocalTreeWalkType, typename LocalTopTreeWalkType, typename ParamType, typename OutputType>
class TreeWalkGPU: public TreeWalk<DerivedType, QueryType, ResultType, LocalTreeWalkType, LocalTopTreeWalkType, ParamType, OutputType>
{
    public:
    using Base = TreeWalk<DerivedType, QueryType, ResultType, LocalTreeWalkType, LocalTopTreeWalkType, ParamType, OutputType>;
    using Base::TreeWalk;
    using Base::tree;
    using Base::priv;
    using Base::output;
    using Base::compute_bunchsize;

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
        int * exportcounts;
        /* Allocate at least 1 element so cudaFree is always valid. */
        cudaError_t err = cudaMalloc(&exportcounts, sizeof(int) * std::max(WorkSetSize, (int64_t)1));
        if(err != cudaSuccess)
            endrun(5, "Failed to allocate device memory for export counts: %s\n", cudaGetErrorString(err));
        if(WorkSetSize == 0)
            return exportcounts;

        const int threadsPerBlock = 256;
        const int blocks = (WorkSetSize + threadsPerBlock - 1) / threadsPerBlock;

        /* First count the exports from each particle and store the counts in exportcounts.*/
        count_toptree_exports<QueryType, LocalTopTreeWalkType, ParamType>
        <<<blocks, threadsPerBlock>>>(queries, tree->Nodes, tree->TopLeaves, tree->NTopLeaves, tree->lastnode, WorkSetSize, &priv, exportcounts);

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
         <<<blocks, threadsPerBlock>>>(queries, tree->Nodes, tree->TopLeaves, tree->NTopLeaves, tree->lastnode,
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
        <<<blocks, threadsPerBlock>>>(queries, results, WorkSetSize, parts, tree->Nodes, &priv);
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
