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
#include "libgadget/utils/mymalloc.h"
#include "treewalk2.h"

#include <cuda_runtime.h>           // For CUDA runtime API functions.
#include <device_launch_parameters.h>  // To support device-related parameters.
#include <thrust/copy.h>                        // thrust::copy_if
#include <thrust/iterator/counting_iterator.h>  // thrust::make_counting_iterator
#include <thrust/execution_policy.h>            // thrust::device
#include <thrust/scan.h>            // thrust::exclusive_scan

/* Each thread counts the number of exports needed for each particle.
 * counts are stored in the exportcounts argument.
 * We then use these counts to allocate memory and fill the export table. */
template <typename QueryType, typename LocalTopTreeWalkType, typename ParamType>
__global__ void count_toptree_exports(
    // The memory here should be allocated cudaManagedMalloc
    particle_data * const parts,
    const NODE * const Nodes,
    const int firstnode,
    const int * const WorkSet,
    const int64_t WorkSetSize,
    // by reference so the destructor does not run,
    // which means in managed memory, and with all sub-arrays in managed memory
    const ParamType * priv,
    int * exportcounts)   // output: flags[i] = is the number of exports needed for this particle.
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= WorkSetSize)
        return;

    LocalTopTreeWalkType lv(tree);
    const int i = WorkSet ? WorkSet[tid] : tid;
    /* Toptree never uses node list */
    QueryType input(parts[i], NULL, firstnode, priv);
    const int rt = lv.toptree_visit(i, input, priv, parts, NULL, 0);
    exportcounts[tid] = rt;
}

template <typename QueryType, typename LocalTopTreeWalkType, typename ParamType>
__global__
void do_toptree_exports(
    // The memory here should be allocated cudaManagedMalloc
    particle_data * const parts,
    const NODE * const Nodes,
    const int firstnode,
    const int * const WorkSet,
    const int64_t WorkSetSize,
    // by reference so the destructor does not run,
    // which means in managed memory, and with all sub-arrays in managed memory
    const ParamType * priv,
    const int * exportoffsets,
    data_index * ExportTable // This is the main export table output.
    )
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= WorkSetSize)
        return;

    LocalTopTreeWalkType lv(tree);

    const int i = WorkSet ? WorkSet[tid] : tid;
    /* Toptree never uses node list */
    QueryType input(parts[i], NULL, firstnode, priv);
    int offset = (tid > 0) * exportoffsets[tid-1];
    /* This will save exports to the memory in ExportTable[exportoffsets[tid]].
     * We ignore return as it is the same as exportcounts. BunchSize is large as we arranged never to overflow.*/
    lv.toptree_visit(i, input, priv, ExportTable[offset], 1<<31);
};

template <typename QueryType, typename ResultType, typename LocalTreeWalkType, typename ParamType, typename OutputType>
__global__
void treewalk_primary_kernel(
    // The memory here should be allocated cudaManagedMalloc
    particle_data * const parts,
    const NODE * const Nodes,
    const int firstnode,
    const int * const WorkSet,
    const int64_t WorkSetSize,
    // by reference so the destructor does not run,
    // which means in managed memory, and with all sub-arrays in managed memory
    const ParamType * priv,
    const OutputType * output,
    // device memory pointers
    unsigned int * maxNinteractions,
    unsigned int * minNinteractions)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= WorkSetSize)
        return;

    const int64_t i = WorkSet ? (int64_t) WorkSet[tid] : tid;
    QueryType input(parts[i], NULL, firstnode, *priv);
    ResultType result(input);
    LocalTreeWalkType lv(Nodes, input);
    unsigned int ninteractions = lv.template visit<TREEWALK_PRIMARY>(input, &result, *priv, parts);
    result.template reduce<TREEWALK_PRIMARY>(i, output, parts);

    atomicMax(maxNinteractions, (unsigned int) ninteractions);
    atomicMin(minNinteractions, (unsigned int) ninteractions);
};

template <typename QueryType, typename ResultType, typename LocalTreeWalkType, typename ParamType, typename OutputType>
__global__
void treewalk_secondary_kernel(
    // The memory here should be allocated cudaManagedMalloc
    particle_data * const parts,
    const NODE * const Nodes,
    ResultType * results,
    QueryType * imports,
    const int64_t WorkSetSize,
    const ParamType * priv)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= WorkSetSize)
        return;

    QueryType * input = &(imports[tid]);
    ResultType * resoutput = new (&results[tid]) ResultType(*input);
    LocalTreeWalkType lv(Nodes, *input);
    lv.template visit<TREEWALK_GHOSTS>(*input, resoutput, *priv, parts);
};

template <typename ParamType, typename OutputType>
__global__
void treewalk_postprocess_kernel(
    // The memory here should be allocated cudaManagedMalloc
    particle_data * const parts,
    const int * WorkSet,
    const int64_t WorkSetSize,
    const ParamType * priv,
    OutputType * output)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= WorkSetSize)
        return;

    const int p_i = WorkSet ? WorkSet[tid] : tid;
    output->postprocess(p_i, parts, priv);
}

struct HasWorkPredicate {
     const particle_data * parts;
     const int * active_set;
     MYCUDAFN bool operator()(int i) const {
         int p_i = active_set ? active_set[i] : i;
         return QueryType::haswork(p_i, parts);
     }
 };

template <typename DerivedType, typename QueryType, typename ResultType, typename LocalTreeWalkType, typename LocalTopTreeWalkType, typename ParamType, typename OutputType>
class TreeWalkGPU: public TreeWalk<DerivedType, QueryType, ResultType, LocalTreeWalkType, LocalTopTreeWalkType, ParamType, OutputType>
{
    public:
    using Base = TreeWalk<DerivedType, QueryType, ResultType, LocalTreeWalkType, LocalTopTreeWalkType, ParamType, OutputType>;
    using Base::TreeWalk;
    using Base::maxNinteractions;
    using Base::minNinteractions;
    using Base::tree;
    using Base::priv;
    using Base::output;

    /* Build the queue by calling the haswork function on each particle in the active_set.
     * Arguments:
     * active_set: these items have haswork called on them.
     * size: size of the active set.*/
    int64_t build_queue(int ** WorkSet, int * active_set, const size_t size, const particle_data * const parts)
    {
        /* Explicitly deal with the case where the queue is zero and there is nothing to do.
         * Some OpenMP compilers (nvcc) seem to still execute the below loop in that case*/
        if(size == 0) {
            *WorkSet = (int *) mymanagedmalloc("ActiveQueue", sizeof(int));
            return size;
        }

        cudaError_t err = cudaMalloc(WorkSet, size * sizeof(int));
        if(err != cudaSuccess)
            endrun(5, "Failed to allocate device memory for active set: %s\n", cudaGetErrorString(err));

        HasWorkPredicate functor{parts, active_set};
        /* This is a standard stream compaction algorithm. It evaluates the haswork function
         * for every particle, stores the results in an array of flags, counts the non-zero flags,
         * and then scatters each particle integer to the right index in the final array. All is parallelized. */
        auto end = thrust::copy_if(
            thrust::device,
            thrust::make_counting_iterator(0),   // input: indices 0..size-1
            thrust::make_counting_iterator((int)size),
            *WorkSet, functor);
        return WorkSetSize = end - WorkSet;
    }

    int64_t ev_toptree(int * WorkSet, const int64_t WorkSetStart, const int64_t WorkSetSize, const particle_data * const parts, ExportMemory * const exportlist)
    {
        int64_t curSize = WorkSetSize - WorkSetStart;
        const int threadsPerBlock = 256;
        const int blocks = (curSize + threadsPerBlock - 1) / threadsPerBlock;
        int64_t currentIndex = WorkSet + WorkSetStart;
        int BufferFullFlag = 0;

        int * exportcounts,
        auto err = cudaMalloc(&exportcounts, sizeof(int) * curSize);
        if(err != cudaSuccess)
            endrun(5, "Failed to allocate device memory for export counts: %s\n", cudaGetErrorString(err));

        /* First count the exports from each particle and store the counts in exportcounts. */
        count_toptree_exports<QueryType, LocalTopTreeWalkType, ParamType>
        <<<blocks, threadsPerBlock>>>(particles, tree->Nodes, tree->firstnode, currentIndex, curSize, &priv, exportcounts);

        /*  inclusive_scan is a partial sum:
         * it counts the total number of particle exports before and including the current point, so we know which
         * elements of the export table to slot this particle's exports into.
         * Particle j has exports from exportcounts[j-1] to exportcounts[j]
         * Final element is Nexports */
        thrust::inclusive_scan(thrust::device, exportcounts, exportcounts+curSize, exportcounts);
        /* Here we check whether our export buffer will fill up. TODO: Avoid recomputing the export counts on subsequent iterations.
         * TODO this also needs logic to get the final count from the device.
         */
        /* If the export buffer filled up, find the first place where it did. */
        /* First element where the exportcount is larger than the bunch size */
        auto iter = thrust::find_if(thrust::device, exportcounts, exportcounts+curSize,
                [exportlist->BunchSize] __device__ (int i) { return i >= exportlist->BunchSize;});
        if(iter != exportcounts+curSize) {
            BufferFullFlag = 1;
            curSize = iter - exportcounts;
            if(curSize == 0)
                endrun(5, "Not enough export space to make progress! lastsuc %ld\n", currentIndex);
            message(1, "Tree export buffer full with %lu exports (%lu Mbytes). First particle %ld new start: %ld size %ld.\n",
                exportcounts[curSize], exportcounts[curSize]*sizeof(QueryType)/1024/1024, WorkSetStart, WorkSetStart + curSize, WorkSetSize);
        }
        /* Note this is the sum including the current element. */
        exportlist->Nexport_thread[0] = exportcounts[curSize];
        exportlist->ExportTable_thread[0] = (DataIndexTable *) mymanagedmalloc("DataIndexTable", exportlist->Nexport_thread[0] * sizeof(DataIndexTable));
        /* Now we run toptree_visit again with the export offsets to make the export table.
         * Likely most particles have zero exports, so this will be somewhat faster than the first run. */
        do_toptree_exports<QueryType, LocalTopTreeWalkType, ParamType>
         <<<blocks, threadsPerBlock>>>(particles, tree->Nodes, tree->firstnode, currentIndex, curSize, &priv, exportoffsets, exportlist->ExportTable_thread);

        cudaError_t status = cudaDeviceSynchronize();
        if (status != cudaSuccess)
            endrun(5, "ev_toptree kernel failed: %s\n", cudaGetErrorString(status));
        // else
            // message(1, "Finished toptree. First particle %ld next start: %ld size %ld.\n", BufferFullFlag, WorkSetStart, currentIndex, WorkSetSize);
        /* Start again with the next chunk not yet evaluated*/
        return currentIndex + curSize;
    }

    // Function to launch kernel (wrapper)
    void ev_primary(int * WorkSet, int64_t WorkSetSize, particle_data * const particles) {
        /* Declare device memory for counters */
        unsigned int * d_maxNinteractions = nullptr;
        unsigned int * d_minNinteractions = nullptr;
        cudaMalloc(&d_maxNinteractions, sizeof(unsigned int));
        cudaMalloc(&d_minNinteractions, sizeof(unsigned int));
        /* Reset before launch. memset to 0x00 gives 0 (for sum/max base),
         * memset to 0xFF gives ULLONG_MAX (correct base for atomicMin). */
        cudaMemcpy(d_maxNinteractions, &maxNinteractions, sizeof(unsigned int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_minNinteractions, &minNinteractions, sizeof(unsigned int), cudaMemcpyHostToDevice);

        const int threadsPerBlock = 256;
        const int blocks = (WorkSetSize + threadsPerBlock - 1) / threadsPerBlock;
        /* All arrays need to be managed malloc or device:
         * particles, tree nodes,
         * WorkSet, counters (device)
         * priv and output should be heap-allocated as placement-new pointers in managed memory */
        treewalk_primary_kernel<QueryType, ResultType, LocalTreeWalkType, ParamType, OutputType>
        <<<blocks, threadsPerBlock>>>(particles, tree->Nodes, tree->firstnode, WorkSet, WorkSetSize, &priv, output, d_maxNinteractions, d_minNinteractions);
        /* Copy results back and accumulate into host-side counters. */
        cudaMemcpy(&maxNinteractions, d_maxNinteractions, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&minNinteractions, d_minNinteractions, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cudaError_t status = cudaDeviceSynchronize();
        if (status != cudaSuccess) {
            // Error handling
            endrun(5, "ev_primary kernel failed: %s\n", cudaGetErrorString(status));
        }
    };

    /* Perform evaluation of a chunk of secondary particles from a single processor.
     *
     * Arguments:
     * - QueryType imports: an array of querys sent from another rank for evaluation on the local tree.
     * - ResultType results: an array of results generated by walking the local tree, for returning to the original rank.
     * - nimports_task: size of the query and result arrays.
     * */
    void ev_secondary(ResultType * results, QueryType * imports, const int64_t WorkSetSize, struct particle_data * const particles)
    {
        const int threadsPerBlock = 256;
        const int blocks = (WorkSetSize + threadsPerBlock - 1) / threadsPerBlock;
        /* All arrays need to be managed malloc or device:
         * priv and output should be heap-allocated as placement-new pointers in managed memory */
        treewalk_secondary_kernel<QueryType, ResultType, LocalTreeWalkType, ParamType, OutputType>
        <<<blocks, threadsPerBlock>>>(particles, tree->Nodes, results, imports, WorkSetSize, &priv);
        cudaError_t status = cudaDeviceSynchronize();
        if (status != cudaSuccess)
            endrun(5, "ev_secondary kernel failed: %s\n", cudaGetErrorString(status));
    }

    /* Do the postprocessing on the GPU. This simply evaluates the postprocess function for every particle. */
    void ev_postprocess(int * WorkSet, int64_t WorkSetSize, particle_data * const particles)
    {
        const int threadsPerBlock = 256;
        const int blocks = (WorkSetSize + threadsPerBlock - 1) / threadsPerBlock;
        treewalk_postprocess_kernel<ParamType, OutputType>
        <<<blocks, threadsPerBlock>>>(particles, WorkSet, WorkSetSize, &priv, output);
        cudaError_t status = cudaDeviceSynchronize();
        if (status != cudaSuccess)
            endrun(5, "ev_postprocess kernel failed: %s\n", cudaGetErrorString(status));
    }
};

#endif
