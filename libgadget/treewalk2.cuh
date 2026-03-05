#ifndef TREEWALK2_CUH
#define TREEWALK2_CUH

#include "treewalk2.h"

#include <cuda_runtime.h>           // For CUDA runtime API functions.
#include <device_launch_parameters.h>  // To support device-related parameters.

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

        // workset is NULL at a PM step
        int threadsPerBlock = 256;
        int blocks = (WorkSetSize + threadsPerBlock - 1) / threadsPerBlock;
        /* All arrays need to be managed malloc or device:
         * particles, tree nodes,
         * WorkSet, counters (device)
         * priv and output should be heap-allocated as placement-new pointers in managed memory */
        treewalk_primary_kernel<QueryType, ResultType, LocalTreeWalkType, ParamType, OutputType>
        <<<blocks, threadsPerBlock>>>(particles, tree->Nodes, tree->firstnode, WorkSet, WorkSetSize, &priv, output, d_maxNinteractions, d_minNinteractions);
        /* Copy results back and accumulate into host-side counters. */
        cudaMemcpy(&maxNinteractions, d_maxNinteractions, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&minNinteractions, d_minNinteractions, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        cudaError_t status = cudaDeviceSynchronize();
        if (status != cudaSuccess) {
            // Error handling
            endrun(5, "ev_primary kernel failed: %s\n", cudaGetErrorString(status));
        }
    };
};

#endif
