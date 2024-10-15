
#include <cuda_runtime.h>           // For CUDA runtime API functions.
#include <device_launch_parameters.h>  // To support device-related parameters.
// #include "treewalk.h"               // Include necessary header for TreeWalk structures and methods
#include "treewalk_kernel.h"
#include "gravshort.h"
// treewalk_kernel.cu

__device__ static MyFloat
grav_get_abs_accel_device(struct particle_data * PP, const double G)
{
    double aold=0;
    int j;
    for(j = 0; j < 3; j++) {
       double ax = PP->FullTreeGravAccel[j] + PP->GravPM[j];
       aold += ax*ax;
    }
    return sqrt(aold) / G;
}

__device__ static void
grav_short_copy_device(int place, TreeWalkQueryGravShort * input, TreeWalk * tw, struct particle_data *particles)
{
    input->OldAcc = grav_get_abs_accel_device(&particles[place], GRAV_GET_PRIV(tw)->G);
}

__device__ static void
treewalk_init_query_device(TreeWalk *tw, TreeWalkQueryBase *query, int i, const int *NodeList, struct particle_data *particles) {
    // Access particle data through particles argument
    for(int d = 0; d < 3; d++) {
        query->Pos[d] = particles[i].Pos[d];  // Use particles instead of P macro
    }

    if (NodeList) {
        memcpy(query->NodeList, NodeList, sizeof(query->NodeList[0]) * NODELISTLENGTH);
    } else {
        query->NodeList[0] = tw->tree->firstnode;  // root node
        query->NodeList[1] = -1;  // terminate immediately
    }
    TreeWalkQueryGravShort * query_short;
    // point query_short to the query
    query_short = (TreeWalkQueryGravShort *) query;
    // tw->fill(i, query, tw);
    grav_short_copy_device(i, query_short, tw, particles);
}

__device__ static void
treewalk_init_result_device(TreeWalk *tw, TreeWalkResultBase *result, TreeWalkQueryBase *query) {
    memset(result, 0, tw->result_type_elsize);  // Initialize the result structure
}

__device__ void
treewalk_reduce_result_device(TreeWalk *tw, TreeWalkResultBase *result, int i, enum TreeWalkReduceMode mode) {
    if (tw->reduce != NULL) {
        tw->reduce(i, result, mode, tw);  // Call the reduce function
    }
}

__global__ void treewalk_kernel(TreeWalk *tw, struct particle_data *particles, int *workset, size_t workset_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < workset_size) {
        int i = workset[tid];

        TreeWalkQueryBase input;
        TreeWalkResultBase output;
        // Initialize query and result using device functions
        treewalk_init_query_device(tw, &input, i, NULL, particles);
        treewalk_init_result_device(tw, &output, &input);

        // Perform treewalk for particle
        LocalTreeWalk lv;
        lv.target = i;
        tw->visit(&input, &output, &lv);

        // Reduce results for this particle
        treewalk_reduce_result_device(tw, &output, i, TREEWALK_PRIMARY);
    }
}

// Function to launch kernel
void run_treewalk_kernel(TreeWalk *tw, struct particle_data *particles, int *workset, size_t workset_size) {
    int threadsPerBlock = 256;
    int blocks = (workset_size + threadsPerBlock - 1) / threadsPerBlock;
    treewalk_kernel<<<blocks, threadsPerBlock>>>(tw, particles, workset, workset_size);
    cudaDeviceSynchronize();
}
