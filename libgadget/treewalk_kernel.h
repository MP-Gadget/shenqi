#ifndef TREEWALK_KERNEL_H
#define TREEWALK_KERNEL_H

#include <cuda_runtime.h>
#include "treewalk.h"
#include "partmanager.h"  // To access particle_data structure

// Declaration of the GPU kernel
// __global__ void treewalk_kernel(TreeWalk *tw, struct particle_data *particles, int *workset, size_t workset_size);

void run_treewalk_kernel(TreeWalk *tw, struct particle_data *particles, int *workset, size_t workset_size, struct gravshort_tree_params * TreeParams_ptr, double GravitySoftening, unsigned long long int *maxNinteractions, unsigned long long int *minNinteractions, unsigned long long int *Ninteractions);



#endif  // TREEWALK_KERNEL_H
