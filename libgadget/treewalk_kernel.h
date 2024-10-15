#ifndef TREEWALK_KERNEL_H
#define TREEWALK_KERNEL_H

#include <cuda_runtime.h>
#include "treewalk.h"
#include "partmanager.h"  // To access particle_data structure

// Declaration of the GPU kernel
// __global__ void treewalk_kernel(TreeWalk *tw, struct particle_data *particles, int *workset, size_t workset_size);

void run_treewalk_kernel(TreeWalk *tw, struct particle_data *particles, int *workset, size_t workset_size);



#endif  // TREEWALK_KERNEL_H
