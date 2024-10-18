// kernel_launch.h
#ifndef PM_KERNEL_H
#define PM_KERNEL_H

#ifdef __cplusplus
extern "C" {
#endif

void launch_potential_transfer(Box3D box_complex, cufftComplex* data, int rank, int size, PetaPM *pm, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif // PM_KERNEL_H