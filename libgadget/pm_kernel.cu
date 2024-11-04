// pm_kernel.cu
#include <cuda_runtime.h>
#include <mpi.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "box_iterator.hpp"
#include "petapm.h"


/* unnormalized sinc function sin(x) / x */
__device__ double sinc_unnormed(double x) {
    if(x < 1e-5 && x > -1e-5) {
        double x2 = x * x;
        return 1.0 - x2 / 6. + x2  * x2 / 120.;
    } else {
        return sin(x) / x;
    }
}


/* the transfer functions for force in fourier space applied to potential */
/* super lanzcos in CH6 P 122 Digital Filters by Richard W. Hamming */
__device__ double diff_kernel(double w) {
/* order N = 1 */
/*
 * This is the same as GADGET-2 but in fourier space:
 * see gadget-2 paper and Hamming's book.
 * c1 = 2 / 3, c2 = 1 / 12
 * */
    return 1 / 6.0 * (8 * sin (w) - sin (2 * w));
}


__global__
void potential_transfer_kernel(BoxIterator<cufftDoubleComplex> begin, BoxIterator<cufftDoubleComplex> end, PetaPM *pm) {
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    begin += tid;

    if (begin < end) {
        // Get global 3D coordinates of the current element in real space
        int x = begin.x();
        int y = begin.y();
        int z = begin.z();

        // Compute the corresponding wave numbers (kx, ky, kz), in grid unit
        int kx = x<=pm->Nmesh/2 ? x : (x-pm->Nmesh);
        int ky = y<=pm->Nmesh/2 ? y : (y-pm->Nmesh);
        int kz = z<=pm->Nmesh/2 ? z : (z-pm->Nmesh);
        int64_t k2 = 0.0;
        k2 += ((int64_t)kx) * kx;
        k2 += ((int64_t)ky) * ky;
        k2 += ((int64_t)kz) * kz;
        
        const double asmth2 = pow((2 * M_PI) * pm->Asmth / pm->Nmesh, 2);
        double f = 1.0;
        const double smth = exp(-k2 * asmth2) / k2;
        const double pot_factor = -pm->G / (M_PI * pm->BoxSize);

        int kpos[3] = {kx, ky, kz};
        // Apply CIC deconvolution
        for (int k = 0; k < 3; k++) {
            double tmp = (kpos[k] * M_PI) / pm->Nmesh;
            tmp = sinc_unnormed(tmp);
            f *= 1.0 / (tmp * tmp);
        }
        const double fac = pot_factor * smth * f * f;
        //CUDA TODO: add massive neutrino back

        // Handle zero mode separately
        if (k2 == 0) {
            begin->x = 0.0;
            begin->y = 0.0;
            return;
        }
        if(tid < 10) {
            printf("GPU data (after first transform): global 3D index [%d %d %d], local index %d is (%f,%f)\n", 
                (int)begin.x(), (int)begin.y(), (int)begin.z(), (int)begin.i(), begin->x, begin->y);
        }
        // Apply scaling factor
        begin->x *= fac;
        begin->y *= fac;
    }
}


__global__ 
void force_transfer_kernel(BoxIterator<cufftDoubleComplex> begin, BoxIterator<cufftDoubleComplex> end, PetaPM *pm, int ik) {
    double tmp0;
    double tmp1;
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    begin += tid;
    int pos;
    
    if (begin < end) {
        // Get global 3D coordinates of the current element in real space
        switch (ik) {
            case 0:
                pos = begin.x();
                break;
            case 1:
                pos = begin.y();
                break;
            case 2:
                pos = begin.z();
                break;
        }
        // Compute the corresponding wave numbers (kx, ky, kz), in grid unit
        int kpos = pos<=pm->Nmesh/2 ? pos : (pos-pm->Nmesh);
        /*
         * negative sign is from force_x = - Del_x pot
         *
         * filter is   i K(w)
         * */
        double fac = -1 * diff_kernel (kpos * (2 * M_PI / pm->Nmesh)) * (pm->Nmesh / pm->BoxSize);
        tmp0 = - begin->y * fac;
        tmp1 = begin->x * fac;
        begin->x = tmp0;
        begin->y = tmp1;
    }
}


extern "C" void launch_potential_transfer(Box3D box_complex, cufftDoubleComplex* data, int rank, int size, PetaPM *pm, cudaStream_t stream) {
    auto [begin_d, end_d] = BoxIterators(box_complex, data);
    const size_t num_elements = std::distance(begin_d, end_d);
    const size_t num_threads  = 256;
    const size_t num_blocks   = (num_elements + num_threads - 1) / num_threads;
    potential_transfer_kernel<<<num_blocks, num_threads, 0, stream>>>(begin_d, end_d, pm);
}


extern "C" void launch_force_x_transfer(Box3D box_complex, cufftDoubleComplex* data, int rank, int size, PetaPM *pm, cudaStream_t stream) {
    auto [begin_d, end_d] = BoxIterators(box_complex, data);
    const size_t num_elements = std::distance(begin_d, end_d);
    const size_t num_threads  = 256;
    const size_t num_blocks   = (num_elements + num_threads - 1) / num_threads;
    force_transfer_kernel<<<num_blocks, num_threads, 0, stream>>>(begin_d, end_d, pm, 0);
}

extern "C" void launch_force_y_transfer(Box3D box_complex, cufftDoubleComplex* data, int rank, int size, PetaPM *pm, cudaStream_t stream) {
    auto [begin_d, end_d] = BoxIterators(box_complex, data);
    const size_t num_elements = std::distance(begin_d, end_d);
    const size_t num_threads  = 256;
    const size_t num_blocks   = (num_elements + num_threads - 1) / num_threads;
    force_transfer_kernel<<<num_blocks, num_threads, 0, stream>>>(begin_d, end_d, pm, 1);
}

extern "C" void launch_force_z_transfer(Box3D box_complex, cufftDoubleComplex* data, int rank, int size, PetaPM *pm, cudaStream_t stream) {
    auto [begin_d, end_d] = BoxIterators(box_complex, data);
    const size_t num_elements = std::distance(begin_d, end_d);
    const size_t num_threads  = 256;
    const size_t num_blocks   = (num_elements + num_threads - 1) / num_threads;
    force_transfer_kernel<<<num_blocks, num_threads, 0, stream>>>(begin_d, end_d, pm, 2);
}











