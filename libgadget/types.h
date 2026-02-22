#ifndef __TYPES_H
#define __TYPES_H

#include <stdint.h>

/*Define some useful types*/

typedef int64_t inttime_t;

typedef uint64_t MyIDType;
#define IDTYPE_MAX UINT64_MAX

#ifndef LOW_PRECISION
#define LOW_PRECISION double
#endif

typedef LOW_PRECISION MyFloat;

#define HAS(val, flag) ((flag & (val)) == (flag))

/* Functions which need to be called in device context should include this macro in the definition.
 * When being compiled by a non-CUDA compiler, the macro will compile to nothing.
 * Functions not decorated default to __host__. Only functions with the __device__ marker can be
 * called from the device. Functions which are __device__ or __global__ should be declared in a .cu file.
 */
#if defined(USE_CUDA) && defined(__CUDACC__)
#define MYCUDAFN __host__ __device__
#else
#define MYCUDAFN
#endif

#endif
