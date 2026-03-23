/*!
 *  CUDA routines for the density (looped) computation
 *
 *  This contains the cuda-specific function calls. A separate file only because the C++ compilers cannot always understand cuda.
 */
#include <stdlib.h>
#include <math.h>

#include "densitytree2.hpp"
#include "treewalk2.cuh"

class DensityTreeWalkGPUCubic: public TreeWalk<DensityTreeWalkGPUCubic, DensityQuery, DensityResult, DensityLocalTreeWalk<CubicDensityKernel>, DensityTopTreeWalk, DensityPriv, DensityOutput> {
    public:
    using TreeWalk::TreeWalk;
};

class DensityTreeWalkGPUQuartic: public TreeWalk<DensityTreeWalkGPUQuartic, DensityQuery, DensityResult, DensityLocalTreeWalk<QuarticDensityKernel>, DensityTopTreeWalk, DensityPriv, DensityOutput> {
    public:
    using TreeWalk::TreeWalk;
};

class DensityTreeWalkGPUQuintic: public TreeWalk<DensityTreeWalkGPUQuintic, DensityQuery, DensityResult, DensityLocalTreeWalk<QuinticDensityKernel>, DensityTopTreeWalk, DensityPriv, DensityOutput> {
    public:
    using TreeWalk::TreeWalk;
};


/*! CUDA treewalk.
 */
void
density_cuda(const ActiveParticles * act, const ForceTree * tree, DensityPriv * priv, DensityOutput * output, particle_data * const parts, int update_hsml, enum DensityKernelType DensityKernelType, MPI_Comm comm)
{
    switch(DensityKernelType) {
        case DENSITY_KERNEL_CUBIC_SPLINE:
            do_density_walk<DensityTreeWalkGPUCubic>(act, tree, priv, output, PartManager->Base, update_hsml, MPI_COMM_WORLD);
            break;
        case DENSITY_KERNEL_QUARTIC_SPLINE:
            do_density_walk<DensityTreeWalkGPUQuartic>(act, tree, priv, output, PartManager->Base, update_hsml, MPI_COMM_WORLD);
            break;
        default: //DENSITY_KERNEL_QUINTIC_SPLINE
            do_density_walk<DensityTreeWalkGPUQuintic>(act, tree, priv, output, PartManager->Base, update_hsml, MPI_COMM_WORLD);
            break;
    }
}
