/*!
 *  CUDA routines for the density (looped) computation
 *
 *  This contains the cuda-specific function calls. A separate file only because the C++ compilers cannot always understand cuda.
 */
#include "hydratree2.hpp"
#include "treewalk2.cuh"

class HydroTreeWalkGPUCubic: public TreeWalkGPU<HydroTreeWalkGPUCubic, HydroQuery, HydroResult, HydroLocalTreeWalk<CubicDensityKernel>, HydroTopTreeWalk, HydroPriv, HydroOutput> {
    public:
    using TreeWalkGPU::TreeWalkGPU;
};

class HydroTreeWalkGPUQuartic: public TreeWalkGPU<HydroTreeWalkGPUQuartic, HydroQuery, HydroResult, HydroLocalTreeWalk<QuarticDensityKernel>, HydroTopTreeWalk, HydroPriv, HydroOutput> {
    public:
    using TreeWalkGPU::TreeWalkGPU;
};

class HydroTreeWalkGPUQuintic: public TreeWalkGPU<HydroTreeWalkGPUQuintic, HydroQuery, HydroResult, HydroLocalTreeWalk<QuinticDensityKernel>, HydroTopTreeWalk, HydroPriv, HydroOutput> {
    public:
    using TreeWalkGPU::TreeWalkGPU;
};


/*! CUDA treewalk.
 */
void
hydro_force_cuda(const ActiveParticles * act, const ForceTree * tree, HydroPriv * priv, HydroOutput * output, enum DensityKernelType DensityKernelType)
{
    switch(DensityKernelType) {
        case DENSITY_KERNEL_CUBIC_SPLINE:
            do_hydro_walk<HydroTreeWalkGPUCubic>(act, tree, priv, output);
            break;
        case DENSITY_KERNEL_QUARTIC_SPLINE:
            do_hydro_walk<HydroTreeWalkGPUQuartic>(act, tree, priv, output);
            break;
        default: //DENSITY_KERNEL_QUINTIC_SPLINE
            do_hydro_walk<HydroTreeWalkGPUQuintic>(act, tree, priv, output);
            break;
    }
}
