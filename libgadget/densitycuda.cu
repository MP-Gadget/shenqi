/*!
 *  CUDA routines for the density (looped) computation
 *
 *  This contains the cuda-specific function calls. A separate file only because the C++ compilers cannot always understand cuda.
 */
#include <stdlib.h>
#include <math.h>

#include "densitytree2.hpp"
#include "treewalk2.cuh"

class DensityTreeWalkGPU: public TreeWalk<DensityTreeWalkGPU, DensityQuery, DensityResult, DensityLocalTreeWalk, DensityTopTreeWalk, DensityPriv, DensityOutput> {
    public:
    using TreeWalk::TreeWalk;
};

/*! CUDA treewalk.
 */
void
density_cuda(const ActiveParticles * act, const ForceTree * tree, DensityPriv * priv, DensityOutput * output, particle_data * const parts, int update_hsml, MPI_Comm comm)
{
    DensityTreeWalkGPU tw("DENSITY", tree, *priv, output);
    /* Do the treewalk with looping for hsml*/
    tw.do_hsml_loop(act->ActiveParticle, act->NumActiveParticle, update_hsml, parts);
    tw.print_stats("/SPH/Density", comm);
}
