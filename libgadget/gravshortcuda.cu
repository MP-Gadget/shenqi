/*! \file gravtree.c
 *  \brief main driver routines for gravitational (short-range) force computation
 *
 *  This contains the cuda-specific function calls. A separate file only because the C++ compilers cannot always understand cuda.
 */
#include <stdlib.h>
#include <math.h>

#include "gravity.h"
#include "gravshort2.hpp"

#include "treewalk2.cuh"
class GravTreeWalkGPU : public TreeWalkGPU <GravTreeWalkGPU, GravTreeQuery, GravTreeResult, GravLocalTreeWalk, GravTopTreeWalk, GravTreeParams, GravTreeOutput> {
    public:
    using TreeWalk::TreeWalk;
};

/*! CUDA treewalk.
 */
void
grav_short_tree_cuda(const ActiveParticles * act, ForceTree * tree, GravTreeParams * priv, GravTreeOutput * output, particle_data * const parts, const size_t MaxExportBufferBytes, MPI_Comm comm)
{
        GravTreeWalkGPU tw("GRAVTREE", tree, *priv, output);
        tw.run_on_queue(act->ActiveParticle, act->NumActiveParticle, parts, comm, MaxExportBufferBytes);
        tw.print_stats("/Tree", comm);
}
