#include <mpi.h>
#include <stdlib.h>
#include <math.h>

#include "timestep.h"
#include "walltime.h"
#include "slotsmanager.h"
#include "treewalk2.h"
#include "density2.h"
#include "utils/endrun.h"
#include "utils/mymalloc.h"
#include "hydra2.h"

#include "hydratree2.hpp"

/*! \file hydra.c
 *  \brief Computation of SPH forces and rate of entropy generation
 *
 *  This file contains the "second SPH loop", where the SPH forces are
 *  computed, and where the rate of change of entropy due to the shock heating
 *  (via artificial viscosity) is computed.
 */

static struct hydro_params HydroParams;

void set_hydropar(struct hydro_params dp)
{
    HydroParams = dp;
}

/*Get parameters*/
struct hydro_params get_hydropar(void)
{
    return HydroParams;
}

/*Set the parameters of the hydro module*/
void
set_hydro_params(ParameterSet * ps)
{
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    if(ThisTask == 0) {
        HydroParams.ArtBulkViscConst = param_get_double(ps, "ArtBulkViscConst");
        HydroParams.DensityContrastLimit = param_get_double(ps, "DensityContrastLimit");
        HydroParams.DensityIndependentSphOn= param_get_int(ps, "DensityIndependentSphOn");
    }
    MPI_Bcast(&HydroParams, sizeof(struct hydro_params), MPI_BYTE, 0, MPI_COMM_WORLD);
}

int DensityIndependentSphOn(void)
{
    return HydroParams.DensityIndependentSphOn;
}


class HydroTreeWalkCubic: public TreeWalk<HydroTreeWalkCubic, HydroQuery, HydroResult, HydroLocalTreeWalk<CubicDensityKernel>, HydroTopTreeWalk, HydroPriv, HydroOutput> {
    public:
    using TreeWalk::TreeWalk;
};

class HydroTreeWalkQuartic: public TreeWalk<HydroTreeWalkQuartic, HydroQuery, HydroResult, HydroLocalTreeWalk<QuarticDensityKernel>, HydroTopTreeWalk, HydroPriv, HydroOutput> {
    public:
    using TreeWalk::TreeWalk;
};

class HydroTreeWalkQuintic: public TreeWalk<HydroTreeWalkQuintic, HydroQuery, HydroResult, HydroLocalTreeWalk<QuinticDensityKernel>, HydroTopTreeWalk, HydroPriv, HydroOutput> {
    public:
    using TreeWalk::TreeWalk;
};

/*! This function is the driver routine for the calculation of hydrodynamical
 *  force and rate of change of entropy due to shock heating for all active
 *  particles .
 */
void
hydro_force(const ActiveParticles * act, const double atime, MyFloat * EntVarPred, DriftKickTimes& times,  Cosmology * CP, const ForceTree * const tree, bool UseGPU)
{
    if(!tree->hmax_computed_flag)
        endrun(5, "Hydro called before hmax computed\n");

    HydroPriv * priv = (HydroPriv *) mymanagedmalloc("GravTreeParams", sizeof(HydroPriv));
    new (priv) HydroPriv(tree->BoxSize, EntVarPred, atime, &times, timebinmgr, CP, HydroParams);

    walltime_measure("/SPH/Hydro/Init");

    HydroOutput output(SlotsManager);

#ifdef USE_CUDA
    if(UseGPU)
        hydro_force_cuda(act, tree, priv, &output, GetDensityKernelType());
    else
#endif
    {
        switch(GetDensityKernelType()) {
            case DENSITY_KERNEL_CUBIC_SPLINE:
                do_hydro_walk<HydroTreeWalkCubic>(act, tree, priv, &output);
                break;
            case DENSITY_KERNEL_QUARTIC_SPLINE:
                do_hydro_walk<HydroTreeWalkQuartic>(act, tree, priv, &output);
                break;
            default: //DENSITY_KERNEL_QUINTIC_SPLINE
                do_hydro_walk<HydroTreeWalkQuintic>(act, tree, priv, &output);
                break;
        }
    }
    priv->~HydroPriv();
    myfree(priv);

}
