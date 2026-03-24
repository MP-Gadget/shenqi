#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#include "localtreewalk2.h"
#include "partmanager.h"
#include "physconst.h"
#include "walltime.h"
#include "density2.h"
#include "treewalk2.h"
#include "slotsmanager.h"
#include "timestep.h"
#include "utils/endrun.h"
#include "utils/mymalloc.h"
#include "gravity.h"
#include "winds.h"
#include "densitytree2.hpp"

static struct density_params DensityParams;

/*Set density module parameters from a density_params struct for the tests*/
void
set_densitypar(struct density_params dp)
{
    DensityParams = dp;
}

/*Get parameters*/
struct density_params get_densitypar(void)
{
    return DensityParams;
}

/*Set the parameters of the density module*/
void
set_density_params(ParameterSet * ps)
{
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    if(ThisTask == 0) {
        DensityParams.DensityKernelType = (enum DensityKernelType) param_get_enum(ps, "DensityKernelType");
        DensityParams.MaxNumNgbDeviation = param_get_double(ps, "MaxNumNgbDeviation");
        DensityParams.DensityResolutionEta = param_get_double(ps, "DensityResolutionEta");
        DensityParams.MinGasHsmlFractional = param_get_double(ps, "MinGasHsmlFractional");
        DensityParams.MinGasHsml = DensityParams.MinGasHsmlFractional * (FORCE_SOFTENING()/2.8);
        DensityParams.BlackHoleNgbFactor = param_get_double(ps, "BlackHoleNgbFactor");
    }
    MPI_Bcast(&DensityParams, sizeof(struct density_params), MPI_BYTE, 0, MPI_COMM_WORLD);
    message(0, "The Density Kernel (type %d) resolution is %g * mean separation, or %g neighbours\n",
        DensityParams.DensityKernelType, DensityParams.DensityResolutionEta, GetNumNgb(GetDensityKernelType()));
}

double
GetNumNgb(enum DensityKernelType KernelType)
{
    double desnumngb;
    if(KernelType == DENSITY_KERNEL_CUBIC_SPLINE)
        desnumngb = CubicDensityKernel::desnumngb(DensityParams.DensityResolutionEta);
    else if (KernelType == DENSITY_KERNEL_QUARTIC_SPLINE)
        desnumngb = QuarticDensityKernel::desnumngb(DensityParams.DensityResolutionEta);
    else
        desnumngb = QuinticDensityKernel::desnumngb(DensityParams.DensityResolutionEta);
    return desnumngb;
}

enum DensityKernelType
GetDensityKernelType(void)
{
    return DensityParams.DensityKernelType;
}

/* The evolved entropy at drift time: evolved dlog a.
 * Used to predict pressure and entropy for SPH */
MYCUDAFN MyFloat
SPH_EntVarPred(const particle_data& particle, const sph_particle_data& sph_part, const DriftKickTimes * times)
{
        const int bin = particle.TimeBinHydro;
        const double dloga = dloga_from_dti(times->Ti_Current - times->Ti_kick[bin], times->Ti_Current);
        double EntVarPred = sph_part.Entropy + sph_part.DtEntropy * dloga;
        /*Entropy limiter for the predicted entropy: makes sure entropy stays positive. */
        if(EntVarPred < 0.05*sph_part.Entropy)
            EntVarPred = 0.05 * sph_part.Entropy;
        /* Just in case*/
        if(EntVarPred <= 0)
            return 0;
        EntVarPred = exp(1./GAMMA * log(EntVarPred));
//         EntVarPred = pow(EntVarPred, 1/GAMMA);
        return EntVarPred;
}

class DensityTreeWalkCubic: public TreeWalk<DensityTreeWalkCubic, DensityQuery, DensityResult, DensityLocalTreeWalk<CubicDensityKernel>, DensityTopTreeWalk, DensityPriv, DensityOutput> {
    public:
    using TreeWalk::TreeWalk;
};

class DensityTreeWalkQuartic: public TreeWalk<DensityTreeWalkQuartic, DensityQuery, DensityResult, DensityLocalTreeWalk<QuarticDensityKernel>, DensityTopTreeWalk, DensityPriv, DensityOutput> {
    public:
    using TreeWalk::TreeWalk;
};

class DensityTreeWalkQuintic: public TreeWalk<DensityTreeWalkQuintic, DensityQuery, DensityResult, DensityLocalTreeWalk<QuinticDensityKernel>, DensityTopTreeWalk, DensityPriv, DensityOutput> {
    public:
    using TreeWalk::TreeWalk;
};

/*! \file density.c
 *  \brief SPH density computation and smoothing length determination
 *
 *  This file contains the "first SPH loop", where the SPH densities and some
 *  auxiliary quantities are computed.  There is also functionality that
 *  corrects the smoothing length if needed.
 */

/*! This function computes the local density for each active SPH particle, the
 * number of neighbours in the current smoothing radius, and the divergence
 * and rotation of the velocity field.  The pressure is updated as well.  If a
 * particle with its smoothing region is fully inside the local domain, it is
 * not exported to the other processors. The function also detects particles
 * that have a number of neighbours outside the allowed tolerance range. For
 * these particles, the smoothing length is adjusted accordingly, and the
 * density() computation is called again.  Note that the smoothing length is
 * not allowed to fall below the lower bound set by MinGasHsml (this may mean
 * that one has to deal with substantially more than normal number of
 * neighbours.)
 */
void
density(const ActiveParticles * act, int update_hsml, int DoEgyDensity, int BlackHoleOn, DriftKickTimes& times, Cosmology * CP, MyFloat ** EntVarPred, MyFloat * GradRho_mag, const ForceTree * const tree, bool UseGPU)
{
    /* This ensures these classes are in managed memory and so accessible on the device. */
    DensityPriv * priv = (DensityPriv *) mymanagedmalloc("DensityPriv", sizeof(DensityPriv));
    new (priv) DensityPriv(DensityParams, update_hsml, DoEgyDensity, BlackHoleOn, &times, tree->BoxSize, CP, act, PartManager);
    DensityOutput * output = (DensityOutput *) mymanagedmalloc("DensityOutput", sizeof(DensityOutput));
    new (output) DensityOutput(GradRho_mag, PartManager->NumPart, SlotsManager->info[0].size, tree->BoxSize, DensityParams.MaxNumNgbDeviation);

    walltime_measure("/SPH/Density/Init");

#ifdef USE_CUDA
    if(UseGPU) {
        density_cuda(act, tree, priv, output, PartManager->Base, update_hsml, DensityParams.DensityKernelType, MPI_COMM_WORLD);
    } else
#endif
    {
        switch(DensityParams.DensityKernelType) {
            case DENSITY_KERNEL_CUBIC_SPLINE:
                do_density_walk<DensityTreeWalkCubic>(act, tree, priv, output, PartManager->Base, update_hsml, MPI_COMM_WORLD);
                break;
            case DENSITY_KERNEL_QUARTIC_SPLINE:
                do_density_walk<DensityTreeWalkQuartic>(act, tree, priv, output, PartManager->Base, update_hsml, MPI_COMM_WORLD);
                break;
            default: //DENSITY_KERNEL_QUINTIC_SPLINE
                do_density_walk<DensityTreeWalkQuintic>(act, tree, priv, output, PartManager->Base, update_hsml, MPI_COMM_WORLD);
                break;
        }
    }

    if(GradRho_mag) {
        int64_t i;
        #pragma omp parallel for
        for(i = 0; i < SlotsManager->info[0].size; i++)
        {
            MyFloat * gr = output->GradRho + (3*i);
            GradRho_mag[i] = sqrt(gr[0]*gr[0] + gr[1] * gr[1] + gr[2] * gr[2]);
        }
    }

    output->~DensityOutput();
    myfree(output);

    *EntVarPred = priv->EntVarPred;
    priv->~DensityPriv();
    myfree(priv);
}

/* Set the initial smoothing length for gas and BH*/
MYCUDAFN void
set_init_hsml(ForceTree * tree, DomainDecomp * ddecomp, const double MeanGasSeparation, struct part_manager_type * const PartManager)
{
    /* Need moments because we use them to set Hsml*/
    force_tree_calc_moments(tree, ddecomp);
    if(!tree->Father)
        endrun(5, "tree Father array not allocated at initial hsml!\n");
    const double DesNumNgb = GetNumNgb(GetDensityKernelType());
    particle_data * const parts = PartManager->Base;
    int i;
    #pragma omp parallel for
    for(i = 0; i < PartManager->NumPart; i++)
    {
        /* These initial smoothing lengths are only used for SPH-like particles.*/
        if(parts[i].Type != 0 && parts[i].Type != 5)
            continue;

        if(parts[i].IsGarbage)
            continue;
        int no = i;

        do {
            int p = force_get_father(no, tree);

            if(p < tree->firstnode)
                break;

            /* Check that we didn't somehow get a bad set of nodes*/
            if(p > tree->numnodes + tree->firstnode)
                endrun(5, "Bad init father: i=%d, mass = %g type %d hsml %g no %d len %g father %d, numnodes %ld firstnode %ld\n",
                    i, parts[i].Mass, parts[i].Type, parts[i].Hsml, no, tree->Nodes[no].len, p, tree->numnodes, tree->firstnode);
            no = p;
        } while(10 * DesNumNgb * parts[i].Mass > tree->Nodes[no].mom.mass);

        /* Validate the tree node contents*/
        if(tree->Nodes[no].len > tree->BoxSize || tree->Nodes[no].mom.mass < parts[i].Mass)
            endrun(5, "Bad tree moments: i=%d, mass = %g type %d hsml %g no %d len %g treemass %g\n",
                    i, parts[i].Mass, parts[i].Type, parts[i].Hsml, no, tree->Nodes[no].len, tree->Nodes[no].mom.mass);
        parts[i].Hsml = MeanGasSeparation;
        if(no >= tree->firstnode) {
            double testhsml = tree->Nodes[no].len * pow(3.0 / (4 * M_PI) * DesNumNgb * parts[i].Mass / tree->Nodes[no].mom.mass, 1.0 / 3);
            /* recover from a poor initial guess */
            if (testhsml < 500. * MeanGasSeparation)
                parts[i].Hsml = testhsml;
        }

        if(parts[i].Hsml <= 0)
            endrun(5, "Bad hsml guess: i=%d, mass = %g type %d hsml %g no %d len %g treemass %g\n",
                    i, parts[i].Mass, parts[i].Type, parts[i].Hsml, no, tree->Nodes[no].len, tree->Nodes[no].mom.mass);
    }
}
