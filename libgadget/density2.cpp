#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#include "libgadget/localtreewalk2.h"
#include "libgadget/partmanager.h"
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

        DensityKernel kernel;
        density_kernel_init(&kernel, 1.0, DensityParams.DensityKernelType);
        message(1, "The Density Kernel type is %s\n", kernel.name);
        message(1, "The Density resolution is %g * mean separation, or %g neighbours\n",
                    DensityParams.DensityResolutionEta, GetNumNgb(GetDensityKernelType()));
        /*These two look like black hole parameters but they are really neighbour finding parameters*/
        DensityParams.BlackHoleNgbFactor = param_get_double(ps, "BlackHoleNgbFactor");
        DensityParams.BlackHoleMaxAccretionRadius = param_get_double(ps, "BlackHoleMaxAccretionRadius");
    }
    MPI_Bcast(&DensityParams, sizeof(struct density_params), MPI_BYTE, 0, MPI_COMM_WORLD);
}

double
GetNumNgb(enum DensityKernelType KernelType)
{
    DensityKernel kernel;
    density_kernel_init(&kernel, 1.0, KernelType);
    return density_kernel_desnumngb(&kernel, DensityParams.DensityResolutionEta);
}

enum DensityKernelType
GetDensityKernelType(void)
{
    return DensityParams.DensityKernelType;
}

/* The evolved entropy at drift time: evolved dlog a.
 * Used to predict pressure and entropy for SPH */
MYCUDAFN MyFloat
SPH_EntVarPred(const particle_data& particle, const DriftKickTimes * times)
{
        const int bin = particle.TimeBinHydro;
        const int PI = particle.PI;
        const double dloga = dloga_from_dti(times->Ti_Current - times->Ti_kick[bin], times->Ti_Current);
        double EntVarPred = SphP[PI].Entropy + SphP[PI].DtEntropy * dloga;
        /*Entropy limiter for the predicted entropy: makes sure entropy stays positive. */
        if(EntVarPred < 0.05*SphP[PI].Entropy)
            EntVarPred = 0.05 * SphP[PI].Entropy;
        /* Just in case*/
        if(EntVarPred <= 0)
            return 0;
        EntVarPred = exp(1./GAMMA * log(EntVarPred));
//         EntVarPred = pow(EntVarPred, 1/GAMMA);
        return EntVarPred;
}

class DensityTreeWalk: public TreeWalk<DensityTreeWalk, DensityQuery, DensityResult, DensityLocalTreeWalk, DensityTopTreeWalk, DensityPriv, DensityOutput> {
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
density(const ActiveParticles * act, int update_hsml, int DoEgyDensity, int BlackHoleOn, DriftKickTimes& times, Cosmology * CP, MyFloat ** EntVarPred, MyFloat * GradRho_mag, const ForceTree * const tree)
{
    /* This ensures these classes are in managed memory and so accessible on the device. */
    DensityPriv * priv = (DensityPriv *) mymanagedmalloc("DensityPriv", sizeof(DensityPriv));
    new (priv) DensityPriv(update_hsml, DoEgyDensity, BlackHoleOn, &times, tree->BoxSize, CP, act, PartManager);
    DensityOutput * output = (DensityOutput *) mymanagedmalloc("DensityOutput", sizeof(DensityOutput));
    new (output) DensityOutput(GradRho_mag, PartManager->NumPart, SlotsManager->info[0].size, tree->BoxSize);

    walltime_measure("/SPH/Density/Init");

#ifdef USE_CUDA
    if(TreeParams.UseGPU) {
        density_cuda(act, tree, priv, output, PartManager->Base, update_hsml, MPI_COMM_WORLD);
    } else
#endif
    {
        DensityTreeWalk tw("DENSITY", tree, *priv, output);
        /* Do the treewalk with looping for hsml*/
        tw.do_hsml_loop(act->ActiveParticle, act->NumActiveParticle, update_hsml, PartManager->Base);
        tw.print_stats("/SPH/Density", MPI_COMM_WORLD);
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
    priv->~DensityPriv();
    myfree(priv);

    *EntVarPred = priv->EntVarPred;
    /* collect some timing information */
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
