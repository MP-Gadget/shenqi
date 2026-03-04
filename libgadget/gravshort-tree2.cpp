/*! \file gravtree.c
 *  \brief main driver routines for gravitational (short-range) force computation
 *
 *  This file contains the code for the gravitational force computation by
 *  means of the tree algorithm. To this end, a tree force is computed for all
 *  active local particles, and particles are exported to other processors if
 *  needed, where they can receive additional force contributions. If the
 *  TreePM algorithm is enabled, the force computed will only be the
 *  short-range part.
 */

#include <mpi.h>
#include <stdlib.h>
#include <math.h>

#include "libgadget/types.h"
#include "utils/endrun.h"
#include "utils/mymalloc.h"
#include "treewalk2.h"
#include "localtreewalk2.h"

#include "forcetree.h"
#include "timestep.h"
#include "walltime.h"
#include "partmanager.h"
#include "gravity.h"
#include "gravshort2.hpp"

GravShortTable::GravShortTable(const enum ShortRangeForceWindowType ShortRangeForceWindowType, const double Asmth)
{
    /*
        * This is a table for the short-range gravity acceleration.
        * This table is computed by comparing with brute force calculation it matches the full PM exact up to 10 mesh sizes
        * for a point source. it is copied to a tighter array for better cache performance (hopefully)
        *
        * Generated with split = 1.25; check with the assertion above!
        * */
    #include "shortrange-kernel.c"
    #define NGRAVTAB2 (sizeof(shortrange_force_kernels) / sizeof(shortrange_force_kernels[0]))
    static_assert(NGRAVTAB == NGRAVTAB2, "Short-range force tables do not match static memory allocation");

    if (ShortRangeForceWindowType == SHORTRANGE_FORCE_WINDOW_TYPE_EXACT) {
        if(Asmth != 1.5) {
            endrun(0, "The short range force window is calibrated for Asmth = 1.5, but running with %g\n", Asmth);
        }
    }

    dx = shortrange_force_kernels[1][0];

    for(size_t i = 0; i < NGRAVTAB; i++)
    {
        /* force_kernels is in units of mesh points; */
        double u = shortrange_force_kernels[i][0] * 0.5 / Asmth;
        switch (ShortRangeForceWindowType) {
            case SHORTRANGE_FORCE_WINDOW_TYPE_EXACT:
                /* Notice that the table is only calibrated for smth of 1.25*/
                shortrange_table[i] = shortrange_force_kernels[i][2]; /* ~ erfc(u) + 2.0 * u / sqrt(M_PI) * exp(-u * u); */
                /* The potential of the calibrated kernel is a bit off, so we still use erfc here; we do not use potential anyways.*/
                shortrange_table_potential[i] = shortrange_force_kernels[i][1];
            break;
            case SHORTRANGE_FORCE_WINDOW_TYPE_ERFC:
                shortrange_table[i] = erfc(u) + 2.0 * u / sqrt(M_PI) * exp(-u * u);
                shortrange_table_potential[i] = erfc(u);
            break;
        }
        /* we don't have a table for that and don't use it anyways. */
        //shortrange_table_tidal[i] = 4.0 * u * u * u / sqrt(M_PI) * exp(-u * u);
    }
}

/* Compute force factor (*fac) and multiply potential (*pot) by the shortrange force window function.
 * If the distance is outside the range of the table, 1 is returned and the caller should assume zero acceleration.
 * If the distance is inside the range of the table, 0 is returned and the force factor and potential values
 * should be applied to the particle query. */
MYCUDAFN int
GravShortTable::apply_short_range_window(const double r, double * fac, double * pot, const double cellsize) const
{
    const double i = (r / cellsize / dx);
    size_t tabindex = floor(i);
    if(tabindex >= NGRAVTAB - 1)
        return 1;
    /* use a linear interpolation; */
    *fac *= (tabindex + 1 - i) * shortrange_table[tabindex] + (i - tabindex) * shortrange_table[tabindex + 1];
    *pot *= (tabindex + 1 - i) * shortrange_table_potential[tabindex] + (i - tabindex) * shortrange_table_potential[tabindex];
    return 0;
}

static struct gravshort_tree_params TreeParams;
/*Softening length*/
static double GravitySoftening;

/* gravitational softening length
 * (given in terms of an `equivalent' Plummer softening length)
 */
double FORCE_SOFTENING(void)
{
    /* Force is Newtonian beyond this.*/
    return 2.8 * GravitySoftening;
}

/*! Sets the (comoving) softening length, converting from units of the mean separation to comoving internal units. */
void
gravshort_set_softenings(double MeanSeparation)
{
    GravitySoftening = TreeParams.FractionalGravitySoftening * MeanSeparation;
    /* 0: Gas is collisional */
    message(0, "GravitySoftening = %g\n", GravitySoftening);
}

/*This is a helper for the tests*/
void set_gravshort_treepar(struct gravshort_tree_params tree_params)
{
    TreeParams = tree_params;
}

struct gravshort_tree_params get_gravshort_treepar(void)
{
    return TreeParams;
}

/* Sets up the module*/
void
set_gravshort_tree_params(ParameterSet * ps)
{
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    if(ThisTask == 0) {
        TreeParams.BHOpeningAngle = param_get_double(ps, "BHOpeningAngle");
        TreeParams.ErrTolForceAcc = param_get_double(ps, "ErrTolForceAcc");
        TreeParams.TreeUseBH= param_get_int(ps, "TreeUseBH");
        TreeParams.Rcut = param_get_double(ps, "TreeRcut");
        TreeParams.FractionalGravitySoftening = param_get_double(ps, "GravitySoftening");
        TreeParams.MaxBHOpeningAngle = param_get_double(ps, "MaxBHOpeningAngle");
        TreeParams.ShortRangeForceWindowType = (enum ShortRangeForceWindowType) param_get_enum(ps, "ShortRangeForceWindowType");
        /* This size is the maximum allowed without the MPI library breaking.*/
        TreeParams.MaxExportBufferBytes = 3584*1024*1024L;
    }
    MPI_Bcast(&TreeParams, sizeof(struct gravshort_tree_params), MPI_BYTE, 0, MPI_COMM_WORLD);
}

class GravTreeWalk : public TreeWalk <GravTreeWalk, GravTreeQuery, GravTreeResult, GravLocalTreeWalk, GravTopTreeWalk, GravTreeParams, GravTreeOutput> {
    public:
    using TreeWalk::TreeWalk;
};

/*! This function computes the gravitational forces for all active particles from all particles in the tree.
 * Particles are only exported to other processors when really
 *  needed, thereby allowing a good use of the communication buffer.
 *  NeutrinoTracer = All.HybridNeutrinosOn && (atime <= All.HybridNuPartTime);
 *  rho0 = CP.Omega0 * 3 * CP.Hubble * CP.Hubble / (8 * M_PI * G)
 *  ActiveParticle should contain only gravitationally active particles.
 *  If this tree contains all particles, as specified by the full_particle_tree_flag, we calculate the short-
 * range gravitational potential and update the fulltreegravaccel. Note that in practice
 * for hierarchical gravity only active particles are in the tree and so this is
 * only true on PM steps where all particles are active.
 */
void
grav_short_tree(const ActiveParticles * act, PetaPM * pm, ForceTree * tree, MyFloat (* AccelStore)[3], double rho0, inttime_t Ti_Current, bool use_gpu)
{
    GravShortTable gravtab(TreeParams.ShortRangeForceWindowType, pm->Asmth);
    GravTreeParams * priv = (GravTreeParams *) mymanagedmalloc("GravTreeParams", sizeof(GravTreeParams));
    new (priv) GravTreeParams(TreeParams, Ti_Current, rho0, pm, tree->BoxSize, gravtab);

    GravTreeOutput * output = (GravTreeOutput *) mymanagedmalloc("GravTreeOutput", sizeof(GravTreeOutput));
    new(output) GravTreeOutput(AccelStore, PartManager->NumPart, tree->full_particle_tree_flag);

    /* Do the treewalk! Run directly on the active list as we want to use all particles. */
    #ifdef USE_CUDA
    if(use_gpu) {
        grav_short_tree_cuda(act, tree, priv, output, PartManager->Base, TreeParams.MaxExportBufferBytes, MPI_COMM_WORLD);
    } else
    #endif
    {
        GravTreeWalk tw("GRAVTREE", tree, *priv, output);
        tw.run_on_queue(act->ActiveParticle, act->NumActiveParticle, PartManager->Base, MPI_COMM_WORLD, TreeParams.MaxExportBufferBytes);
        tw.print_stats("/Tree", MPI_COMM_WORLD);
    }

    output->~GravTreeOutput();
    myfree(output);
    priv->~GravTreeParams();
    myfree(priv);

    /* TreeUseBH > 1 means use the BH criterion on the initial timestep only,
     * avoiding the fully open O(N^2) case.*/
    if(TreeParams.TreeUseBH > 1)
        TreeParams.TreeUseBH = 0;
}
