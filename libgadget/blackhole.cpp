#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#include "physconst.h"
#include "gravity.h"
#include "slotsmanager.h"
#include "blackhole.h"
#include "timestep.h"
#include "density.h"
#include "sfr_eff.h"
#include "types.h"
#include "winds.h"
#include "walltime.h"
#include "bhinfo.h"
#include "bhdynfric.h"
#include "utils/endrun.h"
#include "utils/mymalloc.h"

/*! \file blackhole.c
 *  \brief routines for gas accretion onto black holes, and black hole mergers
 */

struct BlackholeParams blackhole_params;


/*****************************************************************************/

/*Set the parameters of the BH module*/
void set_blackhole_params(ParameterSet * ps)
{
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    if(ThisTask == 0) {
        blackhole_params.BlackHoleAccretionFactor = param_get_double(ps, "BlackHoleAccretionFactor");
        blackhole_params.BlackHoleEddingtonFactor = param_get_double(ps, "BlackHoleEddingtonFactor");
        blackhole_params.BlackHoleFeedbackFactor = param_get_double(ps, "BlackHoleFeedbackFactor");

        blackhole_params.BlackHoleKineticOn = param_get_int(ps,"BlackHoleKineticOn");
        blackhole_params.BHKE_EddingtonThrFactor = param_get_double(ps, "BHKE_EddingtonThrFactor");
        blackhole_params.BHKE_EddingtonMFactor = param_get_double(ps, "BHKE_EddingtonMFactor");
        blackhole_params.BHKE_EddingtonMPivot = param_get_double(ps, "BHKE_EddingtonMPivot");
        blackhole_params.BHKE_EddingtonMIndex = param_get_double(ps, "BHKE_EddingtonMIndex");
        blackhole_params.BHKE_EffRhoFactor = param_get_double(ps, "BHKE_EffRhoFactor");
        blackhole_params.BHKE_EffCap = param_get_double(ps, "BHKE_EffCap");
        blackhole_params.BHKE_InjEnergyThr = param_get_double(ps, "BHKE_InjEnergyThr");
        blackhole_params.BHKE_SfrCritOverDensity = param_get_double(ps, "CritOverDensity");
        /***********************************************************************************/
        blackhole_params.BH_DRAG = param_get_int(ps, "BH_DRAG");
        blackhole_params.MergeGravBound = param_get_int(ps, "MergeGravBound");
        blackhole_params.SeedBHDynMass = param_get_double(ps,"SeedBHDynMass");


        blackhole_params.SeedBlackHoleMass = param_get_double(ps, "SeedBlackHoleMass");
        blackhole_params.MaxSeedBlackHoleMass = param_get_double(ps,"MaxSeedBlackHoleMass");
        blackhole_params.SeedBlackHoleMassIndex = param_get_double(ps,"SeedBlackHoleMassIndex");
        /***********************************************************************************/
    }
    MPI_Bcast(&blackhole_params, sizeof(struct BlackholeParams), MPI_BYTE, 0, MPI_COMM_WORLD);

    set_blackhole_dynfric_params(ps);
}

/* The accretion treewalk, from bhaccretion2.cpp. The tree needs to have gas and BH and is the new-style tree.*/
void blackhole_accretion(int * ActiveBlackHoles, const int64_t NumActiveBlackHoles, ForceTreeHmax * tree, DomainDecomp * ddecomp, struct BHPriv * bhpriv, const KickFactorData& kf, TimeBinMgr * timebinmgr, MPI_Comm comm);

/* The feedback treewalk, from bhfeedback2.cpp. Uses the same tree as the accretion treewalk.*/
void blackhole_feedback(int * ActiveBlackHoles, const int64_t NumActiveBlackHoles, const ForceTreeHmax * const tree, struct BHPriv * bhpriv, const KickFactorData& kf, TimeBinMgr * timebinmgr, MPI_Comm comm);

/*************************************************************************************/

/* Build a list of active black holes, done once and reused for all the later treewalks.
 * These are the black holes which are not being swallowed.*/
int
blackholes_active(const ActiveParticles * act, int ** ActiveBlackHoles, int64_t * NumActiveBlackHoles)
{
    const int64_t nactive = act->NumActiveParticle;
    int * WorkSet = mymalloc("ActiveQueue", int, nactive + 1);
    int64_t nbh = 0;
    /* This is not threaded so that the list stays in order.*/
    for(int64_t i = 0; i < nactive; i++) {
        /*Use raw particle number if the active list is null, otherwise use the active list*/
        const int p_i = act->ActiveParticle ? act->ActiveParticle[i] : (int) i;
        if(Part[p_i].IsGarbage || Part[p_i].Swallowed || Part[p_i].Type != 5)
            continue;
        WorkSet[nbh++] = p_i;
    }
    /* If this queue is empty, nothing to do.*/
    int64_t totbh;
    MPI_Allreduce(&nbh, &totbh, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);

    /* Move the working set high so we can keep the tree we build after making the list and still free this active set.*/
    *NumActiveBlackHoles = nbh;
    if(totbh > 0) {
        *ActiveBlackHoles = mymalloc2("activeBH", int, nbh);
        memcpy(*ActiveBlackHoles, WorkSet, nbh * sizeof(int));
    }
    myfree(WorkSet);
    return totbh;
}

void
blackhole(const ActiveParticles * act, double atime, Cosmology * CP, ForceTreeHmax * gasTree, DomainDecomp * ddecomp, DriftKickTimes * times, TimeBinMgr * timebinmgr, RandTable * rnd, const struct UnitSystem units, FILE * FdBlackHoles, FILE * FdBlackholeDetails, size_t * bhdetailswritten)
{
    /* Do nothing if no black holes*/
    int64_t totbh;
    MPI_Allreduce(&SlotsManager->info[5].size, &totbh, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);
    if(totbh == 0)
        return;

    walltime_measure("/Misc");
    /* Build the queue once, since it is really 'all black holes' and similar for all treewalks.*/
    int * ActiveBlackHoles = NULL;
    int64_t NumActiveBlackHoles = 0;
    totbh = blackholes_active(act, &ActiveBlackHoles, &NumActiveBlackHoles);
    if(totbh == 0) {
        return;
    }

    /* Types used in treewalks:
     * accretion uses: gas + black holes (to flag mergers).
     * feedback uses: gas + black holes (to flag mergers).
     */
    if(!gasTree || !(gasTree->mask & GASMASK) || !(gasTree->mask & BHMASK))
        endrun(5, "Blackhole called with bad tree: mask %d want %d\n", gasTree ? gasTree->mask : 0, GASMASK | BHMASK);

    KickFactorData kf2(times, timebinmgr);
    /*************************************************************************/
    /*  Dynamical Friction Treewalk */
    /*************************************************************************/
    /* Update the kernel quantities for dynamic friction, if required.
     * This takes place on a longer timestep than the hydro acceleration
     * to avoid extra treebuilds. Note this includes the potential minimum.
     * If black hole repositioning is on, the treewalk to reposition
     * to the local potential minimum is run.*/
    blackhole_dynfric(ActiveBlackHoles, NumActiveBlackHoles, ddecomp, kf2, times->Ti_Current, MPI_COMM_WORLD);
    /* Compute the DF acceleration for all active black holes*/
    blackhole_dfaccel(ActiveBlackHoles, NumActiveBlackHoles, atime, CP->GravInternal);

    walltime_measure("/BH/DynFric");

    struct BHPriv priv[1] = {0};
    priv->units = units;
    priv->rnd = rnd;
    /*************************************************************************/
    priv->atime = atime;
    priv->a3inv = 1./(atime * atime * atime);
    priv->hubble = hubble_function(CP, atime);
    priv->CP = CP;

    /* Let's determine which gas particles may be swallowed and calculate total feedback weights */
    priv->SPH_SwallowID = mymalloc("SPH_SwallowID", MyIDType,  SlotsManager->info[0].size);
    memset(priv->SPH_SwallowID, 0, SlotsManager->info[0].size * sizeof(MyIDType));
    /* Let's determine which BHs may be swallowed and calculate total feedback weights */
    priv->BH_SwallowID = mymalloc("BH_SwallowID", MyIDType,  SlotsManager->info[5].size);
    memset(priv->BH_SwallowID, 0, SlotsManager->info[5].size * sizeof(MyIDType));

    /* Computed in accretion, used in feedback*/
    priv->BH_FeedbackWeightSum = mymalloc("BH_FeedbackWeightSum", MyFloat, SlotsManager->info[5].size);

    /* Local to this treewalk*/
    priv->BH_Entropy = mymalloc("BH_Entropy", MyFloat, SlotsManager->info[5].size);
    priv->BH_SurroundingGasVel = mymalloc("BH_SurroundVel", My3Vec, SlotsManager->info[5].size);

    /* For AGN kinetic feedback */
    priv->NumDM = mymalloc("NumDM", MyFloat, SlotsManager->info[5].size);
    priv->MgasEnc = mymalloc("MgasEnc", MyFloat, SlotsManager->info[5].size);
    /* mark the state of AGN kinetic feedback */
    priv->KEflag = mymalloc("KEflag", int, SlotsManager->info[5].size);

    walltime_measure("/BH/Init");

    /*************************************************************************/
    /* This treewalk marks all black holes and gas which can be swallowed with a SwllowID of a potential swallower.
     * The treewalk is symmetric. A swallower needs to be active, the black holes must be within each other's
     * smoothing radius and optionally gravitationally bound. In case a black hole can be swallowed by multiple  */
    blackhole_accretion(ActiveBlackHoles, NumActiveBlackHoles, gasTree, ddecomp, priv, kf2, timebinmgr, MPI_COMM_WORLD);

    /*************************************************************************/

    walltime_measure("/BH/Accretion");

    priv->BH_accreted_Mass = mymalloc("BH_accretedmass", MyFloat,  SlotsManager->info[5].size);
    priv->BH_accreted_BHMass = mymalloc("BH_accreted_BHMass", MyFloat, SlotsManager->info[5].size);
    priv->BH_accreted_momentum = mymalloc("BH_accretemom", My3Vec, SlotsManager->info[5].size);

    /* Now do the swallowing of particles and dump feedback energy */
    /* We also merge BHs here. Only BHs which are not themselves
     * being swallowed are eligible to swallow. If there are multiple BHs within a search radius,
     * the BH with the largest ID will be selected by the SwallowID search and will swallow all the
     * surrounding BHs.
     * Example 1:
     * We have BHs A,B,C, where A and B are close and B and C are close. B.ID < C.ID and A.ID < B.ID.
     * B will be swallowed by C. A will not be swallowed by B as B is itself being swallowed (A will
     * have a non-zero encounter).
     * Example 2:
     * We have BHs A,B,C, where A and C are both close to B. B.ID > C.ID and A.ID < B.ID.
     * In this case B will swallow C and A.
     * Example 3:
     * We have BHs A,B,C, where A and B are close and B and C are close. B.ID < C.ID and A.ID > B.ID.
     * In this case B will be swallowed by whichever of A and C has the larger ID.
    */
    blackhole_feedback(ActiveBlackHoles, NumActiveBlackHoles, gasTree, priv, kf2, timebinmgr, MPI_COMM_WORLD);

    walltime_measure("/BH/Feedback");

    if(FdBlackholeDetails){
        *bhdetailswritten += collect_BH_info(ActiveBlackHoles, NumActiveBlackHoles, priv, PartManager, (struct bh_particle_data*) SlotsManager->info[5].ptr, FdBlackholeDetails);
    }

    myfree(priv->BH_accreted_momentum);
    myfree(priv->BH_accreted_BHMass);
    myfree(priv->BH_accreted_Mass);

    /*****************************************************************/
    myfree(priv->KEflag);
    myfree(priv->MgasEnc);
    myfree(priv->NumDM);

    myfree(priv->BH_SurroundingGasVel);
    myfree(priv->BH_Entropy);

    myfree(priv->BH_FeedbackWeightSum);
    myfree(priv->BH_SwallowID);
    myfree(priv->SPH_SwallowID);

    myfree(ActiveBlackHoles);

    write_blackhole_txt(FdBlackHoles, units, atime);
    walltime_measure("/BH/Info");
}

/* Sample from a power law to get the initial BH mass*/
static double
bh_powerlaw_seed_mass(const MyIDType ID, const RandTable * const rnd)
{
    /* compute random number, uniform in [0,1] */
    const double w = get_random_number(ID+23, rnd);
    /* Normalisation for this power law index*/
    double norm = pow(blackhole_params.MaxSeedBlackHoleMass, 1+blackhole_params.SeedBlackHoleMassIndex)
                - pow(blackhole_params.SeedBlackHoleMass, 1+blackhole_params.SeedBlackHoleMassIndex);
    /* Sample from the CDF:
     * w  = [M^(1+x) - M_0^(1+x)]/[M_1^(1+x) - M_0^(1+x)]
     * w [M_1^(1+x) - M_0^(1+x)] + M_0^(1+x) = M^(1+x)
     * M = pow((w [M_1^(1+x) - M_0^(1+x)] + M_0^(1+x)), 1/(1+x))*/
    double mass = pow(w * norm + pow(blackhole_params.SeedBlackHoleMass, 1+blackhole_params.SeedBlackHoleMassIndex),
                      1./(1+blackhole_params.SeedBlackHoleMassIndex));
    return mass;
}

void
blackhole_make_one(int index, const double atime, const RandTable * const rnd) {
    if(Part[index].Type != 0)
        endrun(7772, "Only Gas turns into blackholes, what's wrong?");

    int child = index;

    /* Make the new particle a black hole: use all the Part[i].Mass
     * so we don't have lots of low mass tracers.
     * If the BH seed mass is small this may lead to a mismatch
     * between the gas and BH mass. */
    child = slots_convert(child, 5, -1, PartManager, SlotsManager);

    /* The accretion mass should always be the seed black hole mass,
     * irrespective of the gravitational mass of the particle.*/
    if(blackhole_params.MaxSeedBlackHoleMass > 0)
        BHP(child).Mass = bh_powerlaw_seed_mass(Part[child].ID, rnd);
    else
        BHP(child).Mass = blackhole_params.SeedBlackHoleMass;

    BHP(child).Mseed = BHP(child).Mass;
    BHP(child).Mdot = 0;
    BHP(child).FormationTime = atime;
    BHP(child).SwallowID = (MyIDType) -1;
    BHP(child).Density = 0;
    /* Make sure initially active*/
    BHP(child).TimeBinDynFric = Part[child].TimeBinHydro;

    /* It is important to initialize MinPotPos to the current position of
     * a BH to avoid drifting to unknown locations (0,0,0) immediately
     * after the BH is created. */
    int j;
    for(j = 0; j < 3; j++) {
        BHP(child).MinPotPos[j] = Part[child].Pos[j];
        BHP(child).DFAccel[j] = 0;
        BHP(child).DF_SurroundingVel[j] = 0;
        BHP(child).DragAccel[j] = 0;
    }
    BHP(child).DF_SurroundingRmsVel = 0;
    BHP(child).DF_SurroundingDensity = 0;
    BHP(child).JumpToMinPot = 0;
    BHP(child).CountProgs = 1;

    if (blackhole_params.SeedBHDynMass>0){
        BHP(child).Mtrack = Part[child].Mass;
        Part[child].Mass = blackhole_params.SeedBHDynMass;
    }
    else{
        BHP(child).Mtrack = -1; /* This column is not used then. */
    }
    if(Part[child].Mass < BHP(child).Mass || Part[child].Mass < BHP(child).Mtrack)
        message(1, "WARNING: BH Mass (%g) for ID %ld is larger than particle mass (%g) or mtrack (%g)\n", BHP(child).Mass, Part[child].ID, Part[child].Mass, BHP(child).Mtrack);

    /* Initialize KineticFdbkEnergy, keep zero if BlackHoleKineticOn is not turned on */
    BHP(child).KineticFdbkEnergy = 0;
    BHP(child).VDisp = 0;
}
