#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#include "physconst.h"
#include "gravity.h"
#include "densitykernel.h"
#include "treewalk.h"
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

/* Do the black hole feedback tree walk. Tree needs to have gas and BH.*/
static void
blackhole_feedback(int * ActiveBlackHoles, int64_t NumActiveBlackHoles, ForceTree * tree, struct BHPriv * priv);

/*************************************************************************************/

/*******************************************************************/
static int
blackhole_haswork(int n, TreeWalk * tw){
    /*Black hole not being swallowed*/
    return (Part[n].Type == 5) && (!Part[n].Swallowed);
}

/* Build a list of active black holes, done once and reused for all the later treewalks.*/
int
blackholes_active(const ActiveParticles * act, int ** ActiveBlackHoles, int64_t * NumActiveBlackHoles)
{
    TreeWalk tw_bh[1] = {{0}};
    tw_bh->haswork = blackhole_haswork;

    /* Build the queue once, since it is really 'all black holes' and similar for all treewalks.*/
    treewalk_build_queue(tw_bh, act->ActiveParticle, act->NumActiveParticle, 0);
    /* If this queue is empty, nothing to do.*/
    int64_t totbh;
    MPI_Allreduce(&tw_bh->WorkSetSize, &totbh, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);

    /* Now we have a BH queue and we can re-use it. Create a new variable so that
     * treewalk_run does not mess with the pointer. */
    /* Move the working set high so we can keep the tree we build after making the list and still free this active set.*/
    *NumActiveBlackHoles = tw_bh->WorkSetSize;
    if(totbh > 0) {
        *ActiveBlackHoles = mymalloc2("activeBH", int, tw_bh->WorkSetSize);
        memcpy(*ActiveBlackHoles, tw_bh->WorkSet, tw_bh->WorkSetSize * sizeof(int));
    }
    myfree(tw_bh->WorkSet);
    return totbh;
}

/* Pointer to the global time bin manager. Hacky but will go away once we port to the C++ tree.*/
static TimeBinMgr * globalTimeBinMgr;

void
blackhole(const ActiveParticles * act, double atime, Cosmology * CP, ForceTree * tree, ForceTreeHmax * gasTree, DomainDecomp * ddecomp, DriftKickTimes * times, TimeBinMgr * timebinmgr, RandTable * rnd, const struct UnitSystem units, FILE * FdBlackHoles, FILE * FdBlackholeDetails, size_t * bhdetailswritten)
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

    globalTimeBinMgr = timebinmgr;
    /* Types used in treewalks:
     * accretion uses: gas + black holes (to flag mergers).
     * feedback uses: gas + black holes (to flag mergers).
     */
    if(!tree->tree_allocated_flag || !(tree->mask & GASMASK) ||  !(tree->mask & BHMASK) )
        endrun(5, "Blackhole called with bad tree allocated %d mask %d want %d\n", tree->tree_allocated_flag, tree->mask, GASMASK | BHMASK);
    if(!gasTree)
        endrun(5, "Blackhole called without a gas tree for the accretion treewalk\n");

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

    struct kick_factor_data kf;
    init_kick_factor_data(&kf, times, timebinmgr, CP);

    struct BHPriv priv[1] = {0};
    priv->units = units;
    priv->rnd = rnd;
    /*************************************************************************/
    priv->atime = atime;
    priv->a3inv = 1./(atime * atime * atime);
    priv->hubble = hubble_function(CP, atime);
    priv->CP = CP;
    priv->kf = &kf;

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

    /* Need hmax for the symmetric BH merger treewalk*/
    if(!tree->hmax_computed_flag)
        force_tree_calc_moments(tree, ddecomp);

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
    blackhole_feedback(ActiveBlackHoles, NumActiveBlackHoles, tree, priv);

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

typedef struct {
    TreeWalkQueryBase base;
    MyFloat Hsml;
    MyFloat Mtrack;
    MyFloat BH_Mass;
    MyIDType ID;
    MyFloat Density;
    MyFloat FeedbackEnergy;
    MyFloat FeedbackWeightSum;
    MyFloat KEFeedbackEnergy;
    int FdbkChannel; /* 0 thermal, 1 kinetic */
    int alignment; /* Ensure alignment*/
} TreeWalkQueryBHFeedback;

typedef struct {
    TreeWalkResultBase base;
    MyFloat Mass; /* the accreted Mdyn */
    MyFloat AccretedMomentum[3];
    MyFloat BH_Mass;
    int BH_CountProgs;
    int BH_minTimeBin;
} TreeWalkResultBHFeedback;

typedef struct {
    TreeWalkNgbIterBase base;
    DensityKernel kernel;
} TreeWalkNgbIterBHFeedback;

/* Adds the injected black hole energy to an internal energy and caps it at a maximum temperature*/
static double
add_injected_BH_energy(double unew, double injected_BH_energy, double mass, double uu_in_cgs)
{
    unew += injected_BH_energy / mass;
    const double u_to_temp_fac = (4 / (8 - 5 * (1 - HYDROGEN_MASSFRAC))) * PROTONMASS / BOLTZMANN * GAMMA_MINUS1 * uu_in_cgs;
    /* Cap temperature*/
    if(unew > 5.0e8 / u_to_temp_fac)
        unew = 5.0e8 / u_to_temp_fac;

    return unew;
}

static int
get_random_dir(int i, double dir[3], const RandTable * const rnd)
{
    double theta = acos(2 * get_random_number(Part[i].ID + 3, rnd) - 1);
    double phi = 2 * M_PI * get_random_number(Part[i].ID + 4, rnd);

    dir[0] = sin(theta) * cos(phi);
    dir[1] = sin(theta) * sin(phi);
    dir[2] = cos(theta);
    return 0;
}

/**
 * perform blackhole swallow / merger;
 */
static void
blackhole_feedback_ngbiter(TreeWalkQueryBHFeedback * I,
        TreeWalkResultBHFeedback * O,
        TreeWalkNgbIterBHFeedback * iter,
        LocalTreeWalk * lv)
{

    if(iter->base.other == -1) {
        O->BH_minTimeBin = TIMEBINS;
        iter->base.mask = GASMASK + BHMASK;
        iter->base.Hsml = I->Hsml;
        /* Needs to be symmetric because the BH mergers should be symmetric*/
        iter->base.symmetric = NGB_TREEFIND_SYMMETRIC;
        density_kernel_init(&iter->kernel, I->Hsml, GetDensityKernelType());
        return;
    }

    int other = iter->base.other;
    double r2 = iter->base.r2;
    double r = iter->base.r;
    /* Exclude self interaction */

    if(Part[other].ID == I->ID) return;

     /* BH does not accrete wind */
    if(winds_is_particle_decoupled(other))
        return;


    /* we have a black hole merger! */
    int PI = Part[other].PI;
    if(Part[other].Type == 5 && BH_GET_PRIV(lv->tw)->BH_SwallowID[PI] != 0)
    {
        if(BH_GET_PRIV(lv->tw)->BH_SwallowID[PI] != I->ID + 1) return;

        /* Swallow the particle*/
        /* A note on Swallowed vs SwallowID: black hole particles which have been completely swallowed
         * (ie, their mass has been added to another particle) have Swallowed = 1.
         * These particles are ignored in future tree walks. SwallowID is set to the swallowing particle.
         */
        BHP(other).SwallowID = BH_GET_PRIV(lv->tw)->BH_SwallowID[PI] - 1;
        BHP(other).SwallowTime = BH_GET_PRIV(lv->tw)->atime;
        Part[other].Swallowed = 1;
        /* Set encounter to zero when we merge*/
        BHP(other).encounter = 0;
        O->BH_CountProgs += BHP(other).CountProgs;
        O->BH_Mass += (BHP(other).Mass);

        /* Active mass tracer for the other BH*/
        double othermass = Part[other].Mass;
        if (blackhole_params.SeedBHDynMass>0 && I->Mtrack>0){
            /* If the other BH has not yet reached the dynamical mass, we should accrete the Mtrack*/
            if(BHP(other).Mtrack < blackhole_params.SeedBHDynMass)
                othermass = BHP(other).Mtrack;
        }
        /* Add the accreted mass tracer to the total accreted Mass.
         * We will decide in postprocess if it goes into Mtrack or Mass*/
        O->Mass += othermass;

        MyFloat VelPred[3];
        DM_VelPred(other, VelPred, BH_GET_PRIV(lv->tw)->kf);
        /* Conserve momentum during accretion*/
        int d;
        for(d = 0; d < 3; d++)
            O->AccretedMomentum[d] += (othermass * VelPred[d]);

        if(BHP(other).SwallowTime < BH_GET_PRIV(lv->tw)->atime)
            endrun(2, "Encountered BH %i swallowed at earlier time %g\n", other, BHP(other).SwallowTime);

        int tid = omp_get_thread_num();
        BH_GET_PRIV(lv->tw)->N_BH_swallowed[tid]++;

    }

    MyIDType * SPH_SwallowID = BH_GET_PRIV(lv->tw)->SPH_SwallowID;

    /* perform thermal or kinetic feedback into non-swallowed particles. */
    if(Part[other].Type == 0 && SPH_SwallowID[Part[other].PI] == 0 &&
        (r2 < iter->kernel.HH))
    {
        /* For accretion stability, set the BH timestep to the smallest gas timestep.
         * Ignore swallowed or wind gas particles.*/
        if (O->BH_minTimeBin > Part[other].TimeBinHydro)
            O->BH_minTimeBin = Part[other].TimeBinHydro;

        double u = r * iter->kernel.Hinv;
        double mass_j = Part[other].Mass;
        double wk = density_kernel_wk(&iter->kernel, u);

        /* thermal feedback */
        if(I->FeedbackWeightSum > 0 && I->FeedbackEnergy > 0 && I->FdbkChannel == 0 && Part[other].Mass > 0){
            const double injected_BH = I->FeedbackEnergy * mass_j * wk / I->FeedbackWeightSum;
            /* Set a flag for star-forming particles:
                * we want these to cool to the EEQOS via
                * tcool rather than trelax.*/
            if(sfreff_on_eeqos(&SPHP(other), BH_GET_PRIV(lv->tw)->a3inv)) {
                /* We cannot atomically set a bitfield.
                 * This flag is never read in this thread loop, and we are careful not to
                 * do this with a swallowed particle (as this can race with IsGarbage being set).
                 * So lack of atomicity is (I think) not a problem.*/
                //#pragma omp atomic write
                Part[other].BHHeated = 1;
            }
            const double enttou = pow(SPHP(other).Density * BH_GET_PRIV(lv->tw)->a3inv, GAMMA_MINUS1) / GAMMA_MINUS1;
            const double uu_in_cgs = BH_GET_PRIV(lv->tw)->units.UnitEnergy_in_cgs / BH_GET_PRIV(lv->tw)->units.UnitMass_in_g;

            double entold, entnew;
            double * entptr = &(SPHP(other).Entropy);
            #pragma omp atomic read
            entold = *entptr;
            do {
                entnew = add_injected_BH_energy(entold * enttou, injected_BH, Part[other].Mass, uu_in_cgs) / enttou;
                /* Swap in the new gas entropy only if the old one hasn't changed.*/
            } while(!__atomic_compare_exchange(entptr, &entold, &entnew, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED));
        }

        /* kinetic feedback */
        if(I->KEFeedbackEnergy > 0 && I->FdbkChannel == 1 && I->Density > 0){
            /* Kick the gas particle*/
            double dvel = sqrt(2 * I->KEFeedbackEnergy * wk / I->Density);
            double dir[3];
            get_random_dir(other, dir, BH_GET_PRIV(lv->tw)->rnd);
            int j;
            for(j = 0; j < 3; j++){
                #pragma omp atomic update
                Part[other].Vel[j] += (dvel*dir[j]);
            }
        }
    }

    /* Swallowing a gas */
    /* This will only be true on one thread so we do not need a lock here*/
    /* Note that it will rarely happen that gas is swallowed by a BH which is itself swallowed.
     * In that case we do not swallow this particle: all swallowing changes before this are temporary*/
    if(Part[other].Type == 0 && SPH_SwallowID[Part[other].PI] == I->ID+1)
    {
        O->Mass += Part[other].Mass;
        MyFloat VelPred[3];
        SPH_VelPred(other, VelPred, BH_GET_PRIV(lv->tw)->kf);
        /* Conserve momentum during accretion*/
        int d;
        for(d = 0; d < 3; d++)
            O->AccretedMomentum[d] += (Part[other].Mass * VelPred[d]);

        slots_mark_garbage(other, PartManager, SlotsManager);

        int tid = omp_get_thread_num();
        BH_GET_PRIV(lv->tw)->N_sph_swallowed[tid]++;
    }
}
static int
blackhole_feedback_haswork(int n, TreeWalk * tw)
{
    /*Black hole not being swallowed*/
    int PI = Part[n].PI;
    return (Part[n].Type == 5) && (!Part[n].Swallowed) && (BH_GET_PRIV(tw)->BH_SwallowID[PI] == 0);
}

static void
blackhole_feedback_copy(int i, TreeWalkQueryBHFeedback * I, TreeWalk * tw)
{
    I->Hsml = Part[i].Hsml;
    I->BH_Mass = BHP(i).Mass;
    I->ID = Part[i].ID;
    I->Density = BHP(i).Density;
    int PI = Part[i].PI;

    I->FeedbackWeightSum = BH_GET_PRIV(tw)->BH_FeedbackWeightSum[PI];
    I->FdbkChannel = 0; /* thermal feedback mode */

    double dtime = globalTimeBinMgr->get_dloga_for_bin(Part[i].TimeBinHydro, Part[i].Ti_drift) / BH_GET_PRIV(tw)->hubble;

    I->FeedbackEnergy = blackhole_params.BlackHoleFeedbackFactor * 0.1 * BHP(i).Mdot * dtime *
                pow(LIGHTCGS / BH_GET_PRIV(tw)->units.UnitVelocity_in_cm_per_s, 2);
    I->KEFeedbackEnergy = 0;
    if (blackhole_params.BlackHoleKineticOn == 1 && BH_GET_PRIV(tw)->KEflag[PI] > 0){
        I->FdbkChannel = 1; /* kinetic feedback mode, (no thermal feedback for this timestep) */
        /* KEflag = 1: KEFeedbackEnergy is accumulating; KEflag = 2: KEFeedbackEnergy is released. */
        if (BH_GET_PRIV(tw)->KEflag[PI] == 2){
            I->KEFeedbackEnergy = BHP(i).KineticFdbkEnergy;
        }
    }
}

static void
blackhole_feedback_reduce(int place, TreeWalkResultBHFeedback * remote, enum TreeWalkReduceMode mode, TreeWalk * tw)
{
    int k;
    int PI = Part[place].PI;

    TREEWALK_REDUCE(BH_GET_PRIV(tw)->BH_accreted_Mass[PI], remote->Mass);
    TREEWALK_REDUCE(BH_GET_PRIV(tw)->BH_accreted_BHMass[PI], remote->BH_Mass);
    for(k = 0; k < 3; k++) {
        TREEWALK_REDUCE(BH_GET_PRIV(tw)->BH_accreted_momentum[PI][k], remote->AccretedMomentum[k]);
    }
    if (mode == TREEWALK_PRIMARY || BHP(place).minTimeBin > remote->BH_minTimeBin) {
        BHP(place).minTimeBin = remote->BH_minTimeBin;
    }
    TREEWALK_REDUCE(BHP(place).CountProgs, remote->BH_CountProgs);
}

static void
blackhole_feedback_postprocess(int n, TreeWalk * tw)
{
    const int PI = Part[n].PI;
    if(BH_GET_PRIV(tw)->BH_accreted_BHMass[PI] > 0){
       BHP(n).Mass += BH_GET_PRIV(tw)->BH_accreted_BHMass[PI];
    }
    if(BH_GET_PRIV(tw)->BH_accreted_Mass[PI] > 0)
    {
        /* velocity feedback due to accretion; momentum conservation.
         * This does nothing with repositioning on.*/
        const MyFloat accmass = BH_GET_PRIV(tw)->BH_accreted_Mass[PI];
        int k;
        /* Need to add the momentum from Mtrack as well*/
        for(k = 0; k < 3; k++)
            Part[n].Vel[k] = (Part[n].Vel[k] * Part[n].Mass + BH_GET_PRIV(tw)->BH_accreted_momentum[PI][k]) / (Part[n].Mass + accmass);
        /* Add the mass to Mtrack if there is room*/
        const double SeedBHDynMass = blackhole_params.SeedBHDynMass;
        if(SeedBHDynMass > 0 && BHP(n).Mtrack + accmass < SeedBHDynMass) {
            /* Still seed mass regime*/
            BHP(n).Mtrack += accmass;
        } else if(BHP(n).Mtrack < SeedBHDynMass) {
            /* Transitioning to regular BH */
            Part[n].Mass = BHP(n).Mtrack + accmass;
            BHP(n).Mtrack = SeedBHDynMass;
        }
        else {
            /* Already regular BH, add accretion to regular mass*/
            Part[n].Mass += accmass;
        }
    }

    /* Reset KineticFdbkEnerg to 0 after released */
    if(BH_GET_PRIV(tw)->KEflag[PI] == 2){
        BHP(n).KineticFdbkEnergy = 0;
    }
}

/* Do the black hole feedback tree walk. Tree needs to have gas and BH.*/
static void
blackhole_feedback(int * ActiveBlackHoles, int64_t NumActiveBlackHoles, ForceTree * tree, struct BHPriv * priv)
{
    if(!(tree->mask & GASMASK) || !(tree->mask & BHMASK))
        endrun(5, "Error: BH tree types GAS: %d BH %d\n", tree->mask & GASMASK, tree->mask & BHMASK);

    TreeWalk tw_feedback[1] = {{0}};
    tw_feedback->ev_label = "BH_FEEDBACK";
    tw_feedback->visit = (TreeWalkVisitFunction) treewalk_visit_ngbiter;
    tw_feedback->ngbiter_type_elsize = sizeof(TreeWalkNgbIterBHFeedback);
    tw_feedback->ngbiter = (TreeWalkNgbIterFunction) blackhole_feedback_ngbiter;
    tw_feedback->haswork = blackhole_feedback_haswork;
    tw_feedback->fill = (TreeWalkFillQueryFunction) blackhole_feedback_copy;
    tw_feedback->postprocess = (TreeWalkProcessFunction) blackhole_feedback_postprocess;
    tw_feedback->reduce = (TreeWalkReduceResultFunction) blackhole_feedback_reduce;
    tw_feedback->query_type_elsize = sizeof(TreeWalkQueryBHFeedback);
    tw_feedback->result_type_elsize = sizeof(TreeWalkResultBHFeedback);
    tw_feedback->tree = tree;
    tw_feedback->priv = priv;

    /* Ionization counters*/
    priv[0].N_sph_swallowed = ta_malloc("n_sph_swallowed", int64_t, omp_get_max_threads());
    priv[0].N_BH_swallowed = ta_malloc("n_BH_swallowed", int64_t, omp_get_max_threads());
    memset(priv[0].N_sph_swallowed, 0, sizeof(int64_t) * omp_get_max_threads());
    memset(priv[0].N_BH_swallowed, 0, sizeof(int64_t) * omp_get_max_threads());

    treewalk_run(tw_feedback, ActiveBlackHoles, NumActiveBlackHoles);

    int i;
    int64_t Ntot_gas_swallowed, Ntot_BH_swallowed;
    int64_t N_sph_swallowed = 0, N_BH_swallowed = 0;
    for(i = 0; i < omp_get_max_threads(); i++) {
        N_sph_swallowed += priv[0].N_sph_swallowed[i];
        N_BH_swallowed += priv[0].N_BH_swallowed[i];
    }
    ta_free(priv[0].N_BH_swallowed);
    ta_free(priv[0].N_sph_swallowed);

    MPI_Reduce(&N_sph_swallowed, &Ntot_gas_swallowed, 1, MPI_INT64, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&N_BH_swallowed, &Ntot_BH_swallowed, 1, MPI_INT64, MPI_SUM, 0, MPI_COMM_WORLD);

    message(0, "Accretion done: %ld gas particles swallowed, %ld BH particles swallowed\n", Ntot_gas_swallowed, Ntot_BH_swallowed);
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
