#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#include "physconst.h"
#include "gravity.h"
#include "densitykernel.h"
#include "treewalk2.h"
#include "slotsmanager.h"
#include "blackhole.h"
#include "timestep.h"
#include "density2.h"
#include "types.h"
#include "winds.h"
#include "walltime.h"
#include "bhdynfric.h"
#include "utils/endrun.h"
#include "utils/mymalloc.h"

/*! \file bhaccretion2.cpp
 *  \brief The black hole accretion treewalk. This computes the accretion rate onto each black hole,
 *  and marks the gas particles and black holes which may be swallowed in the feedback treewalk.
 */

static double blackhole_soundspeed(double entropy, double rho, const double atime) {
    /* rho is comoving !*/
    if(rho <= 0)
        return 0;
    double cs = sqrt(GAMMA * entropy * pow(rho, GAMMA_MINUS1));

    cs *= pow(atime, -1.5 * GAMMA_MINUS1);

    return cs;
}

/* check if two BHs are gravitationally bounded, input dv, da, dx in code unit */
/* same as Bellovary2011, Tremmel2017 */
static int
check_grav_bound(double dx[3], double dv[3], double da[3], const double atime)
{
    int j;
    double KE = 0;
    double PE = 0;

    for(j = 0; j < 3; j++){
        KE += 0.5 * pow(dv[j], 2);
        PE += da[j] * dx[j];
    }

    KE /= (atime * atime); /* convert to proper velocity */
    PE /= atime; /* convert to proper unit */

    /* The gravitationally bound condition is PE + KE < 0.
     * Still merge if it is marginally bound so that we merge
     * particles at zero distance and velocity from each other.*/
    return (PE + KE <= 0);
}

/* Inputs to the accretion treewalk, which are not changed by it.*/
class BHAccretionPriv : public ParamTypeBase {
    public:
    /* Time factors*/
    double atime;
    KickFactorData kf;
    const RandTable * rnd;
    sph_particle_data * SphParts;
    bh_particle_data * BhParts;
    /* Parameters of the model*/
    struct BlackholeParams params;
    enum DensityKernelType KernelType;
    /* Black holes closer than this may merge.*/
    double MergeDistance;
    bool RepositionEnabled;

    BHAccretionPriv(const double BoxSize, const double i_atime, const KickFactorData& i_kf, const RandTable * const i_rnd, slots_manager_type * slotsmanager):
        ParamTypeBase(BoxSize), atime(i_atime), kf(i_kf), rnd(i_rnd),
        SphParts(slotsmanager->sph_slot()), BhParts(slotsmanager->bh_slot()),
        params(blackhole_params), KernelType(GetDensityKernelType()),
        /* We use 2 times the gravitational softening as the merging criterion.*/
        MergeDistance(2*FORCE_SOFTENING()/2.8), RepositionEnabled(BHGetRepositionEnabled() == 1)
    {}
};

class BHAccretionQuery : public TreeWalkQueryBase<BHAccretionPriv>
{
    public:
    MyFloat Density;
    MyFloat Hsml;
    MyFloat Mass;
    MyFloat BH_Mass;
    MyFloat Vel[3];
    MyFloat Accel[3];
    MyIDType ID;
    MyFloat Mtrack;

    BHAccretionQuery(const particle_data& particle, const int * const i_NodeList, const BHAccretionPriv& priv):
    TreeWalkQueryBase<BHAccretionPriv>(particle, i_NodeList, priv), Hsml(particle.Hsml), Mass(particle.Mass), ID(particle.ID)
    {
        const bh_particle_data& bh = priv.BhParts[particle.PI];
        for(int k = 0; k < 3; k++)
        {
            Vel[k] = particle.Vel[k];
            Accel[k] = particle.FullTreeGravAccel[k] + particle.GravPM[k] + bh.DFAccel[k];
        }
        BH_Mass = bh.Mass;
        Density = bh.Density;
        Mtrack = bh.Mtrack;
    }
};

/* The outputs of the accretion treewalk.
 * The arrays are owned by the BHPriv structure, as they are shared with the feedback treewalk.*/
class BHAccretionOutput {
    public:
    /* Marks particles which may be swallowed. These are indexed by the slot of the particle
     * being swallowed, not the black hole doing the swallowing. They are thus stored on the
     * rank doing the visiting and are written from ngbiter.*/
    MyIDType * SPH_SwallowID;
    MyIDType * BH_SwallowID;
    /* These are indexed by the black hole slot of the walking black hole.*/
    MyFloat * BH_FeedbackWeightSum;
    MyFloat * BH_Entropy;
    MyFloat (*BH_SurroundingGasVel)[3];
    MyFloat * NumDM;
    MyFloat * MgasEnc;
    int * KEflag;
    bh_particle_data * BhParts;
    /* For the postprocess function*/
    double a3inv;
    double hubble;
    struct UnitSystem units;
    Cosmology * CP;
    TimeBinMgr * timebinmgr;

    BHAccretionOutput(struct BHPriv * bhpriv, TimeBinMgr * i_timebinmgr, slots_manager_type * slotsmanager):
        SPH_SwallowID(bhpriv->SPH_SwallowID), BH_SwallowID(bhpriv->BH_SwallowID),
        BH_FeedbackWeightSum(bhpriv->BH_FeedbackWeightSum), BH_Entropy(bhpriv->BH_Entropy),
        BH_SurroundingGasVel(bhpriv->BH_SurroundingGasVel), NumDM(bhpriv->NumDM), MgasEnc(bhpriv->MgasEnc),
        KEflag(bhpriv->KEflag), BhParts(slotsmanager->bh_slot()),
        a3inv(bhpriv->a3inv), hubble(bhpriv->hubble), units(bhpriv->units), CP(bhpriv->CP), timebinmgr(i_timebinmgr)
    {}

    /* Compute the accretion rate and the drag from the kernel sums, and decide whether there is kinetic feedback.*/
    void
    postprocess(const int i, particle_data * const parts, const BHAccretionPriv * priv)
    {
        const struct BlackholeParams& params = priv->params;
        const particle_data& part = parts[i];
        const int PI = part.PI;
        bh_particle_data& bh = BhParts[PI];
        double mdot = 0;    /* if no accretion model is enabled, we have mdot=0 */
        /* Note: we take here a radiative efficiency of 0.1 for Eddington accretion */
        const double meddington = (4 * M_PI * GRAVITY * LIGHTCGS * PROTONMASS / (0.1 * LIGHTCGS * LIGHTCGS * THOMPSON)) * bh.Mass
                * units.UnitTime_in_s / CP->HubbleParam;

        if(bh.Density > 0)
        {
            BH_Entropy[PI] /= bh.Density;
            for(int k = 0; k < 3; k++)
                BH_SurroundingGasVel[PI][k] /= bh.Density;

            double bhvel = 0;
            for(int k = 0; k < 3; k++)
                bhvel += pow(part.Vel[k] - BH_SurroundingGasVel[PI][k], 2);

            bhvel = sqrt(bhvel);
            bhvel /= priv->atime;
            double rho = bh.Density;
            double rho_proper = rho * a3inv;

            double soundspeed = blackhole_soundspeed(BH_Entropy[PI], rho, priv->atime);

            double norm = pow((pow(soundspeed, 2) + pow(bhvel, 2)), 1.5);

            if(norm > 0)
                mdot = 4. * M_PI * params.BlackHoleAccretionFactor * CP->GravInternal * CP->GravInternal *
                    bh.Mass * bh.Mass * rho_proper / norm;
        }

        if(params.BlackHoleEddingtonFactor > 0.0 &&
            mdot > params.BlackHoleEddingtonFactor * meddington) {
            mdot = params.BlackHoleEddingtonFactor * meddington;
        }
        bh.Mdot = mdot;

        double dtime = timebinmgr->get_dloga_for_bin(part.TimeBinHydro, part.Ti_drift) / hubble;

        bh.Mass += bh.Mdot * dtime;

        /*************************************************************************/

        if(params.BH_DRAG > 0){
            /* a_BH = (v_gas - v_BH) Mdot/M_BH                                   */
            /* motivated by BH gaining momentum from the accreted gas            */
            /*c.f.section 3.2,in http://www.tapir.caltech.edu/~phopkins/public/notes_blackholes.pdf */
            double fac = 0;
            if (params.BH_DRAG == 1) fac = bh.Mdot/part.Mass;
            if (params.BH_DRAG == 2) fac = params.BlackHoleEddingtonFactor * meddington/bh.Mass;
            fac *= priv->atime; /* dv = acc * kick_fac = acc * a^{-1}dt, therefore acc = a*dv/dt  */
            for(int k = 0; k < 3; k++) {
                bh.DragAccel[k] = -(part.Vel[k] - BH_SurroundingGasVel[PI][k])*fac;
            }
        }
        else{
            for(int k = 0; k < 3; k++){
                bh.DragAccel[k] = 0;
            }
        }
        /*************************************************************************/

        if(params.BlackHoleKineticOn == 1){
            /* accumulate kenetic feedback energy by dE = epsilon x mdot x c^2 */
            /* epsilon = Min(rho_BH/(BHKE_EffRhoFactor*rho_sfr),BHKE_EffCap)   */
            /* KE is released when exceeding injection energy threshold        */
            KEflag[PI] = 0;
            double Edd_ratio = bh.Mdot/meddington;
            double lam_thresh = params.BHKE_EddingtonThrFactor;
            double x = params.BHKE_EddingtonMFactor * pow(bh.Mass/params.BHKE_EddingtonMPivot, params.BHKE_EddingtonMIndex);
            if (lam_thresh > x)
                lam_thresh = x;
            if (Edd_ratio < lam_thresh){
                /* mark this timestep is accumulating KE feedback energy */
                KEflag[PI] = 1;
                const double rho_crit_baryon = CP->OmegaBaryon * 3 * pow(CP->Hubble, 2) / (8 * M_PI * CP->GravInternal);
                const double rho_sfr = params.BHKE_SfrCritOverDensity * rho_crit_baryon;
                double epsilon = (bh.Density/rho_sfr)/params.BHKE_EffRhoFactor;
                if (epsilon > params.BHKE_EffCap){
                    epsilon = params.BHKE_EffCap;
                }

                bh.KineticFdbkEnergy += epsilon * (bh.Mdot * dtime * pow(LIGHTCGS / units.UnitVelocity_in_cm_per_s, 2));
            }
            /* decide whether to release KineticFdbkEnergy*/
            double KE_thresh = 0.5 * bh.VDisp * bh.VDisp * MgasEnc[PI];
            KE_thresh *= params.BHKE_InjEnergyThr;

            if (bh.VDisp > 0 && bh.KineticFdbkEnergy > KE_thresh){
                /* mark KineticFdbkEnergy is ready to be released in the feedback treewalk */
                KEflag[PI] = 2;
            }
        }
    }
};

class BHAccretionResult : public TreeWalkResultBase<BHAccretionQuery, BHAccretionOutput> {
    public:
    /* marks the event when two BHs encounter each other */
    int encounter = 0;
    MyFloat FeedbackWeightSum = 0;
    MyFloat SmoothedEntropy = 0;
    MyFloat GasVel[3] = {0,0,0};
    /* used for AGN kinetic feedback */
    MyFloat NumDM = 0;
    MyFloat MgasEnc = 0;

    BHAccretionResult(const BHAccretionQuery& query):
        TreeWalkResultBase<BHAccretionQuery, BHAccretionOutput>(query) {}

    template<TreeWalkReduceMode mode>
    void reduce(int place, const BHAccretionOutput * output, struct particle_data * const parts)
    {
        const int PI = parts[place].PI;
        bh_particle_data& bh = output->BhParts[PI];

        /* Set encounter to true if it is true on any remote*/
        if (mode == TREEWALK_PRIMARY || bh.encounter < encounter) {
            bh.encounter = encounter;
        }

        TREEWALK_REDUCE(output->BH_FeedbackWeightSum[PI], FeedbackWeightSum);
        TREEWALK_REDUCE(output->BH_Entropy[PI], SmoothedEntropy);
        for (int k = 0; k < 3; k++){
            TREEWALK_REDUCE(output->BH_SurroundingGasVel[PI][k], GasVel[k]);
        }
        TREEWALK_REDUCE(output->NumDM[PI], NumDM);
        TREEWALK_REDUCE(output->MgasEnc[PI], MgasEnc);
    }
};

/* This treewalk marks all black holes and gas which can be swallowed with a SwallowID of a potential swallower,
 * and computes the kernel sums of the gas around the black hole.
 * The treewalk is symmetric, for the black hole mergers.*/
class BHAccretionLocalTreeWalk: public LocalNgbTreeWalk<BHAccretionLocalTreeWalk, BHAccretionQuery, BHAccretionResult, BHAccretionPriv, HmaxNODE, NGB_TREEFIND_SYMMETRIC, GASMASK + BHMASK>
{
    public:
    DensityKernel kernel;
    /* The swallow markers for the particles on the rank doing the visiting. Shared between threads, so updated atomically.*/
    MyIDType * const SPH_SwallowID;
    MyIDType * const BH_SwallowID;

    BHAccretionLocalTreeWalk(const HmaxNODE * const Nodes, const BHAccretionQuery& input, const enum DensityKernelType KernelType, MyIDType * const i_SPH_SwallowID, MyIDType * const i_BH_SwallowID):
    LocalNgbTreeWalk<BHAccretionLocalTreeWalk, BHAccretionQuery, BHAccretionResult, BHAccretionPriv, HmaxNODE, NGB_TREEFIND_SYMMETRIC, GASMASK + BHMASK>(Nodes, input),
    SPH_SwallowID(i_SPH_SwallowID), BH_SwallowID(i_BH_SwallowID)
    {
        density_kernel_init(&kernel, input.Hsml, KernelType);
    }

    void ngbiter(BHAccretionQuery& input, const particle_data& particle, BHAccretionResult * output, const BHAccretionPriv& priv)
    {
        double dist[3];
        const double r2 = this->get_distance(input, particle, priv.BoxSize, dist);

        /* Symmetric search: the particles must be within the larger of the two smoothing lengths.*/
        const double hsearch = DMAX(particle.Hsml, input.Hsml);
        if(r2 > hsearch * hsearch)
            return;
        const double r = sqrt(r2);

        if(particle.Mass < 0) return;

        /* BH does not accrete wind */
        if(particle.Type == 0 && winds_is_particle_decoupled(&priv.SphParts[particle.PI])) return;

        /* Accretion / merger doesn't do self interaction */
        if(particle.ID == input.ID) return;

        /* we have a black hole merger. Now we use 2 times GravitationalSoftening as merging criteria.
         * Note there is another condition: they must also be closer than the SPH smoothing length,
         * as enforced by the tree search above. */
        if(particle.Type == 5 && r < priv.MergeDistance)
        {
            output->encounter = 1; // mark the event when two BHs encounter each other

            int flag = 0; // the flag for BH merge

            if(priv.RepositionEnabled) // directly merge if reposition is enabled
                flag = 1;
            if(priv.params.MergeGravBound == 0)
                flag = 1;
            /* apply Grav Bound check only when Reposition is disabled, otherwise BHs would be repositioned to the same location but not merge */
            if(priv.params.MergeGravBound == 1 && !priv.RepositionEnabled){
                double dx[3];
                double dv[3];
                double da[3];
                MyFloat VelPred[3];
                priv.kf.DM_VelPred(particle, VelPred);
                for(int d = 0; d < 3; d++){
                    dx[d] = dist[d];
                    dv[d] = input.Vel[d] - VelPred[d];
                    /* we include long range PM force, short range force from the last long timestep and DF */
                    da[d] = (input.Accel[d] - particle.FullTreeGravAccel[d] - particle.GravPM[d] - priv.BhParts[particle.PI].DFAccel[d]);
                }
                flag = check_grav_bound(dx,dv,da, priv.atime);
            }

            /* Mark the BH via SwallowID.*/
            if(flag == 1)
            {
                MyIDType readid, newswallowid;

                MyIDType * swal = BH_SwallowID + particle.PI;
                #pragma omp atomic read
                readid = *swal;

                /* Here we mark the black hole as "ready to be swallowed" using the SwallowID.
                 * The actual swallowing is done in the feedback treewalk by setting Swallowed = 1
                 * and merging the masses.*/
                do {
                    /* Already marked, prefer to be swallowed by a bigger ID */
                    if(readid != 0 && readid < input.ID ) {
                        /* Already marked, prefer to be swallowed by a bigger ID */
                        newswallowid = input.ID + 1;
                    } else if(readid == 0 && (particle.ID < input.ID || !is_timebin_active(particle.TimeBinHydro, particle.Ti_drift))) {
                        /* Unmarked, the BH with bigger ID swallows This avoids two BHs trying to swallow each other.
                         * (in which case neither would merge). If only one is active, only one can swallow.*/
                        newswallowid = input.ID + 1;
                    }
                    else
                        break;
                /* Swap in the new id only if the old one hasn't changed:
                 * in principle an extension, but supported on at least clang >= 9, gcc >= 5 and icc >= 18.*/
                } while(!__atomic_compare_exchange_n(swal, &readid, newswallowid, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED));
            }
        }

        /* Everything else needs a gas particle inside the kernel of the black hole*/
        if(particle.Type != 0 || !(r2 < kernel.HH))
            return;

        const double u = r * kernel.Hinv;
        const double wk = density_kernel_wk(&kernel, u);
        const double mass_j = particle.Mass;
        const sph_particle_data& sph = priv.SphParts[particle.PI];

        output->SmoothedEntropy += (mass_j * wk * sph.Entropy);
        MyFloat VelPred[3];
        priv.kf.SPH_VelPred(particle, sph, VelPred);
        output->GasVel[0] += (mass_j * wk * VelPred[0]);
        output->GasVel[1] += (mass_j * wk * VelPred[1]);
        output->GasVel[2] += (mass_j * wk * VelPred[2]);
        /* here we have a gas particle; check for swallowing */

        /* compute accretion probability */
        double p = 0;

        MyFloat BHPartMass = input.Mass;
        /* If SeedBHDynMass is larger than gas paricle mass, we use Mtrack to do the gas accretion
         * when BHP.Mass < SeedBHDynMass. Mtrack is initialized as gas particle mass and is capped
         * at SeedBHDynMass. Mtrack traces the BHP.Mass by stochastically swallowing gas and
         * therefore ensures mass conservation.*/
        if(priv.params.SeedBHDynMass > 0 && input.Mtrack < priv.params.SeedBHDynMass)
            BHPartMass = input.Mtrack;

        /* This is an averaged Mdot, because Mdot increases BH_Mass but not Mass.
         * So if the total accretion is significantly above the dynamical mass,
         * a particle is swallowed. */
        if((input.BH_Mass - BHPartMass) > 0 && input.Density > 0)
            p = (input.BH_Mass - BHPartMass) * wk / input.Density;

        /* compute random number, uniform in [0,1] */
        const double w = get_random_number(particle.ID, priv.rnd);
        if(w < p)
        {
            MyIDType * swal = SPH_SwallowID + particle.PI;
            MyIDType readid, newswallowid;
            #pragma omp atomic read
            readid = *swal;
            do {
                /* Already marked, prefer to be swallowed by a bigger ID.
                 * Not marked, the SwallowID is 0 */
                if(readid < input.ID + 1) {
                    newswallowid = input.ID + 1;
                }
                else
                    break;
                /* Swap in the new id only if the old one hasn't changed*/
            } while(!__atomic_compare_exchange_n(swal, &readid, newswallowid, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED));
        }

        /* update the feedback weighting */
        output->FeedbackWeightSum += (mass_j * wk);

        /* collect info for sigmaDM and Menc for kinetic feedback */
        if(priv.params.BlackHoleKineticOn == 1)
            output->MgasEnc += mass_j;
    }
};

class BHAccretionTopTreeWalk: public TopTreeWalk<BHAccretionQuery, BHAccretionPriv, HmaxNODE, NGB_TREEFIND_SYMMETRIC> { using TopTreeWalk::TopTreeWalk; };

class BHAccretionTreeWalk: public TreeWalk<BHAccretionTreeWalk, BHAccretionQuery, BHAccretionResult, BHAccretionLocalTreeWalk, BHAccretionTopTreeWalk, BHAccretionPriv, BHAccretionOutput, ForceTreeHmax> {
    public:
    using TreeWalk::TreeWalk;

    /* Give each local walk the swallow markers, which are stored on the rank doing the visiting.*/
    BHAccretionLocalTreeWalk make_local_walk(const BHAccretionQuery& input)
    {
        return BHAccretionLocalTreeWalk(tree->get_nodes(), input, priv.KernelType, output->SPH_SwallowID, output->BH_SwallowID);
    }
};

/* The symmetric treewalk needs hmax to be valid for every gas particle and black hole in the tree.
 * A tree which has been through density() and the hydro already has it. One which was built afresh
 * does not include the active particles in hmax, so add them here.*/
static void
blackhole_ensure_hmax(ForceTreeHmax * tree, DomainDecomp * ddecomp)
{
    if(tree->hmax_computed_flag)
        return;
    const particle_data * const parts = PartManager->Base;
    #pragma omp parallel for
    for(int64_t i = 0; i < PartManager->NumPart; i++) {
        /* Only particles which were in the tree when it was built:
         * new particles since then are stars, and garbage and swallowed black holes are not added.*/
        if(parts[i].IsGarbage || (parts[i].Type != 0 && parts[i].Type != 5))
            continue;
        if(parts[i].Type == 5 && parts[i].Swallowed)
            continue;
        tree->update_tree_hmax_father(i, parts[i].Pos, parts[i].Hsml);
    }
    tree->force_tree_calc_hmax(ddecomp);
}

void
blackhole_accretion(int * ActiveBlackHoles, const int64_t NumActiveBlackHoles, ForceTreeHmax * tree, DomainDecomp * ddecomp, struct BHPriv * bhpriv, const KickFactorData& kf, TimeBinMgr * timebinmgr, MPI_Comm comm)
{
    blackhole_ensure_hmax(tree, ddecomp);

    BHAccretionPriv priv(PartManager->BoxSize, bhpriv->atime, kf, bhpriv->rnd, SlotsManager);
    BHAccretionOutput output(bhpriv, timebinmgr, SlotsManager);
    BHAccretionTreeWalk tw("BH_ACCRETION", tree, priv, &output);
    tw.run(ActiveBlackHoles, NumActiveBlackHoles, PartManager->Base, comm);
}
