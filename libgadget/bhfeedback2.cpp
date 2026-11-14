#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#include "physconst.h"
#include "densitykernel.h"
#include "treewalk2.h"
#include "slotsmanager.h"
#include "blackhole.h"
#include "density2.h"
#include "sfr_eff.h"
#include "types.h"
#include "winds.h"
#include "utils/endrun.h"
#include "utils/mymalloc.h"

/*! \file bhfeedback2.cpp
 *  \brief The black hole feedback treewalk. This swallows the gas particles and black holes
 *  marked by the accretion treewalk and deposits the feedback energy into the surrounding gas.
 */

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

static void
get_random_dir(const MyIDType ID, double dir[3], const RandTable * const rnd)
{
    double theta = acos(2 * get_random_number(ID + 3, rnd) - 1);
    double phi = 2 * M_PI * get_random_number(ID + 4, rnd);

    dir[0] = sin(theta) * cos(phi);
    dir[1] = sin(theta) * sin(phi);
    dir[2] = cos(theta);
}

/* Inputs to the feedback treewalk. The treewalk does not change these arrays.*/
class BHFeedbackPriv : public ParamTypeBase {
    public:
    /* Time factors*/
    double atime;
    double a3inv;
    double hubble;
    KickFactorData kf;
    TimeBinMgr * timebinmgr;
    struct UnitSystem units;
    const RandTable * rnd;
    /* The swallow markers set by the accretion treewalk. These are indexed by the slot of the
     * particle being swallowed and so are read on the rank doing the visiting.*/
    const MyIDType * SPH_SwallowID;
    const MyIDType * BH_SwallowID;
    /* Indexed by the slot of the walking black hole and used to make the query.*/
    const MyFloat * BH_FeedbackWeightSum;
    const int * KEflag;
    const bh_particle_data * BhParts;
    /* Parameters of the model*/
    struct BlackholeParams params;
    enum DensityKernelType KernelType;

    BHFeedbackPriv(const double BoxSize, const struct BHPriv * const bhpriv, const KickFactorData& i_kf, TimeBinMgr * i_timebinmgr, slots_manager_type * slotsmanager):
        ParamTypeBase(BoxSize), atime(bhpriv->atime), a3inv(bhpriv->a3inv), hubble(bhpriv->hubble), kf(i_kf), timebinmgr(i_timebinmgr),
        units(bhpriv->units), rnd(bhpriv->rnd),
        SPH_SwallowID(bhpriv->SPH_SwallowID), BH_SwallowID(bhpriv->BH_SwallowID),
        BH_FeedbackWeightSum(bhpriv->BH_FeedbackWeightSum), KEflag(bhpriv->KEflag), BhParts(slotsmanager->bh_slot()),
        params(blackhole_params), KernelType(GetDensityKernelType())
    {}
};

class BHFeedbackQuery : public TreeWalkQueryBase<BHFeedbackPriv>
{
    public:
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

    BHFeedbackQuery(const particle_data& particle, const int * const i_NodeList, const BHFeedbackPriv& priv):
    TreeWalkQueryBase<BHFeedbackPriv>(particle, i_NodeList, priv), Hsml(particle.Hsml), ID(particle.ID), alignment(0)
    {
        const int PI = particle.PI;
        const bh_particle_data& bh = priv.BhParts[PI];
        /* The old treewalk read Mtrack in ngbiter but never set it in the query, so it was uninitialised.*/
        Mtrack = bh.Mtrack;
        BH_Mass = bh.Mass;
        Density = bh.Density;

        FeedbackWeightSum = priv.BH_FeedbackWeightSum[PI];
        FdbkChannel = 0; /* thermal feedback mode */

        const double dtime = priv.timebinmgr->get_dloga_for_bin(particle.TimeBinHydro, particle.Ti_drift) / priv.hubble;

        FeedbackEnergy = priv.params.BlackHoleFeedbackFactor * 0.1 * bh.Mdot * dtime *
                    pow(LIGHTCGS / priv.units.UnitVelocity_in_cm_per_s, 2);
        KEFeedbackEnergy = 0;
        if (priv.params.BlackHoleKineticOn == 1 && priv.KEflag[PI] > 0){
            FdbkChannel = 1; /* kinetic feedback mode, (no thermal feedback for this timestep) */
            /* KEflag = 1: KEFeedbackEnergy is accumulating; KEflag = 2: KEFeedbackEnergy is released. */
            if (priv.KEflag[PI] == 2){
                KEFeedbackEnergy = bh.KineticFdbkEnergy;
            }
        }
    }
};

/* The outputs of the feedback treewalk.
 * The arrays are owned by the BHPriv structure, as they are written to the black hole details file.*/
class BHFeedbackOutput {
    public:
    /* These are indexed by the black hole slot of the walking black hole.*/
    MyFloat * BH_accreted_Mass;
    MyFloat * BH_accreted_BHMass;
    MyFloat (*BH_accreted_momentum)[3];
    const int * KEflag;
    bh_particle_data * BhParts;
    /* The rest is the storage on the rank doing the visiting: the particles which are swallowed or heated,
     * and counters for the swallowed particles, one for each thread.*/
    sph_particle_data * SphParts;
    particle_data * Parts;
    int64_t * N_sph_swallowed;
    int64_t * N_BH_swallowed;
    int nthreads;

    BHFeedbackOutput(struct BHPriv * bhpriv, particle_data * const i_Parts, slots_manager_type * slotsmanager):
        BH_accreted_Mass(bhpriv->BH_accreted_Mass), BH_accreted_BHMass(bhpriv->BH_accreted_BHMass),
        BH_accreted_momentum(bhpriv->BH_accreted_momentum), KEflag(bhpriv->KEflag),
        BhParts(slotsmanager->bh_slot()), SphParts(slotsmanager->sph_slot()), Parts(i_Parts), nthreads(omp_get_max_threads())
    {
        N_sph_swallowed = mymalloc("n_sph_swallowed", int64_t, nthreads);
        N_BH_swallowed = mymalloc("n_BH_swallowed", int64_t, nthreads);
        memset(N_sph_swallowed, 0, sizeof(int64_t) * nthreads);
        memset(N_BH_swallowed, 0, sizeof(int64_t) * nthreads);
    }

    ~BHFeedbackOutput()
    {
        myfree(N_BH_swallowed);
        myfree(N_sph_swallowed);
    }

    /* Total number of gas particles on this rank which were swallowed.*/
    int64_t tot_sph_swallowed() const
    {
        int64_t tot = 0;
        for(int i = 0; i < nthreads; i++)
            tot += N_sph_swallowed[i];
        return tot;
    }

    /* Total number of black holes on this rank which were swallowed.*/
    int64_t tot_BH_swallowed() const
    {
        int64_t tot = 0;
        for(int i = 0; i < nthreads; i++)
            tot += N_BH_swallowed[i];
        return tot;
    }

    /* Add the swallowed mass and momentum to the black hole.*/
    void
    postprocess(const int n, particle_data * const parts, const BHFeedbackPriv * priv)
    {
        particle_data& part = parts[n];
        const int PI = part.PI;
        bh_particle_data& bh = BhParts[PI];
        if(BH_accreted_BHMass[PI] > 0){
            bh.Mass += BH_accreted_BHMass[PI];
        }
        if(BH_accreted_Mass[PI] > 0)
        {
            /* velocity feedback due to accretion; momentum conservation.
             * This does nothing with repositioning on.*/
            const MyFloat accmass = BH_accreted_Mass[PI];
            /* Need to add the momentum from Mtrack as well*/
            for(int k = 0; k < 3; k++)
                part.Vel[k] = (part.Vel[k] * part.Mass + BH_accreted_momentum[PI][k]) / (part.Mass + accmass);
            /* Add the mass to Mtrack if there is room*/
            const double SeedBHDynMass = priv->params.SeedBHDynMass;
            if(SeedBHDynMass > 0 && bh.Mtrack + accmass < SeedBHDynMass) {
                /* Still seed mass regime*/
                bh.Mtrack += accmass;
            } else if(bh.Mtrack < SeedBHDynMass) {
                /* Transitioning to regular BH */
                part.Mass = bh.Mtrack + accmass;
                bh.Mtrack = SeedBHDynMass;
            }
            else {
                /* Already regular BH, add accretion to regular mass*/
                part.Mass += accmass;
            }
        }

        /* Reset KineticFdbkEnerg to 0 after released */
        if(KEflag[PI] == 2){
            bh.KineticFdbkEnergy = 0;
        }
    }
};

class BHFeedbackResult : public TreeWalkResultBase<BHFeedbackQuery, BHFeedbackOutput> {
    public:
    MyFloat Mass = 0; /* the accreted Mdyn */
    MyFloat AccretedMomentum[3] = {0,0,0};
    MyFloat BH_Mass = 0;
    int BH_CountProgs = 0;
    int BH_minTimeBin = TIMEBINS;

    BHFeedbackResult(const BHFeedbackQuery& query):
        TreeWalkResultBase<BHFeedbackQuery, BHFeedbackOutput>(query) {}

    template<TreeWalkReduceMode mode>
    void reduce(int place, const BHFeedbackOutput * output, struct particle_data * const parts)
    {
        const int PI = parts[place].PI;
        bh_particle_data& bh = output->BhParts[PI];

        TREEWALK_REDUCE(output->BH_accreted_Mass[PI], Mass);
        TREEWALK_REDUCE(output->BH_accreted_BHMass[PI], BH_Mass);
        for(int k = 0; k < 3; k++) {
            TREEWALK_REDUCE(output->BH_accreted_momentum[PI][k], AccretedMomentum[k]);
        }
        if (mode == TREEWALK_PRIMARY || bh.minTimeBin > BH_minTimeBin) {
            bh.minTimeBin = BH_minTimeBin;
        }
        /* Add the progenitors of the swallowed black holes to those this black hole already has.
         * This is not a TREEWALK_REDUCE: the primary must not overwrite the existing count.*/
        bh.CountProgs += BH_CountProgs;
    }
};

/* Performs the black hole swallowing and mergers, and the feedback.
 * This treewalk changes the particles it finds: they are on the rank doing the visiting.
 * The treewalk is symmetric because the black hole mergers should be symmetric.*/
class BHFeedbackLocalTreeWalk: public LocalNgbTreeWalk<BHFeedbackLocalTreeWalk, BHFeedbackQuery, BHFeedbackResult, BHFeedbackPriv, HmaxNODE, NGB_TREEFIND_SYMMETRIC, GASMASK + BHMASK>
{
    public:
    DensityKernel kernel;
    /* The particles on the rank doing the visiting, which this walk may change.
     * A particle may be found by several threads at once, so shared quantities are updated atomically.*/
    BHFeedbackOutput * const visited;
    /* Counters for this thread.*/
    int64_t * const N_sph_swallowed;
    int64_t * const N_BH_swallowed;

    BHFeedbackLocalTreeWalk(const HmaxNODE * const Nodes, const BHFeedbackQuery& input, const enum DensityKernelType KernelType, BHFeedbackOutput * const i_visited, const int tid):
    LocalNgbTreeWalk<BHFeedbackLocalTreeWalk, BHFeedbackQuery, BHFeedbackResult, BHFeedbackPriv, HmaxNODE, NGB_TREEFIND_SYMMETRIC, GASMASK + BHMASK>(Nodes, input),
    visited(i_visited), N_sph_swallowed(&i_visited->N_sph_swallowed[tid]), N_BH_swallowed(&i_visited->N_BH_swallowed[tid])
    {
        density_kernel_init(&kernel, input.Hsml, KernelType);
    }

    void ngbiter(BHFeedbackQuery& input, const particle_data& const_particle, BHFeedbackResult * output, const BHFeedbackPriv& priv)
    {
        double dist[3];
        const double r2 = this->get_distance(input, const_particle, priv.BoxSize, dist);

        /* Symmetric search: the particles must be within the larger of the two smoothing lengths.*/
        const double hsearch = DMAX(const_particle.Hsml, input.Hsml);
        if(r2 > hsearch * hsearch)
            return;

        /* Exclude self interaction */
        if(const_particle.ID == input.ID) return;

        /* The particle is in the particle table of this rank, and we will change it.*/
        const int other = &const_particle - visited->Parts;
        particle_data& particle = visited->Parts[other];

        /* BH does not accrete wind */
        if(particle.Type == 0 && winds_is_particle_decoupled(&visited->SphParts[particle.PI]))
            return;

        /* we have a black hole merger! */
        if(particle.Type == 5 && priv.BH_SwallowID[particle.PI] != 0)
        {
            if(priv.BH_SwallowID[particle.PI] != input.ID + 1) return;

            bh_particle_data& otherbh = visited->BhParts[particle.PI];
            /* Swallow the particle*/
            /* A note on Swallowed vs SwallowID: black hole particles which have been completely swallowed
             * (ie, their mass has been added to another particle) have Swallowed = 1.
             * These particles are ignored in future tree walks. SwallowID is set to the swallowing particle.
             */
            otherbh.SwallowID = priv.BH_SwallowID[particle.PI] - 1;
            otherbh.SwallowTime = priv.atime;
            particle.Swallowed = 1;
            /* Set encounter to zero when we merge*/
            otherbh.encounter = 0;
            output->BH_CountProgs += otherbh.CountProgs;
            output->BH_Mass += otherbh.Mass;

            /* Active mass tracer for the other BH*/
            double othermass = particle.Mass;
            if (priv.params.SeedBHDynMass > 0 && input.Mtrack > 0){
                /* If the other BH has not yet reached the dynamical mass, we should accrete the Mtrack*/
                if(otherbh.Mtrack < priv.params.SeedBHDynMass)
                    othermass = otherbh.Mtrack;
            }
            /* Add the accreted mass tracer to the total accreted Mass.
             * We will decide in postprocess if it goes into Mtrack or Mass*/
            output->Mass += othermass;

            MyFloat VelPred[3];
            priv.kf.DM_VelPred(particle, VelPred);
            /* Conserve momentum during accretion*/
            for(int d = 0; d < 3; d++)
                output->AccretedMomentum[d] += (othermass * VelPred[d]);

            (*N_BH_swallowed)++;
        }

        if(particle.Type != 0)
            return;

        sph_particle_data& sph = visited->SphParts[particle.PI];
        const MyIDType swallowid = priv.SPH_SwallowID[particle.PI];

        /* perform thermal or kinetic feedback into non-swallowed particles. */
        if(swallowid == 0 && (r2 < kernel.HH))
        {
            /* For accretion stability, set the BH timestep to the smallest gas timestep.
             * Ignore swallowed or wind gas particles.*/
            if (output->BH_minTimeBin > particle.TimeBinHydro)
                output->BH_minTimeBin = particle.TimeBinHydro;

            const double u = sqrt(r2) * kernel.Hinv;
            const double mass_j = particle.Mass;
            const double wk = density_kernel_wk(&kernel, u);

            /* thermal feedback */
            if(input.FeedbackWeightSum > 0 && input.FeedbackEnergy > 0 && input.FdbkChannel == 0 && particle.Mass > 0){
                const double injected_BH = input.FeedbackEnergy * mass_j * wk / input.FeedbackWeightSum;
                /* Set a flag for star-forming particles:
                    * we want these to cool to the EEQOS via
                    * tcool rather than trelax.*/
                if(sfreff_on_eeqos(&sph, priv.a3inv)) {
                    /* We cannot atomically set a bitfield.
                     * This flag is never read in this thread loop, and we are careful not to
                     * do this with a swallowed particle (as this can race with IsGarbage being set).
                     * So lack of atomicity is (I think) not a problem.*/
                    //#pragma omp atomic write
                    particle.BHHeated = 1;
                }
                const double enttou = pow(sph.Density * priv.a3inv, GAMMA_MINUS1) / GAMMA_MINUS1;
                const double uu_in_cgs = priv.units.UnitEnergy_in_cgs / priv.units.UnitMass_in_g;

                double entold, entnew;
                double * entptr = &(sph.Entropy);
                #pragma omp atomic read
                entold = *entptr;
                do {
                    entnew = add_injected_BH_energy(entold * enttou, injected_BH, particle.Mass, uu_in_cgs) / enttou;
                    /* Swap in the new gas entropy only if the old one hasn't changed.*/
                } while(!__atomic_compare_exchange(entptr, &entold, &entnew, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED));
            }

            /* kinetic feedback */
            if(input.KEFeedbackEnergy > 0 && input.FdbkChannel == 1 && input.Density > 0){
                /* Kick the gas particle*/
                const double dvel = sqrt(2 * input.KEFeedbackEnergy * wk / input.Density);
                double dir[3];
                get_random_dir(particle.ID, dir, priv.rnd);
                for(int j = 0; j < 3; j++){
                    #pragma omp atomic update
                    particle.Vel[j] += (dvel*dir[j]);
                }
            }
        }

        /* Swallowing a gas */
        /* This will only be true on one thread so we do not need a lock here*/
        /* Note that it will rarely happen that gas is swallowed by a BH which is itself swallowed.
         * In that case we do not swallow this particle: all swallowing changes before this are temporary*/
        if(swallowid == input.ID+1)
        {
            output->Mass += particle.Mass;
            MyFloat VelPred[3];
            priv.kf.SPH_VelPred(particle, sph, VelPred);
            /* Conserve momentum during accretion*/
            for(int d = 0; d < 3; d++)
                output->AccretedMomentum[d] += (particle.Mass * VelPred[d]);

            slots_mark_garbage(other, PartManager, SlotsManager);

            (*N_sph_swallowed)++;
        }
    }
};

class BHFeedbackTopTreeWalk: public TopTreeWalk<BHFeedbackQuery, BHFeedbackPriv, HmaxNODE, NGB_TREEFIND_SYMMETRIC> { using TopTreeWalk::TopTreeWalk; };

class BHFeedbackTreeWalk: public TreeWalk<BHFeedbackTreeWalk, BHFeedbackQuery, BHFeedbackResult, BHFeedbackLocalTreeWalk, BHFeedbackTopTreeWalk, BHFeedbackPriv, BHFeedbackOutput, ForceTreeHmax> {
    public:
    using TreeWalk::TreeWalk;

    /* Give each local walk the particles of the rank doing the visiting, which it changes, and the counters for its thread.*/
    BHFeedbackLocalTreeWalk make_local_walk(const BHFeedbackQuery& input)
    {
        return BHFeedbackLocalTreeWalk(tree->get_nodes(), input, priv.KernelType, output, omp_get_thread_num());
    }
};

/* Do the black hole feedback tree walk. The tree needs to have gas and BH, and hmax must have been computed:
 * the accretion treewalk does this.*/
void
blackhole_feedback(int * ActiveBlackHoles, const int64_t NumActiveBlackHoles, const ForceTreeHmax * const tree, struct BHPriv * bhpriv, const KickFactorData& kf, TimeBinMgr * timebinmgr, MPI_Comm comm)
{
    particle_data * const parts = PartManager->Base;
    /* Black holes which are themselves marked to be swallowed do not swallow or do feedback.
     * This depends on the output of the accretion treewalk, so cannot be done by haswork.*/
    int * FeedbackBlackHoles = mymalloc("FeedbackBH", int, NumActiveBlackHoles + 1);
    int64_t NumFeedbackBlackHoles = 0;
    for(int64_t i = 0; i < NumActiveBlackHoles; i++) {
        const int n = ActiveBlackHoles[i];
        if(parts[n].Type == 5 && !parts[n].Swallowed && bhpriv->BH_SwallowID[parts[n].PI] == 0)
            FeedbackBlackHoles[NumFeedbackBlackHoles++] = n;
    }

    int64_t N_sph_swallowed, N_BH_swallowed;
    {
        BHFeedbackPriv priv(PartManager->BoxSize, bhpriv, kf, timebinmgr, SlotsManager);
        BHFeedbackOutput output(bhpriv, parts, SlotsManager);
        BHFeedbackTreeWalk tw("BH_FEEDBACK", tree, priv, &output);
        tw.run(FeedbackBlackHoles, NumFeedbackBlackHoles, parts, comm);
        N_sph_swallowed = output.tot_sph_swallowed();
        N_BH_swallowed = output.tot_BH_swallowed();
    }
    myfree(FeedbackBlackHoles);

    int64_t Ntot_gas_swallowed, Ntot_BH_swallowed;
    MPI_Reduce(&N_sph_swallowed, &Ntot_gas_swallowed, 1, MPI_INT64, MPI_SUM, 0, comm);
    MPI_Reduce(&N_BH_swallowed, &Ntot_BH_swallowed, 1, MPI_INT64, MPI_SUM, 0, comm);

    message(0, "Accretion done: %ld gas particles swallowed, %ld BH particles swallowed\n", Ntot_gas_swallowed, Ntot_BH_swallowed);
}
