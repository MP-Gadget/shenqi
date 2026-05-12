/*Prototypes and structures for the wind model*/

#include <math.h>
#include <string.h>
#include <omp.h>
#include <execution>
#include <algorithm>
#include "treewalk2.h"
#include "localtreewalk2.h"
#include "winds.h"
#include "physconst.h"
#include "slotsmanager.h"
#include "timebinmgr.h"
#include "walltime.h"
#include "utils/endrun.h"
#include "utils/mymalloc.h"

/*Parameters of the wind model*/
static struct WindParams
{
    enum WindModel WindModel;  /*!< Which wind model is in use? */
    double WindFreeTravelLength;
    double WindFreeTravelDensFac;
    /*Density threshold at which to recouple wind particles.*/
    double WindFreeTravelDensThresh;
    /* Maximum time in internal time units to allow the wind to be free-streaming.*/
    double MaxWindFreeTravelTime;
    /* used in VS08 and SH03*/
    double WindEfficiency;
    double WindSpeed;
    double WindEnergyFraction;
    /* used in OFJT10*/
    double WindSigma0;
    double WindSpeedFactor;
    /* Minimum wind velocity for kicked particles, in internal velocity units*/
    double MinWindVelocity;
    /* Fraction of wind energy in thermal energy*/
    double WindThermalFactor;
} wind_params;

/*Set the parameters of the wind module.
 ofjt10 is Okamoto, Frenk, Jenkins and Theuns 2010 https://arxiv.org/abs/0909.0265
 VS08 is Dalla Vecchia & Schaye 2008 https://arxiv.org/abs/0801.2770
 SH03 is Springel & Hernquist 2003 https://arxiv.org/abs/astro-ph/0206395*/
void set_winds_params(ParameterSet * ps)
{
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    if(ThisTask == 0) {
        /*Wind model parameters*/
        wind_params.WindModel = (enum WindModel) param_get_enum(ps, "WindModel");
        /* The following two are for VS08 and SH03*/
        wind_params.WindEfficiency = param_get_double(ps, "WindEfficiency");
        wind_params.WindEnergyFraction = param_get_double(ps, "WindEnergyFraction");

        /* The following two are for OFJT10*/
        wind_params.WindSigma0 = param_get_double(ps, "WindSigma0");
        wind_params.WindSpeedFactor = param_get_double(ps, "WindSpeedFactor");

        wind_params.WindThermalFactor = param_get_double(ps, "WindThermalFactor");
        wind_params.MinWindVelocity = param_get_double(ps, "MinWindVelocity");
        wind_params.MaxWindFreeTravelTime = param_get_double(ps, "MaxWindFreeTravelTime");
        wind_params.WindFreeTravelLength = param_get_double(ps, "WindFreeTravelLength");
        wind_params.WindFreeTravelDensFac = param_get_double(ps, "WindFreeTravelDensFac");
    }
    MPI_Bcast(&wind_params, sizeof(struct WindParams), MPI_BYTE, 0, MPI_COMM_WORLD);
}

void
init_winds(double FactorSN, double EgySpecSN, double PhysDensThresh, double UnitTime_in_s)
{
    wind_params.WindSpeed = sqrt(2 * wind_params.WindEnergyFraction * FactorSN * EgySpecSN / (1 - FactorSN));
    /* Convert wind free travel time from Myr to internal units*/
    wind_params.MaxWindFreeTravelTime = wind_params.MaxWindFreeTravelTime * SEC_PER_MEGAYEAR / UnitTime_in_s;
    wind_params.WindFreeTravelDensThresh = wind_params.WindFreeTravelDensFac * PhysDensThresh;
    if(HAS(wind_params.WindModel, WIND_FIXED_EFFICIENCY)) {
        wind_params.WindSpeed /= sqrt(wind_params.WindEfficiency);
        message(0, "Windspeed: %g MaxDelay %g\n", wind_params.WindSpeed, wind_params.MaxWindFreeTravelTime);
    } else if(HAS(wind_params.WindModel, WIND_USE_HALO)) {
        message(0, "Reference Windspeed: %g, MaxDelay %g\n", wind_params.WindSigma0 * wind_params.WindSpeedFactor, wind_params.MaxWindFreeTravelTime);
    } else {
        /* Check for undefined wind models*/
        endrun(1, "WindModel = 0x%X is strange. This shall not happen.\n", wind_params.WindModel);
    }
}

int
winds_are_subgrid(void)
{
    if(HAS(wind_params.WindModel, WIND_SUBGRID))
        return 1;
    else
        return 0;
}

int
winds_is_particle_decoupled(int i)
{
    if(HAS(wind_params.WindModel, WIND_DECOUPLE_SPH)
        && Part[i].Type == 0 && SPHP(i).DelayTime > 0)
            return 1;
    return 0;
}

double
winds_get_speed(void)
{
    return wind_params.WindSpeed;
}

double
winds_get_dens_thresh(void)
{
    return wind_params.WindFreeTravelDensThresh;
}

class WindWeightPriv : public ParamTypeBase {
    public:
    sph_particle_data * SphParts;
    star_particle_data * StarParts;
    WindWeightPriv(const double BoxSize, slots_manager_type& SlotsManager):
    ParamTypeBase(BoxSize), SphParts(SlotsManager.sph_slot()), StarParts(SlotsManager.star_slot())
    { }
};

class WindWeightQuery : public TreeWalkQueryBase<WindWeightPriv> {
    public:
    double Hsml;
    MYCUDAFN WindWeightQuery(const particle_data& particle, const int * const i_NodeList, const int firstnode, const WindWeightPriv& priv):
    TreeWalkQueryBase<WindWeightPriv>(particle, i_NodeList, firstnode, priv), Hsml(particle.Hsml)
    {}
};

class WindWeightOutput {
    public:
    int64_t winddata_sz;
    int * nvisited;
    double * TotalWeight;
    WindWeightOutput(slots_manager_type * SlotsManager) {
        /* Subgrid winds come from gas, regular wind from stars: size array accordingly.*/
        winddata_sz = SlotsManager->info[4].size;
        if(HAS(wind_params.WindModel, WIND_SUBGRID))
            winddata_sz = SlotsManager->info[0].size;
        TotalWeight = (double * ) mymalloc("WindWeight", winddata_sz * sizeof(double));
        memset(TotalWeight, 0, winddata_sz * sizeof(double));
        nvisited = (int * ) mymalloc("nvisited", winddata_sz * sizeof(int));
        memset(nvisited, 0, winddata_sz * sizeof(int));
    }

    int64_t get_tot_visited()
    {
        int64_t tot_nvisited = 0;
        #pragma omp parallel for reduction(+: tot_nvisited)
        for(int i = 0; i < winddata_sz; i++) {
            tot_nvisited += nvisited[i];
        }
        return tot_nvisited;
    }

    void print_visited(const int64_t NumNewStars)
    {
        int64_t tot_nvisited = 0, max_nvisited = 0;
        #pragma omp parallel for reduction(+: tot_nvisited) reduction(max: max_nvisited)
        for(int i = 0; i < winddata_sz; i++) {
            tot_nvisited += nvisited[i];
            if(max_nvisited < nvisited[i])
                max_nvisited = nvisited[i];
        }
        double mean_nvisited = static_cast<double>(tot_nvisited) / static_cast<double>(NumNewStars);
        message(3, "WINDS: Mean visited: %g max: %ld\n", mean_nvisited, max_nvisited);
    }

    MYCUDAFN void postprocess(const int i, particle_data * const parts, const WindWeightPriv * priv){}

    ~WindWeightOutput()
    {
        myfree(nvisited);
        myfree(TotalWeight);
    }
};

class WindWeightResult : public TreeWalkResultBase<WindWeightQuery, WindWeightOutput> {
    public:
    int nvisited;
    double TotalWeight;
    WindWeightResult(const WindWeightQuery& input) : TreeWalkResultBase<WindWeightQuery, WindWeightOutput>(input), nvisited(0), TotalWeight(0)
    { }

    template<TreeWalkReduceMode mode>
    MYCUDAFN void reduce(int place, WindWeightOutput * output, struct particle_data * const parts)
    {
        int pi = parts[place].PI;
        TREEWALK_REDUCE(output->TotalWeight[pi], TotalWeight);
        /* Do not do this on the GPU! */
        TREEWALK_REDUCE(output->nvisited[pi], nvisited);
    }
};

/* this evaluator walks the tree and sums the total mass of surrounding gas
 * particles as described in VS08. */
class WindWeightLocalTreeWalk: public LocalNgbTreeWalk<WindWeightLocalTreeWalk, WindWeightQuery, WindWeightResult, WindWeightPriv, NGB_TREEFIND_ASYMMETRIC, GASMASK>
{
    public:
    MYCUDAFN WindWeightLocalTreeWalk(const NODE * const Nodes, const WindWeightQuery& input):
    LocalNgbTreeWalk<WindWeightLocalTreeWalk, WindWeightQuery, WindWeightResult, WindWeightPriv, NGB_TREEFIND_ASYMMETRIC, GASMASK>(Nodes, input) {}

    MYCUDAFN void ngbiter(WindWeightQuery& input, const particle_data& particle, WindWeightResult * output, const WindWeightPriv& priv)
    {
        double dist[3];
        double r2 = this->get_distance(input, particle, priv.BoxSize, dist);

        /* Check we are within the density kernel*/
        if(r2 <= 0 || !(r2 < input.Hsml * input.Hsml))
            return;

        /* skip earlier wind particles, which receive
            * no feedback energy */
        if(priv.SphParts[particle.PI].DelayTime > 0) return;

        /* NOTE: think twice if we want a symmetric tree walk when wk is used. */
        //double wk = density_kernel_wk(&kernel, r);
        double wk = 1.0;
        output->TotalWeight += wk * particle.Mass;
        /* Sum up all particles visited on this processor*/
        output->nvisited++;
        /*
        message(1, "ThisTask = %d %ld ngb=%d NGB=%d TotalWeight=%g V2sum=%g V1sum=%g %g %g\n",
        ThisTask, I->ID, numngb, O->Ngb, O->TotalWeight, O->V2sum,
        O->V1sum[0], O->V1sum[1], O->V1sum[2]);
        */
    }
};

class WindWeightTopTreeWalk: public TopTreeWalk<WindWeightQuery, WindWeightPriv, NGB_TREEFIND_ASYMMETRIC> { using TopTreeWalk::TopTreeWalk; };

class WindWeightTreeWalk: public TreeWalk<WindWeightTreeWalk, WindWeightQuery, WindWeightResult, WindWeightLocalTreeWalk, WindWeightTopTreeWalk, WindWeightPriv, WindWeightOutput> {
    public:
    using TreeWalk::TreeWalk;
};

static void
get_wind_params(double * vel, double * windeff, double * utherm, const double vdisp, const double time);

class WindKickPriv : public ParamTypeBase {
    public:
    double * TotalWeight;
    int * nvisited;
    double Time;
    RandTable * rnd;
    sph_particle_data * SphParts;
    star_particle_data * StarParts;
    WindKickPriv(const double BoxSize, const double i_Time, WindWeightOutput& output, RandTable * i_rnd, slots_manager_type& SlotsManager):
    ParamTypeBase(BoxSize), TotalWeight(output.TotalWeight), nvisited(output.nvisited),Time(i_Time), rnd(i_rnd), SphParts(SlotsManager.sph_slot()), StarParts(SlotsManager.star_slot())
    {

    }
};

class WindKickQuery : public TreeWalkQueryBase<WindKickPriv> {
    public:
    MyIDType ID;
    double Mass;
    double Hsml;
    double Vdisp;
    double TotalWeight;
    int nvisited;
    MYCUDAFN WindKickQuery(const particle_data& particle, const int * const i_NodeList, const int firstnode, const WindKickPriv& priv):
    TreeWalkQueryBase<WindKickPriv>(particle, i_NodeList, firstnode, priv), ID(particle.ID), Mass(particle.Mass), Hsml(particle.Hsml),
    TotalWeight(priv.TotalWeight[particle.PI]), nvisited(priv.nvisited[particle.PI])
    {
        if(particle.Type != 4)
            endrun(5, "Particle has type %d not a star, id %ld mass %g\n", particle.Type, particle.ID, particle.Mass);
        Vdisp = priv.StarParts[particle.PI].VDisp;
    }
};

/* Structure to store a potential kick
 * to a gas particle from a newly formed star.
 * We add a queue of these and resolve them
 * after the treewalk. Note this star may
 * be on another processor.*/
struct StarKick
{
    /* Index of the kicked particle*/
    particle_data * particle;
    /* Distance to the star. The closest star does the kick.*/
    double StarDistance;
    /* Star ID, for resolving ties.*/
    MyIDType StarID;
    /* Kick velocity if this kick is the one used*/
    double StarKickVelocity;
    /* Thermal energy included in the kick*/
    double StarTherm;

    /* Comparison function to sort the StarKicks by particle id, distance and star ID.
     * The closest star is used. */
    bool operator < (const StarKick& starb) const
    {
        if(particle > starb.particle)
            return false;
        if(particle < starb.particle)
            return true;
        if(StarDistance > starb.StarDistance)
            return false;
        if(StarDistance < starb.StarDistance)
            return true;
        if(StarID > starb.StarID)
            return false;
        if(StarID < starb.StarID)
            return true;
        return false;
    }
};

class WindKickOutput {
    public:
    int64_t maxkicks;
    int64_t nkicks;
    struct StarKick * kicks;
    WindKickOutput(const int64_t nvisited) : maxkicks(nvisited+2), nkicks(0) {
        /* Some particles may be kicked by multiple stars on the same timestep.
        * To ensure this happens only once and does not depend on the order in
        * which the loops are executed, particles are kicked by the nearest new star.
        * This struct stores all such possible kicks, and we sort it out after the treewalk.*/
        kicks = (struct StarKick * ) mymalloc("StarKicks", (maxkicks+1) * sizeof(struct StarKick));
        /* Explicitly initialise so we are safe to do a reduce even if there are no kicks.*/
        kicks[0].StarKickVelocity = 0;
    }

    MYCUDAFN void postprocess(const int i, particle_data * const parts, const WindKickPriv * priv){}

    ~WindKickOutput()
    {
        myfree(kicks);
    }
};

#define NKICKS 75

class WindKickResult : public TreeWalkResultBase<WindKickQuery, WindKickOutput> {
    public:
    struct StarKick kicks[NKICKS];
    int64_t nkicks;
    int64_t maxkicks;
    WindKickResult(const WindKickQuery& input) : TreeWalkResultBase<WindKickQuery, WindKickOutput>(input), nkicks(0), maxkicks(std::min(NKICKS, input.nvisited))
    { }
    template<TreeWalkReduceMode mode>
    MYCUDAFN void reduce(int place, WindKickOutput * output, struct particle_data * const parts)
    {
        int firstkick;
        #pragma omp atomic capture
        {
            firstkick = output->nkicks;
            output->nkicks += nkicks;
        }
        for(int i = 0; i < nkicks; i++) {
            output->kicks[firstkick++] = kicks[i];
        }
    }
};

/* this evaluator walks the tree and sums the total mass of surrounding gas
 * particles as described in VS08. */
class WindKickLocalTreeWalk: public LocalNgbTreeWalk<WindKickLocalTreeWalk, WindKickQuery, WindKickResult, WindKickPriv, NGB_TREEFIND_ASYMMETRIC, GASMASK>
{
    public:
    MYCUDAFN WindKickLocalTreeWalk(const NODE * const Nodes, const WindKickQuery& input):
    LocalNgbTreeWalk<WindKickLocalTreeWalk, WindKickQuery, WindKickResult, WindKickPriv, NGB_TREEFIND_ASYMMETRIC, GASMASK>(Nodes, input) {}

    MYCUDAFN void ngbiter(WindKickQuery& input, const particle_data& particle, WindKickResult * output, const WindKickPriv& priv)
    {
        double dist[3];
        double r2 = this->get_distance(input, particle, priv.BoxSize, dist);

        /* Check we are within the density kernel*/
        if(r2 <= 0 || !(r2 < input.Hsml * input.Hsml))
            return;

        /* skip earlier wind particles */
        if(priv.SphParts[particle.PI].DelayTime > 0)
            return;

        /* No eligible gas particles not in wind*/
        if(input.TotalWeight == 0 || input.Vdisp <= 0) return;

        /* Paranoia*/
        if(particle.Type != 0 || particle.IsGarbage || particle.Swallowed)
            return;

        /* Get the velocity, thermal energy and efficiency of the kick*/
        double utherm = 0, v=0, windeff = 0;
        get_wind_params(&v, &windeff, &utherm, input.Vdisp, priv.Time);

        double p = windeff * input.Mass / input.TotalWeight;
        double random = get_random_number(input.ID + particle.ID, priv.rnd);

        if (random < p && v > 0) {
            /* Store a potential kick. This might not be the kick actually used,
             * because another star particle may be closer, but we can resolve
             * that after the treewalk*/
            /* Use a single global kick list.*/
            int64_t ikick = output->nkicks;
            if(ikick >= output->maxkicks) {
                message(5, "Not enough room in kick queue: %ld > %ld for particle ID %ld starid %ld distance %g\n",
                       ikick, output->maxkicks, particle.ID, input.ID, sqrt(r2));
                return;
            }
            struct StarKick * kick = &output->kicks[ikick];
            kick->StarDistance = sqrt(r2);
            kick->StarID = input.ID;
            kick->StarKickVelocity = v;
            kick->StarTherm = utherm;
            kick->particle = const_cast<particle_data *>(&particle);
            output->nkicks++;
        }
    }
};

class WindKickTopTreeWalk: public TopTreeWalk<WindKickQuery, WindKickPriv, NGB_TREEFIND_ASYMMETRIC> { using TopTreeWalk::TopTreeWalk; };

class WindKickTreeWalk: public TreeWalk<WindKickTreeWalk, WindKickQuery, WindKickResult, WindKickLocalTreeWalk, WindKickTopTreeWalk, WindKickPriv, WindKickOutput> {
    public:
    using TreeWalk::TreeWalk;
};

/* Returns 1 if the winds ever decouple, 0 otherwise*/
int winds_ever_decouple(void)
{
    if(!HAS(wind_params.WindModel, WIND_DECOUPLE_SPH))
        return 0;
    if(wind_params.MaxWindFreeTravelTime > 0)
        return 1;
    else
        return 0;
}

/* This function spawns winds for the subgrid model, which comes from the star-forming gas.
 * Does a little more calculation than is really necessary, due to shared code, but that shouldn't matter. */
void
winds_subgrid(int * MaybeWind, int NumMaybeWind, const double Time, MyFloat * StellarMasses, const RandTable * const rnd)
{
    /*The non-subgrid model does nothing here*/
    if(!HAS(wind_params.WindModel, WIND_SUBGRID))
        return;

    if(!MPIU_Any(NumMaybeWind > 0, MPI_COMM_WORLD))
        return;

    int n;
    #pragma omp parallel for
    for(n = 0; n < NumMaybeWind; n++)
    {
        int i = MaybeWind ? MaybeWind[n] : n;
        /* Notice that StellarMasses is indexed like PI, not i!*/
        MyFloat sm = StellarMasses[Part[i].PI];
        winds_make_after_sf(i, sm, SPHP(i).VDisp, Time, rnd);
    }
    walltime_measure("/Cooling/Wind");
}

static int
get_wind_dir(int64_t ID, double dir[3], const RandTable * const rnd) {
    /* v and vmean are in internal units (km/s *a ), not km/s !*/
    /* returns 0 if particle i is converted to wind. */
    // message(1, "%ld Making ID=%ld (%g %g %g) to wind with v= %g\n", ID, Part[i].ID, Part[i].Pos[0], Part[i].Pos[1], Part[i].Pos[2], v);
    /* ok, make the particle go into the wind */
    double theta = acos(2 * get_random_number(ID + 3, rnd) - 1);
    double phi = 2 * M_PI * get_random_number(ID + 4, rnd);

    dir[0] = sin(theta) * cos(phi);
    dir[1] = sin(theta) * sin(phi);
    dir[2] = cos(theta);
    return 0;
}

/* Do the actual kick of the gas particle*/
static void
wind_do_kick(particle_data& particle, double vel, double therm, double atime, const RandTable * const rnd, sph_particle_data * SphParts)
{
    /* Kick the gas particle*/
    double dir[3];
    get_wind_dir(particle.ID, dir, rnd);
    if(vel > 0 && atime > 0) {
        for(int j = 0; j < 3; j++)
            particle.Vel[j] += vel * dir[j];
        /* StarTherm is internal energy per unit mass. Need to convert to entropy*/
        const double enttou = pow(SphParts[particle.PI].Density / pow(atime, 3), GAMMA_MINUS1) / GAMMA_MINUS1;
        SphParts[particle.PI].Entropy += therm/enttou;
        if(winds_ever_decouple()) {
            double delay = wind_params.WindFreeTravelLength / (vel / atime);
            if(delay > wind_params.MaxWindFreeTravelTime)
                delay = wind_params.MaxWindFreeTravelTime;
            SphParts[particle.PI].DelayTime = delay;
        }
    }
}

/*Do a treewalk for the wind model. This only changes newly created star particles.*/
void
winds_and_feedback(int * NewStars, const int64_t NumNewStars, const double Time, RandTable * rnd, ForceTree * tree, DomainDecomp * ddecomp)
{
    /*The subgrid model does nothing here*/
    if(HAS(wind_params.WindModel, WIND_SUBGRID))
        return;

    if(!MPIU_Any(NumNewStars > 0, MPI_COMM_WORLD))
        return;

    /* Flags that we need to free the tree to preserve memory order*/
    bool tree_alloc_in_wind = false;
    if(!tree->tree_allocated_flag) {
        message(0, "Building tree in wind\n");
        tree_alloc_in_wind = true;
        force_tree_rebuild_mask(tree, ddecomp, GASMASK, NULL);
        walltime_measure("/Cooling/Build");
    }

    WindWeightPriv priv(PartManager->BoxSize, *SlotsManager);

    WindWeightOutput output(SlotsManager);
    WindWeightTreeWalk tw("WIND_WEIGHT", tree, priv, &output);

    /* Find densities*/
    tw.run(NewStars, NumNewStars, PartManager->Base, ddecomp->DomainComm);

    WindKickPriv privkick(PartManager->BoxSize, Time, output, rnd, *SlotsManager);
    WindKickOutput outputkick(output.get_tot_visited());
    output.print_visited(NumNewStars);

    /* Then run feedback: types used: gas. */
    WindKickTreeWalk twkick("WIND_KICK", tree, privkick, &outputkick);
    twkick.run(NewStars, NumNewStars, PartManager->Base, ddecomp->DomainComm);

    /* Sort the possible kicks*/
    std::sort(std::execution::par_unseq, outputkick.kicks, outputkick.kicks + outputkick.nkicks);
    /* Not parallel as the number of kicked particles should be pretty small*/
    particle_data * last_part = NULL;
    int64_t nkicked = 0;
    for(int i = 0; i < outputkick.nkicks; i++) {
        /* Only do the kick for the first particle, which is the closest*/
        if(outputkick.kicks[i].particle == last_part)
            continue;
        particle_data * particle = outputkick.kicks[i].particle;
        last_part = particle;
        nkicked++;
        wind_do_kick(*particle, outputkick.kicks[i].StarKickVelocity, outputkick.kicks[i].StarTherm, Time, rnd, SlotsManager->sph_slot());
        if(outputkick.kicks[i].StarKickVelocity <= 0 || !isfinite(outputkick.kicks[i].StarKickVelocity) || !isfinite(privkick.SphParts[particle->PI].DelayTime))
        {
            endrun(5, "Odd v: other = %ld, DT = %g v = %g i = %d, nkicks %ld maxkicks %ld dist %g id %ld\n",
                   particle->ID, privkick.SphParts[particle->PI].DelayTime, outputkick.kicks[i].StarKickVelocity, i, outputkick.nkicks, outputkick.maxkicks,
                   outputkick.kicks[i].StarDistance, outputkick.kicks[i].StarID);
        }
    }
    /* Get total number of potential new stars to allocate memory.*/
    int64_t tot_newstars, tot_kicks, tot_applied;
    double maxvel;
    MPI_Reduce(&NumNewStars, &tot_newstars, 1, MPI_INT64, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&outputkick.nkicks, &tot_kicks, 1, MPI_INT64, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&nkicked, &tot_applied, 1, MPI_INT64, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&outputkick.kicks[0].StarKickVelocity, &maxvel, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    message(0, "Made %ld gas wind, discarded %ld kicks from %ld stars. Vel %g\n", tot_applied, tot_kicks - tot_applied, tot_newstars, maxvel);

    if(tree_alloc_in_wind)
        force_tree_free(tree);
    walltime_measure("/Cooling/Wind");
}

/*Evolve a wind particle, reducing its DelayTime*/
void
winds_evolve(int i, double a3inv, double hubble, TimeBinMgr * timebinmgr)
{
    /*Remove a wind particle from the delay mode if the (physical) density has dropped sufficiently.*/
    if(SPHP(i).DelayTime > 0 && SPHP(i).Density * a3inv < wind_params.WindFreeTravelDensThresh) {
        SPHP(i).DelayTime = 0;
    }
    /*Reduce the time until the particle can form stars again by the current timestep*/
    if(SPHP(i).DelayTime > 0) {
        /* Enforce the maximum in case of restarts*/
        if(SPHP(i).DelayTime > wind_params.MaxWindFreeTravelTime)
            SPHP(i).DelayTime = wind_params.MaxWindFreeTravelTime;
        const double dloga = timebinmgr->get_dloga_for_bin(Part[i].TimeBinHydro, Part[i].Ti_drift);
        /*  the proper time duration of the step */
        const double dtime = dloga / hubble;
        SPHP(i).DelayTime = fmax(SPHP(i).DelayTime - dtime, 0);
    }
}

/* Get the parameters of the wind kick*/
static void
get_wind_params(double * vel, double * windeff, double * utherm, const double vdisp, const double time)
{
    /* Physical velocity*/
    double vphys = vdisp / time;
    *utherm = wind_params.WindThermalFactor * 1.5 * vphys * vphys;
    if(HAS(wind_params.WindModel, WIND_FIXED_EFFICIENCY)) {
        *windeff = wind_params.WindEfficiency;
        *vel = wind_params.WindSpeed * time;
    } else if(HAS(wind_params.WindModel, WIND_USE_HALO)) {
        *windeff = pow(wind_params.WindSigma0, 2) / (vphys * vphys + 2 * (*utherm));
        *vel = wind_params.WindSpeedFactor * vdisp;
    } else {
        endrun(1, "WindModel = 0x%X is strange. This shall not happen.\n", wind_params.WindModel);
    }
    /* Minimum wind velocity. This ensures particles do not remain in the wind forever*/
    if(*vel < wind_params.MinWindVelocity * time)
        *vel = wind_params.MinWindVelocity * time;
}

int
winds_make_after_sf(int i, double sm, double vdisp, double atime, const RandTable * const rnd)
{
    if(!HAS(wind_params.WindModel, WIND_SUBGRID))
        return 0;

    /* Get the velocity, thermal energy and efficiency of the kick*/
    double utherm = 0, vel=0, windeff = 0;
    get_wind_params(&vel, &windeff, &utherm, vdisp, atime);

    /* Here comes the Springel Hernquist 03 wind model */
    /* Notice that this is the mass of the gas particle after forking a star, Mass - Mass/GENERATIONS.*/
    double pw = windeff * sm / Part[i].Mass;
    double prob = 1 - exp(-pw);
    if(get_random_number(Part[i].ID + 2, rnd) < prob) {
        wind_do_kick(Part[i], vel, utherm, atime, rnd, SlotsManager->sph_slot());
    }
    return 0;
}
