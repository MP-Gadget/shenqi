#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <boost/math/quadrature/gauss_kronrod.hpp>
#include <boost/math/tools/toms748_solve.hpp>
#include <omp.h>

#include "physconst.h"
#include "walltime.h"
#include "slotsmanager.h"
#include "treewalk2.h"
#include "metal_return.h"
#include "densitykernel.h"
#include "density.h"
#include "cosmology.h"
#include "utils/spinlocks.h"
#include "utils/mymalloc.h"
#include "utils/endrun.h"
#include "utils/system.h"
#include "metal_tables.h"

/*! \file metal_return.c
 *  \brief Compute the mass return rate of metals from stellar evolution.
 *
 *  This file returns metals from stars with some delay.
 *  Delayed sources followed are AGB stars, SNII and Sn1a.
 *  9 Species specific yields are stored in the stars and the gas particles.
 *  Gas enrichment is not run every timestep, but only for stars that have
 *  significant enrichment, or are young.
 *  The model closely follows Illustris-TNG, https://arxiv.org/abs/1703.02970
 *  However the tables used are slightly different: we consider SNII between 8 and 40 Msun
 *  following Kobayashi 2006, where they use a hybrid of Kobayashi and Portinari.
 *  AGB yields are from Karakas 2010, like TNG, but stars with mass > 6.5 are
 *  from Doherty 2014, not Fishlock 2014. More details of the model can be found in
 *  the Illustris model Vogelsberger 2013: https://arxiv.org/abs/1305.2913
 *  As the Kobayashi table only goes to 13 Msun, stars with masses 8-13 Msun
 *  are assumed to yield like a 13 Msun star, but scaled by a factor of (M/13).
 */

#if NMETALS != NSPECIES
    #pragma error " Inconsistency in metal number between slots and metals"
#endif

/* Find stellar density, returning the total SPH Volume weights for each particle.
 * From stellar_density2.cpp, only used here*/
void stellar_density(const ActiveParticles * act, MyFloat * StarVolumeSPH, MyFloat * MassReturn, const ForceTreeHmax * const tree, const int SPHWeighting, const double MaxNgbDeviation);

static struct metal_return_params
{
    double Sn1aN0;
    int SPHWeighting;
    double MaxNgbDeviation;
} MetalParams;

/* For tests*/
void set_metal_params(double Sn1aN0)
{
    MetalParams.Sn1aN0 = Sn1aN0;
}

/*Set the parameters of the hydro module*/
void
set_metal_return_params(ParameterSet * ps)
{
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    if(ThisTask == 0) {
        MetalParams.Sn1aN0 = param_get_double(ps, "MetalsSn1aN0");
        MetalParams.SPHWeighting = param_get_int(ps, "MetalsSPHWeighting");
        MetalParams.MaxNgbDeviation = param_get_double(ps, "MetalsMaxNgbDeviation");
    }
    MPI_Bcast(&MetalParams, sizeof(struct metal_return_params), MPI_BYTE, 0, MPI_COMM_WORLD);
}

/* Build the interpolators for each yield table. We use bilinear interpolation
 * so there is no extra memory allocation and we never free the tables*/
void setup_metal_table_interp(struct interps * interp)
{
    interp->lifetime_interp        = {lifetime_metallicity, LIFE_NMET,  lifetime_masses, LIFE_NMASS,  lifetime};
    interp->agb_mass_interp        = {agb_metallicities,   AGB_NMET,   agb_masses,      AGB_NMASS,   agb_total_mass};
    interp->agb_metallicity_interp = {agb_metallicities,   AGB_NMET,   agb_masses,      AGB_NMASS,   agb_total_metals};
    int i;
    for(i=0; i<NMETALS; i++)
        interp->agb_metals_interp[i] = {agb_metallicities, AGB_NMET,   agb_masses,      AGB_NMASS,   agb_yield[i]};
    interp->snii_mass_interp        = {snii_metallicities, SNII_NMET,  snii_masses,     SNII_NMASS,  snii_total_mass};
    interp->snii_metallicity_interp = {snii_metallicities, SNII_NMET,  snii_masses,     SNII_NMASS,  snii_total_metals};
    for(i=0; i<NMETALS; i++)
        interp->snii_metals_interp[i] = {snii_metallicities, SNII_NMET, snii_masses,    SNII_NMASS,  snii_yield[i]};
}

/* Find stars returning enough metals to the gas.
 * This is a wrapper function to allow for
 * different private structs in different treewalks*/
int
metals_haswork(int i, MyFloat * MassReturn)
{
    if(Part[i].Type != 4)
        return 0;
    int pi = Part[i].PI;
    /* Don't do enrichment from all stars, just those with significant enrichment*/
    if(MassReturn[pi] < 1e-3 * (Part[i].Mass + STARP(i).TotalMassReturned))
        return 0;
    return 1;
}

/* The Chabrier IMF used for computing SnII and AGB yields.
 * See 1305.2913 eq 3. Note he writes 'log' where he means 'log10'.*/
static double chabrier_imf(double mass)
{
    if(mass <= 1) {
        return 0.852464 / mass * exp(- pow(log10(mass / 0.079)/ 0.69, 2)/2);
    }
    else {
        return 0.237912 * pow(mass, -2.3);
    }
}

/* Compute the time elapsed between two scale factors, in Myr.*/
double atime_to_myr(Cosmology *CP, double atime1, double atime2)
{
    /* t = dt/da da = 1/(Ha) da*/
    auto atime_integ = [CP] (const double atime) {
        return 1/(hubble_function(CP, atime) * atime);
    };
    double tmyr = boost::math::quadrature::gauss_kronrod<double, 61>::integrate(atime_integ, atime1, atime2);
    return tmyr * CP->UnitTime_in_s / (CP->HubbleParam * SEC_PER_MEGAYEAR);
}

/* Functions for the root finder*/
struct massbin_find_params
{
    double dtfind;
    double stellarmetal;
    const Bilinear2D * lifetime_tables;
};

/* This is the inverse of the lifetime function from the tables.
 * Need to find the stars with a given lifetime*/
static double
massendlife(double mass, const struct massbin_find_params * p)
{
    double tlifemyr = p->lifetime_tables->eval(p->stellarmetal, mass) / 1e6;
    return tlifemyr - p->dtfind;
}

/* Solve the lifetime function to find the lowest and highest mass bin that dies this timestep*/
static double do_rootfinding(struct massbin_find_params *p, double mass_low, double mass_high)
{
    auto f = [p](double mass) { return massendlife(mass, p); };
    auto tol = [](double a, double b) {
        return std::abs(a - b) < 0.005 * std::min(std::abs(a), std::abs(b));
    };
    std::uintmax_t max_iter = MAXITER;
    auto [lo, hi] = boost::math::tools::toms748_solve(f, mass_low, mass_high, tol, max_iter);
    return (lo + hi) / 2;
}

/* Find the mass bins which die in this timestep using the lifetime table.
 * dtstart, dtend - time at start and end of timestep in Myr.
 * stellarmetal - metallicity of the star.
 * lifetime_tables - 2D interpolation table of the lifetime.
 * masshigh, masslow - pointers in which to store the high and low lifetime limits
 */
void find_mass_bin_limits(double * masslow, double * masshigh, const double dtstart, const double dtend, double stellarmetal, const Bilinear2D * lifetime_tables)
{
    /* Clamp metallicities to the table values.*/
    if(stellarmetal < lifetime_metallicity[0])
        stellarmetal = lifetime_metallicity[0];
    if(stellarmetal > lifetime_metallicity[LIFE_NMET-1])
        stellarmetal = lifetime_metallicity[LIFE_NMET-1];

    /* Find the root with Boost TOMS 748. */
    struct massbin_find_params p;
    p.lifetime_tables = lifetime_tables;
    p.stellarmetal = stellarmetal;
    /* First find stars that died before the end of this timebin*/
    p.dtfind = dtend;
    /* If no stars have died yet*/
    if(massendlife(MAXMASS, &p) >= 0)
    {
        *masslow = MAXMASS;
        *masshigh = MAXMASS;
        return;
    }
    /* All stars die before the end of this timestep*/
    if(massendlife(agb_masses[0], &p) <= 0)
        *masslow = lifetime_masses[0];
    else
        *masslow = do_rootfinding(&p, agb_masses[0], MAXMASS);

    /* Now find stars that died before the start of this timebin*/
    p.dtfind = dtstart;
    /* Now we know that life(masslow) = dtend, so life(masslow) > dtstart, so life(masslow) - dtstart > 0
     * This is when no stars have died at the beginning of this timestep.*/
    if(massendlife(MAXMASS, &p) >= 0)
        *masshigh = MAXMASS;
    /* This can sometimes happen due to root finding inaccuracy.
     * Just do this star next timestep.*/
    else if(massendlife(*masslow, &p) <= 0)
        *masshigh = *masslow;
    else
        *masshigh = do_rootfinding(&p, *masslow, MAXMASS);
}

/* Parameters of the interpolator
 * to hand to the imf integral.
 * Use different interpolation structures
 * for mass return, metal return and yield.*/
struct imf_integ_params
{
    const Bilinear2D * interp;
    double metallicity;
};

/* Integrand for a function which computes a Chabrier IMF weighted quantity.*/
double chabrier_imf_integ (double mass, const struct imf_integ_params& para)
{
    /* This is needed so that the yield for SNII with masses between 8 and 13 Msun
     * are the same as the smallest mass in the table, 13 Msun,
     * but they still contribute their number density to the IMF.*/
    double intpmass = mass;
    if(mass < para.interp->ys[0])
        intpmass = para.interp->ys[0];
    if(mass > para.interp->ys[para.interp->ny - 1])
        intpmass = para.interp->ys[para.interp->ny - 1];
    double weight = para.interp->eval(para.metallicity, intpmass);
    /* This rescales the return by the original mass of the star, if it was outside the table.
     * It means that, for example, an 8 Msun star does not return more than 8 Msun. */
    weight *= (mass/intpmass);
    return weight * chabrier_imf(mass);
}

/* Compute factor to normalise the total mass in the IMF to unity.
 * Split at mass=1 where the lognormal and power-law branches meet, since
 * the derivative is discontinuous there even though the value is continuous.*/
double compute_imf_norm()
{
    auto chabrier_mass = [] (const double mass) {
        return mass * chabrier_imf(mass);
    };
    return boost::math::quadrature::gauss_kronrod<double, 61>::integrate(chabrier_mass, MINMASS, 1.0)
         + boost::math::quadrature::gauss_kronrod<double, 61>::integrate(chabrier_mass, 1.0, MAXMASS);
}

/* Compute number of Sn1a: has units of N0 = 1.3e-3, which is SN1A/(unit initial mass in M_sun).
 * Zero for age < 40 Myr. */
double sn1a_number(double dtmyrstart, double dtmyrend, double hub)
{
    /* Number of Sn1a events follows a delay time distribution (1305.2913, eq. 10) */
    const double sn1aindex = 1.12;
    const double tau8msun = 40;
    if(dtmyrend < tau8msun)
        return 0;
    /* Lower integration limit modelling formation time of WDs*/
    if(dtmyrstart < tau8msun)
        dtmyrstart  = tau8msun;
    /* Total number of Sn1a events from this star: integral evaluated from t=tau8msun to t=hubble time.*/
    const double totalSN1a = 1- pow(1/(hub*HUBBLE * SEC_PER_MEGAYEAR)/tau8msun, 1-sn1aindex);
    /* This is the integral of the DTD, normalised to the N0 rate which is in SN/M_sun.*/
    double Nsn1a = MetalParams.Sn1aN0 /totalSN1a * (pow(dtmyrstart / tau8msun, 1-sn1aindex) - pow(dtmyrend / tau8msun, 1-sn1aindex));
    return Nsn1a;
}

/* Compute yield of AGB stars: this is normalised to the yield which has units of Msun / (unit Msun in the initial SSP and so is really dimensionless.)*/
double compute_agb_yield(const Bilinear2D * agb_interp, double stellarmetal, double masslow, double masshigh)
{
    struct imf_integ_params para;
    /* Only return AGB metals for the range of AGB stars*/
    if (masshigh > SNAGBSWITCH)
        masshigh = SNAGBSWITCH;
    if (masslow < agb_masses[0])
        masslow = agb_masses[0];
    if (stellarmetal > agb_metallicities[AGB_NMET-1])
        stellarmetal = agb_metallicities[AGB_NMET-1];
    if (stellarmetal < agb_metallicities[0])
        stellarmetal = agb_metallicities[0];
    /* This happens if no bins in range had dying stars this timestep*/
    if(masslow >= masshigh)
        return 0;
    para.interp = agb_interp;
    para.metallicity = stellarmetal;
    double abserr;
    double agbyield = boost::math::quadrature::gauss_kronrod<double, 61>::integrate(
        [para] (const double mass){
            return chabrier_imf_integ(mass, para);
        },
        masslow, masshigh, 15, 1e-4, &abserr);
    return agbyield;
}

double compute_snii_yield(const Bilinear2D * snii_interp, double stellarmetal, double masslow, double masshigh)
{
    struct imf_integ_params para;
    double abserr;
    /* Only return metals for the range of SNII stars.*/
    if (masshigh > snii_masses[SNII_NMASS-1])
        masshigh = snii_masses[SNII_NMASS-1];
    if (masslow < SNAGBSWITCH)
        masslow = SNAGBSWITCH;
    if (stellarmetal > snii_metallicities[SNII_NMET-1])
        stellarmetal = snii_metallicities[SNII_NMET-1];
    if (stellarmetal < snii_metallicities[0])
        stellarmetal = snii_metallicities[0];
    para.interp = snii_interp;
    para.metallicity = stellarmetal;
    /* This happens if no bins in range had dying stars this timestep*/
    if(masslow >= masshigh)
        return 0;
    double yield = boost::math::quadrature::gauss_kronrod<double, 61>::integrate(
    [para] (const double mass){
        return chabrier_imf_integ(mass, para);
    },
    masslow, masshigh, 15, 1e-4, &abserr);
    return yield;
}

/* Compute the total mass yield for this star in this timestep*/
static double mass_yield(double dtmyrstart, double dtmyrend, double stellarmetal, double hub, struct interps * interp, double imf_norm, double masslow, double masshigh)
{
    /* Number of AGB stars/SnII by integrating the IMF*/
    double agbyield = compute_agb_yield(&interp->agb_mass_interp, stellarmetal, masslow, masshigh);
    double sniiyield = compute_snii_yield(&interp->snii_mass_interp, stellarmetal, masslow, masshigh);
    /* Fraction of the IMF which goes off this timestep. Normalised by the total IMF so we get a fraction of the SSP.*/
    double massyield = (agbyield + sniiyield)/imf_norm;
    /* Mass yield from Sn1a*/
    double Nsn1a = sn1a_number(dtmyrstart, dtmyrend, hub);
    massyield += Nsn1a * sn1a_total_metals;
    //message(3, "masslow %g masshigh %g stellarmetal %g dystart %g dtend %g agb %g snii %g sn1a %g imf_norm %g\n",
    //        masslow, masshigh, stellarmetal, dtmyrstart, dtmyrend, agbyield, sniiyield, Nsn1a * sn1a_total_metals, imf_norm);
    return massyield;
}

/* Compute the total metal yield for this star in this timestep*/
static double metal_yield(double dtmyrstart, double dtmyrend, double stellarmetal, double hub, struct interps * interp, MyFloat * MetalYields, double imf_norm, double masslow, double masshigh)
{
    double MetalGenerated = 0;
    /* Number of AGB stars/SnII by integrating the IMF*/
    MetalGenerated += compute_agb_yield(&interp->agb_metallicity_interp, stellarmetal, masslow, masshigh);
    MetalGenerated += compute_snii_yield(&interp->snii_metallicity_interp, stellarmetal, masslow, masshigh);
    MetalGenerated /= imf_norm;

    int i;
    for(i = 0; i < NMETALS; i++)
    {
        MetalYields[i] = 0;
        MetalYields[i] += compute_agb_yield(&interp->agb_metals_interp[i], stellarmetal, masslow, masshigh);
        MetalYields[i] += compute_snii_yield(&interp->snii_metals_interp[i], stellarmetal, masslow, masshigh);
        MetalYields[i] /= imf_norm;
    }
    double Nsn1a = sn1a_number(dtmyrstart, dtmyrend, hub);
    for(i = 0; i < NMETALS; i++)
        MetalYields[i] += Nsn1a * sn1a_yields[i];
    MetalGenerated += Nsn1a * sn1a_total_metals;

    return MetalGenerated;
}

/* Initialise the private structure, finding stellar mass return and ages*/
int64_t
metal_return_init(const ActiveParticles * act, Cosmology * CP, struct MetalReturnPriv * priv, const double atime)
{
    priv->hub = CP->HubbleParam;

    /* Initialize*/
    setup_metal_table_interp(&priv->interp);
    priv->StellarAges = mymalloc("StellarAges", MyFloat, SlotsManager->info[4].size);
    priv->MassReturn = mymalloc("MassReturn", MyFloat, SlotsManager->info[4].size);
    priv->LowDyingMass = mymalloc("LowDyingMass", MyFloat, SlotsManager->info[4].size);
    priv->HighDyingMass = mymalloc("HighDyingMass", MyFloat, SlotsManager->info[4].size);
    priv->StarVolumeSPH = mymalloc("StarVolumeSPH", MyFloat, SlotsManager->info[4].size);

    priv->imf_norm = compute_imf_norm();
    /* Maximum possible mass return for below*/
    double maxmassfrac = mass_yield(0, 1/(CP->HubbleParam*HUBBLE * SEC_PER_MEGAYEAR), snii_metallicities[SNII_NMET-1], CP->HubbleParam, &priv->interp, priv->imf_norm,agb_masses[0], MAXMASS);

    int64_t haswork = 0;
    /* First find the mass return as a fraction of the total mass and the age of the star.
     * This is done first so we can skip density computation for not active stars*/
    #pragma omp parallel for reduction(+: haswork)
    for(int i=0; i < act->NumActiveParticle;i++)
    {
        int p_i = act->ActiveParticle ? act->ActiveParticle[i] : i;
        if(Part[p_i].Type != 4)
            continue;
        const int slot = Part[p_i].PI;
        priv->StellarAges[slot] = atime_to_myr(CP, STARP(p_i).FormationTime, atime);
        /* Note this takes care of units*/
        double initialmass = Part[p_i].Mass + STARP(p_i).TotalMassReturned;
        find_mass_bin_limits(&priv->LowDyingMass[slot], &priv->HighDyingMass[slot], STARP(p_i).LastEnrichmentMyr, priv->StellarAges[Part[p_i].PI], STARP(p_i).Metallicity, &priv->interp.lifetime_interp);

        priv->MassReturn[slot] = initialmass * mass_yield(STARP(p_i).LastEnrichmentMyr, priv->StellarAges[Part[p_i].PI], STARP(p_i).Metallicity, CP->HubbleParam, &priv->interp, priv->imf_norm, priv->LowDyingMass[slot], priv->HighDyingMass[slot]);
        //message(3, "Particle %d PI %d massgen %g mass %g initmass %g\n", p_i, Part[p_i].PI, priv->MassReturn[Part[p_i].PI], Part[p_i].Mass, initialmass);
        /* Guard against making a zero mass particle and warn since this should not happen.*/
        if(STARP(p_i).TotalMassReturned + priv->MassReturn[slot] > initialmass * maxmassfrac) {
            if(priv->MassReturn[slot] / STARP(p_i).TotalMassReturned > 0.01)
                message(1, "Large mass return id %ld %g from %d mass %g initial %g (maxfrac %g) age %g lastenrich %g metal %g dymass %g %g\n",
                    Part[p_i].ID, priv->MassReturn[slot], p_i, STARP(p_i).TotalMassReturned, initialmass, maxmassfrac, priv->StellarAges[Part[p_i].PI], STARP(p_i).LastEnrichmentMyr, STARP(p_i).Metallicity, priv->LowDyingMass[slot], priv->HighDyingMass[slot]);
            priv->MassReturn[slot] = initialmass * maxmassfrac - STARP(p_i).TotalMassReturned;
            if(priv->MassReturn[slot] < 0) {
                priv->MassReturn[slot] = 0;
            }
            /* Ensure that we skip this step*/
            if(!metals_haswork(p_i, priv->MassReturn))
                STARP(p_i).LastEnrichmentMyr = priv->StellarAges[Part[p_i].PI];
        }
        /* Keep count of how much work we need to do*/
        if(metals_haswork(p_i, priv->MassReturn))
            haswork++;
    }
    return haswork;
}

/* Free memory allocated by metal_return_init */
void
metal_return_priv_free(struct MetalReturnPriv * priv)
{
    myfree(priv->StarVolumeSPH);
    myfree(priv->HighDyingMass);
    myfree(priv->LowDyingMass);
    myfree(priv->MassReturn);
    myfree(priv->StellarAges);
}

/* Inputs to the metal return treewalk, which are not changed by it.*/
class MetalsPriv : public ParamTypeBase {
    public:
    /* Contains the mass return, ages and stellar volumes found before the treewalk, indexed by the star slot.*/
    struct MetalReturnPriv * metals;
    const star_particle_data * StarParts;
    int SPHWeighting;
    enum DensityKernelType KernelType;

    MetalsPriv(const double BoxSize, struct MetalReturnPriv * const i_metals, slots_manager_type * slotsmanager):
        ParamTypeBase(BoxSize), metals(i_metals), StarParts(slotsmanager->star_slot()),
        SPHWeighting(MetalParams.SPHWeighting), KernelType(GetDensityKernelType())
    {}
};

class MetalsQuery : public TreeWalkQueryBase<MetalsPriv>
{
    public:
    MyFloat Metallicity;
    MyFloat Mass;
    MyFloat Hsml;
    MyFloat StarVolumeSPH;
    /* This is the metal/mass generated this timestep.*/
    MyFloat MetalSpeciesGenerated[NMETALS];
    MyFloat MassGenerated;
    MyFloat MetalGenerated;

    /* This constructor is unusually important:
     * it computes the total amount of metals to be returned in this timestep.*/
    MetalsQuery(const particle_data& particle, const int * const i_NodeList, const MetalsPriv& priv):
    TreeWalkQueryBase<MetalsPriv>(particle, i_NodeList, priv), Mass(particle.Mass), Hsml(particle.Hsml)
    {
        const int pi = particle.PI;
        const star_particle_data& star = priv.StarParts[pi];
        struct MetalReturnPriv * const metals = priv.metals;
        Metallicity = star.Metallicity;
        StarVolumeSPH = metals->StarVolumeSPH[pi];
        const double InitialMass = particle.Mass + star.TotalMassReturned;
        const double dtmyrend = metals->StellarAges[pi];
        const double dtmyrstart = star.LastEnrichmentMyr;
        /* This is the total mass returned from this stellar population this timestep. Note this is already in the desired units.*/
        MassGenerated = metals->MassReturn[pi];
        /* This returns the total amount of metal produced this timestep, and also fills out MetalSpeciesGenerated, which is an
         * element by element table of the metal produced by dying stars this timestep.*/
        const double total_z_yield = metal_yield(dtmyrstart, dtmyrend, Metallicity, metals->hub, &metals->interp, MetalSpeciesGenerated, metals->imf_norm, metals->LowDyingMass[pi], metals->HighDyingMass[pi]);
        /* The total metal returned is the metal ejected into the ISM this timestep. total_z_yield is given as a fraction of the initial SSP.*/
        MetalGenerated = InitialMass * total_z_yield;
        /* It should be positive! If it is not, this is some integration error
         * in the yield table as we cannot destroy metal which is not present.*/
        if(MetalGenerated < 0)
            MetalGenerated = 0;
        /* Similarly for all the other metal species*/
        for(int i = 0; i < NMETALS; i++) {
            MetalSpeciesGenerated[i] *= InitialMass;
            if(MetalSpeciesGenerated[i] < 0)
                MetalSpeciesGenerated[i] = 0;
        }
    }
};

class MetalsOutput {
    public:
    /* The mass actually returned to the gas by each star, indexed by the star slot.
     * This may be less than the MassReturn used to make the query, as gas particles which are too heavy are skipped.
     * It is a separate array because the queries for exported stars may be made after the primary treewalk has reduced.*/
    MyFloat * MassReturned;
    star_particle_data * StarParts;
    /* The rest is the storage on the rank doing the visiting: the gas particles
     * which receive the metals, and a lock for each of them.*/
    particle_data * Parts;
    sph_particle_data * SphParts;
    struct SpinLocks * spin;
    /* Maximum of the new gas mass*/
    double MaxGasMass;

    MetalsOutput(const double i_MaxGasMass, particle_data * const i_Parts, slots_manager_type * slotsmanager):
        StarParts(slotsmanager->star_slot()), Parts(i_Parts), SphParts(slotsmanager->sph_slot()), MaxGasMass(i_MaxGasMass)
    {
        MassReturned = mymalloc("MassReturned", MyFloat, slotsmanager->info[4].size);
        spin = init_spinlocks(slotsmanager->info[0].size);
    }

    ~MetalsOutput()
    {
        free_spinlocks(spin);
        myfree(MassReturned);
    }

    /* Update the mass and enrichment variables for the star.
     * Note that the stellar metallicity is not updated, as the
     * metal-forming stars are now dead and their metals in the gas.*/
    void
    postprocess(const int place, particle_data * const parts, const MetalsPriv * priv)
    {
        const int pi = parts[place].PI;
        /* Conserve mass returned*/
        parts[place].Mass -= MassReturned[pi];
        StarParts[pi].TotalMassReturned += MassReturned[pi];
        /* Update the last enrichment time*/
        StarParts[pi].LastEnrichmentMyr = priv->metals->StellarAges[pi];
    }
};

class MetalsResult : public TreeWalkResultBase<MetalsQuery, MetalsOutput> {
    public:
    /* This is the total mass returned to
     * the surrounding gas particles, for mass conservation.*/
    MyFloat MassReturn = 0;

    MetalsResult(const MetalsQuery& query):
        TreeWalkResultBase<MetalsQuery, MetalsOutput>(query) {}

    /* Store the amount of mass actually returned.*/
    template<TreeWalkReduceMode mode>
    void reduce(int place, const MetalsOutput * output, struct particle_data * const parts)
    {
        TREEWALK_REDUCE(output->MassReturned[parts[place].PI], MassReturn);
    }
};

/*! For all gas particles within the density radius of this star,
 * add a fraction of the total mass and metals generated,
 * weighted by the SPH kernel distance from the star.
 * This changes the gas particles, which are on the rank doing the visiting.
 */
class MetalsLocalTreeWalk: public LocalNgbTreeWalk<MetalsLocalTreeWalk, MetalsQuery, MetalsResult, MetalsPriv, HmaxNODE, NGB_TREEFIND_ASYMMETRIC, GASMASK>
{
    public:
    DensityKernel kernel;
    /* The gas particles on the rank doing the visiting. Several stars may enrich
     * the same gas particle at once, so each gas particle is locked while it is changed.*/
    MetalsOutput * const visited;

    MetalsLocalTreeWalk(const HmaxNODE * const Nodes, const MetalsQuery& input, const enum DensityKernelType KernelType, MetalsOutput * const i_visited):
    LocalNgbTreeWalk<MetalsLocalTreeWalk, MetalsQuery, MetalsResult, MetalsPriv, HmaxNODE, NGB_TREEFIND_ASYMMETRIC, GASMASK>(Nodes, input),
    visited(i_visited)
    {
        density_kernel_init(&kernel, input.Hsml, KernelType);
    }

    void ngbiter(MetalsQuery& input, const particle_data& const_particle, MetalsResult * output, const MetalsPriv& priv)
    {
        double dist[3];
        const double r2 = this->get_distance(input, const_particle, priv.BoxSize, dist);

        if(!(r2 > 0 && r2 < kernel.HH))
            return;

        double wk = 1;
        const double u = sqrt(r2) * kernel.Hinv;

        if(priv.SPHWeighting)
            wk = density_kernel_wk(&kernel, u);
        double ThisMetals[NMETALS];
        if(input.StarVolumeSPH == 0)
            endrun(3, "StarVolumeSPH %g hsml %g\n", input.StarVolumeSPH, input.Hsml);

        /* The particle is in the particle table of this rank, and we will change it.*/
        const int other = &const_particle - visited->Parts;
        particle_data& particle = visited->Parts[other];
        const int pi = particle.PI;
        sph_particle_data& sph = visited->SphParts[pi];

        lock_spinlock(pi, visited->spin);
        /* Volume of particle weighted by the SPH kernel*/
        const double volume = particle.Mass / sph.Density;
        const double returnfraction = wk * volume / input.StarVolumeSPH;
        const double thismass = returnfraction * input.MassGenerated;
        /* Ensure that the gas particles don't become overweight.
         * If there are few gas particles around, the star clusters
         * will hold onto their metals.*/
        if(particle.Mass + thismass > visited->MaxGasMass) {
            unlock_spinlock(pi, visited->spin);
            return;
        }
        /* Add metals weighted by SPH kernel*/
        for(int i = 0; i < NMETALS; i++)
            ThisMetals[i] = returnfraction * input.MetalSpeciesGenerated[i];
        const double thismetal = returnfraction * input.MetalGenerated;
        /* Add the metals to the particle.*/
        for(int i = 0; i < NMETALS; i++)
            sph.Metals[i] = (sph.Metals[i] * particle.Mass + ThisMetals[i])/(particle.Mass + thismass);
        /* Update total metallicity*/
        sph.Metallicity = (sph.Metallicity * particle.Mass + thismetal)/(particle.Mass + thismass);
        /* Update mass*/
        const double massfrac = (particle.Mass + thismass) / particle.Mass;
        particle.Mass *= massfrac;
        /* Density also needs a correction so the volume fraction is unchanged.
         * This ensures that volume = Mass/Density is unchanged for the next particle
         * and thus the weighting still sums to unity.*/
        sph.Density *= massfrac;
        /* Keep track of how much was returned for conservation purposes*/
        output->MassReturn += thismass;
        const double newmass = particle.Mass;
        unlock_spinlock(pi, visited->spin);
        if(newmass <= 0)
            endrun(3, "New mass %g new metal %g in particle %d id %ld from star mass %g metallicity %g\n",
                   newmass, sph.Metallicity, other, particle.ID, input.Mass, input.Metallicity);
    }
};

class MetalsTopTreeWalk: public TopTreeWalk<MetalsQuery, MetalsPriv, HmaxNODE, NGB_TREEFIND_ASYMMETRIC> { using TopTreeWalk::TopTreeWalk; };

class MetalsTreeWalk: public TreeWalk<MetalsTreeWalk, MetalsQuery, MetalsResult, MetalsLocalTreeWalk, MetalsTopTreeWalk, MetalsPriv, MetalsOutput, ForceTreeHmax> {
    public:
    using TreeWalk::TreeWalk;

    /* Give each local walk the gas particles of the rank doing the visiting, which it changes.*/
    MetalsLocalTreeWalk make_local_walk(const MetalsQuery& input)
    {
        return MetalsLocalTreeWalk(tree->get_nodes(), input, priv.KernelType, output);
    }
};

/* Run the treewalk which returns the metals to the gas.*/
static void
metal_return_treewalk(const ActiveParticles * act, const ForceTreeHmax * const tree, struct MetalReturnPriv * metals, MPI_Comm comm)
{
    particle_data * const parts = PartManager->Base;
    /* Find the stars returning enough metals to the gas. This depends on the
     * mass return computed above, so cannot be done by the treewalk haswork.*/
    const int64_t nactive = act->NumActiveParticle;
    int * MetalStars = mymalloc("MetalStars", int, nactive + 1);
    int64_t NumMetalStars = 0;
    for(int64_t i = 0; i < nactive; i++) {
        const int p_i = act->ActiveParticle ? act->ActiveParticle[i] : (int) i;
        if(parts[p_i].IsGarbage || parts[p_i].Swallowed)
            continue;
        if(metals_haswork(p_i, metals->MassReturn))
            MetalStars[NumMetalStars++] = p_i;
    }
    {
        MetalsPriv priv(PartManager->BoxSize, metals, SlotsManager);
        MetalsOutput output(metals->MaxGasMass, parts, SlotsManager);
        MetalsTreeWalk tw("METALS", tree, priv, &output);
        tw.run(MetalStars, NumMetalStars, parts, comm);
    }
    myfree(MetalStars);
}

/*! This function is the driver routine for the calculation of metal return. */
void
metal_return(const ActiveParticles * act, const ForceTreeHmax * const gasTree, Cosmology * CP, const double atime, const double AvgGasMass)
{
    /* Do nothing if no stars yet*/
    int64_t totstar;
    MPI_Allreduce(&SlotsManager->info[4].size, &totstar, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);
    if(totstar == 0)
        return;

    struct MetalReturnPriv priv[1];

    int64_t nwork = metal_return_init(act, CP, priv, atime);

    /* Maximum mass of a gas particle after enrichment: cap it at a few times the initial mass.
     * FIXME: Ideally we should here fork a new particle with a smaller gas mass. We should
     * figure out then how set the gas entropy. A possibly better idea is to add
     * a generic routine to split gas particles into the density code.*/
    priv->MaxGasMass = 4* AvgGasMass;

    int64_t totwork;
    MPI_Allreduce(&nwork, &totwork, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);

    walltime_measure("/SPH/Metals/Init");

    if(totwork == 0) {
        metal_return_priv_free(priv);
        return;
    }

    if(!gasTree || !(gasTree->mask & GASMASK))
        endrun(5, "metal_return called with bad tree: mask %d\n", gasTree ? gasTree->mask : 0);

    /* Compute total number of weights around each star for actively returning stars*/
    stellar_density(act, priv->StarVolumeSPH, priv->MassReturn, gasTree, MetalParams.SPHWeighting, MetalParams.MaxNgbDeviation);

    /* Do the metal return*/
    metal_return_treewalk(act, gasTree, priv, MPI_COMM_WORLD);

    metal_return_priv_free(priv);

    /* collect some timing information */
    walltime_measure("/SPH/Metals/Yield");
}
