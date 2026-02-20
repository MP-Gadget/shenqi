#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "libgadget/partmanager.h"
#include "libgadget/timestep.h"
#include "localtreewalk2.h"
#include "physconst.h"
#include "walltime.h"
#include "slotsmanager.h"
#include "treewalk2.h"
#include "timebinmgr.h"
#include "density2.h"
#include "timefac.h"
#include "winds.h"
#include "utils/endrun.h"
#include "utils/mymalloc.h"
#include "hydra2.h"

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

/* Function to get the center of mass density and HSML correction factor for an SPH particle with index i.
 * Encodes the main difference between pressure-entropy SPH and regular SPH.*/
static MyFloat SPH_EOMDensity(const struct sph_particle_data * const pi)
{
    if(HydroParams.DensityIndependentSphOn)
        return pi->EgyWtDensity;
    else
        return pi->Density;
}

/* Compute pressure using the predicted density (EgyWtDensity if Pressure-Entropy SPH,
 * Density otherwise) and predicted entropy*/
static double
PressurePredict(MyFloat EOMDensityPred, double EntVarPred)
{
    /* As always, this kind of micro-optimisation is dangerous.
     * However, it was timed at 10x faster with the gcc 12.1 libm (!)
     * and about 30% faster with icc 19.1. */
     if(EntVarPred * EOMDensityPred <= 0)
         return 0;
     return exp(GAMMA * log(EntVarPred * EOMDensityPred));
//     return pow(EntVarPred * EOMDensityPred, GAMMA);
}

class HydroPriv : public ParamTypeBase {
    public:
    double atime;
    double hubble;
    MyFloat * EntVarPred;
    /* Time-dependent constant factors, brought out here because
     * they need an expensive pow().*/
    double fac_mu;
    double fac_vsic_fix;
    double hubble_a2;
    DriftKickTimes * times;
    double * PressurePred;
    KickFactorData kf;
    double drifts[TIMEBINS+1];

    HydroPriv(const double BoxSize, MyFloat * i_EntVarPred, const double i_atime, DriftKickTimes * const i_times, Cosmology * CP) :
    ParamTypeBase(BoxSize), atime(i_atime), hubble(hubble_function(CP, atime)),
    EntVarPred(i_EntVarPred), fac_mu(pow(atime, 3 * (GAMMA - 1) / 2) / atime), fac_vsic_fix(hubble * pow(atime, 3 * GAMMA_MINUS1)),
    hubble_a2(hubble * atime * atime), times(i_times), kf(i_times, CP)
    {
        /* Cache the pressure for speed*/
        PressurePred = NULL;
        /* Compute pressure for particles used in density: if almost all particles are active, just pre-compute it and avoid thread contention.
        * For very small numbers of particles the memset is more expensive than just doing the exponential math,
        * so we don't pre-compute at all.*/
        if(EntVarPred) {
            PressurePred = (double *) mymalloc("PressurePred", SlotsManager->info[0].size * sizeof(double));
            /* Do it in slot order for memory locality*/
            #pragma omp parallel for
            for(int i = 0; i < SlotsManager->info[0].size; i++) {
                if(EntVarPred[i] == 0)
                    PressurePred[i] = 0;
                else
                    PressurePred[i] = PressurePredict(SPH_EOMDensity(&SphP[i]), EntVarPred[i]);
            }
        }

        /* Initialize some time factors*/
        memset(drifts, 0, sizeof(drifts[0])*(TIMEBINS+1));
        #pragma omp parallel for
        for(int i = times->mintimebin; i <= TIMEBINS; i++)
        {
            /* For density: last active drift time is Ti_kick - 1/2 timestep as the kick time is half a timestep ahead.
            * For active particles no density drift is needed.*/
            if(!is_timebin_active(i, times->Ti_Current))
                drifts[i] = get_exact_drift_factor(CP, times->Ti_lastactivedrift[i], times->Ti_Current);
        }
    }
    ~HydroPriv()
    {
        if(PressurePred)
            myfree(PressurePred);
    }
};

class HydroOutput {};

class HydroQuery : public TreeWalkQueryBase<HydroPriv> {
    public:
    /* These are only used for DensityIndependentSphOn*/
    MyFloat EgyRho;
    MyFloat EntVarPred;
    double Vel[3];
    MyFloat Hsml;
    MyFloat Mass;
    MyFloat Density;
    MyFloat Pressure;
    MyFloat F1;
    MyFloat SPH_DhsmlDensityFactor;
    MyFloat dloga;
    HydroQuery(const particle_data& particle, const int * const i_NodeList, const int firstnode, const HydroPriv& priv):
    TreeWalkQueryBase(particle, i_NodeList, firstnode, priv), EgyRho(SphP[particle.PI].EgyWtDensity),
    Hsml(particle.Hsml), Mass(particle.Mass), Density(SphP[particle.PI].Density), SPH_DhsmlDensityFactor(SphP[particle.PI].DhsmlEgyDensityFactor),
    dloga(get_dloga_for_bin(particle.TimeBinHydro, priv.times->Ti_Current))
    {
        priv.kf.SPH_VelPred(particle, Vel);
        if(priv.EntVarPred)
            EntVarPred = priv.EntVarPred[particle.PI];
        else
            EntVarPred = SPH_EntVarPred(particle, priv.times);

        const double eomdensity = SPH_EOMDensity(&SphP[particle.PI]);
        if(priv.PressurePred)
            Pressure = priv.PressurePred[particle.PI];
        else
            Pressure = PressurePredict(eomdensity, EntVarPred);
        /* calculation of F1 */
        const double soundspeed_i = sqrt(GAMMA * Pressure / eomdensity);
        F1 = fabs(SphP[particle.PI].DivVel) /
            (fabs(SphP[particle.PI].DivVel) + SphP[particle.PI].CurlVel +
             0.0001 * soundspeed_i / Hsml / priv.fac_mu);
    }
};

class HydroResult: public TreeWalkResultBase<HydroQuery, HydroOutput> {
    public:
    MyFloat Acc[3] = {0};
    MyFloat DtEntropy = 0;
    MyFloat MaxSignalVel = 0;
    HydroResult(const HydroQuery query): TreeWalkResultBase(query), Acc(0,0,0), DtEntropy(0), MaxSignalVel(0)
    {
        MaxSignalVel = sqrt(GAMMA * query.Pressure / query.EgyRho);
    }

    template<TreeWalkReduceMode mode>
    void reduce(int place, const HydroOutput& priv, struct particle_data * const parts)
    {
        TreeWalkResultBase::reduce<mode>(place, priv, parts);
        struct sph_particle_data * sphpart = &SphP[parts[place].PI];
        for(int k = 0; k < 3; k++)
            TREEWALK_REDUCE(sphpart->HydroAccel[k], Acc[k]);

        TREEWALK_REDUCE(sphpart->DtEntropy, DtEntropy);

        if constexpr(mode == TREEWALK_PRIMARY)
           if(sphpart->MaxSignalVel < MaxSignalVel)
               sphpart->MaxSignalVel = MaxSignalVel;
    }
};

/* Find the density predicted forward to the current drift time.
 * The Density in the SPHP struct is evaluated at the last time
 * the particle was active. Good for both EgyWtDensity and Density,
 * cube of the change in Hsml in drift.c. */
double
SPH_DensityPred(MyFloat Density, MyFloat DivVel, double dtdrift)
{
    /* Note minus sign!*/
    double DensityPred = Density - DivVel * Density * dtdrift;
    /* The guard should not be necessary, because the timestep is also limited. by change in hsml.
     * But add it just in case the BH has kicked the particle. The factor is set because
     * it is less than the cube of the Courant factor.*/
    if(DensityPred >= 1e-6 * Density)
        return DensityPred;
    else
        return 1e-6 * Density;
}

/* This is a symmetric NGB treewalk for hydro forces. */
class HydroLocalTreeWalk: public LocalNgbTreeWalk<HydroLocalTreeWalk, HydroQuery, HydroResult, HydroPriv, NGB_TREEFIND_SYMMETRIC, GASMASK>
{
    public:
    double p_over_rho2_i;
    double soundspeed_i;
    DensityKernel kernel_i;

    HydroLocalTreeWalk(const ForceTree * const tree, const HydroQuery& input): LocalNgbTreeWalk(tree, input)
    {
        MyFloat densityest = input.EgyRho;
        if(!HydroParams.DensityIndependentSphOn)
            densityest = input.Density;
        /* initialize variables before SPH loop is started */
        density_kernel_init(&kernel_i, input.Hsml, GetDensityKernelType());
        soundspeed_i = sqrt(GAMMA * input.Pressure / densityest);
        p_over_rho2_i = input.Pressure / (densityest * densityest);
    }

    /*! This function is the 'core' of the SPH force computation. A target
     *  particle is specified which may either be local, or reside in the
     *  communication buffer.     *
     * @param input  Query data
     * @param output Result accumulator
     */
    void ngbiter(const HydroQuery& input, const int other, HydroResult * output, const HydroPriv& priv, const struct particle_data * const parts)
    {
        if(parts[other].Mass == 0) {
            endrun(12, "Encountered zero mass particle during hydro;"
                      " We haven't implemented tracer particles and this shall not happen\n");
        }

        /* Wind particles do not interact hydrodynamically: don't produce hydro acceleration
         * or change the signalvel.*/
        if(winds_is_particle_decoupled(other))
            return;

        DensityKernel kernel_j;
        density_kernel_init(&kernel_j, parts[other].Hsml, GetDensityKernelType());

        /* Check we are within the density kernel*/
        if(r2 <= 0 || !(r2 < kernel_i.HH || r2 < kernel_j.HH))
            return;

        MyFloat VelPred[3];
        priv.kf.SPH_VelPred(parts[other], VelPred);

        double EntVarPred;
        if(priv.EntVarPred) {
            #pragma omp atomic read
            EntVarPred = priv.EntVarPred[parts[other].PI];
            /* Lazily compute the predicted quantities. We need to do this again here, even though we do it in density,
            * because this treewalk is symmetric and that one is asymmetric. In density() hmax has not been computed
            * yet so we cannot merge them. We can do this
            * with minimal locking since nothing happens should we compute them twice.
            * Zero can be the special value since there should never be zero entropy.*/
            if(EntVarPred == 0) {
                EntVarPred = SPH_EntVarPred(parts[other], priv.times);
                #pragma omp atomic write
                priv.EntVarPred[Part[other].PI] = EntVarPred;
            }
        }
        else
            EntVarPred = SPH_EntVarPred(parts[other], priv.times);

        /* Predict densities. Note that for active timebins the density is up to date so SPH_DensityPred is just returns the current densities.
         * This improves on the technique used in Gadget-2 by being a linear prediction that does not become pathological in deep timebins.*/
        const int bin = parts[other].TimeBinHydro;
        const sph_particle_data * const sphp_j = &SphP[parts[other].PI];
        const double density_j = SPH_DensityPred(sphp_j->Density, sphp_j->DivVel, priv.drifts[bin]);
        const double eomdensity = SPH_DensityPred(SPH_EOMDensity(sphp_j), sphp_j->DivVel, priv.drifts[bin]);;

        /* Compute pressure lazily*/
        double Pressure_j;

        if(priv.PressurePred) {
            #pragma omp atomic read
            Pressure_j = priv.PressurePred[parts[other].PI];
            if(Pressure_j == 0) {
                Pressure_j = PressurePredict(eomdensity, EntVarPred);
                #pragma omp atomic write
                priv.PressurePred[parts[other].PI] = Pressure_j;
            }
        }
        else
            Pressure_j = PressurePredict(eomdensity, EntVarPred);


        const double p_over_rho2_j = Pressure_j / (eomdensity * eomdensity);
        const double soundspeed_j = sqrt(GAMMA * Pressure_j / eomdensity);

        double dv[3];
        for(int d = 0; d < 3; d++) {
            dv[d] = input.Vel[d] - VelPred[d];
        }

        const double vdotr = dotproduct(dist, dv);
        const double vdotr2 = vdotr + priv.hubble_a2 * r2;

        const double r = sqrt(r2);
        const double dwk_i = density_kernel_dwk(&kernel_i, r * kernel_i.Hinv);
        const double dwk_j = density_kernel_dwk(&kernel_j, r * kernel_j.Hinv);

        double visc = 0;

        if(vdotr2 < 0)	/* ... artificial viscosity visc is 0 by default*/
        {
            /*See Gadget-2 paper: eq. 13*/
            const double mu_ij = priv.fac_mu * vdotr2 / r;	/* note: this is negative! */
            const double rho_ij = 0.5 * (input.Density + density_j);
            double vsig = soundspeed_i + soundspeed_j - 3 * mu_ij;

            if(vsig > output->MaxSignalVel)
                output->MaxSignalVel = vsig;

            /* Note this uses the CurlVel of an inactive particle, which is not at the present drift time*/
            const double f2 = fabs(sphp_j->DivVel) / (fabs(sphp_j->DivVel) +
                    SPHP(other).CurlVel + 0.0001 * soundspeed_j / priv.fac_mu / Part[other].Hsml);

            /*Gadget-2 paper, eq. 14*/
            visc = 0.25 * HydroParams.ArtBulkViscConst * vsig * (-mu_ij) / rho_ij * (input.F1 + f2);
            /* .... end artificial viscosity evaluation */
            /* now make sure that viscous acceleration is not too large */

            /*XXX: why is this dloga ?*/
            double dloga = 2 * DMAX(input.dloga, get_dloga_for_bin(Part[other].TimeBinHydro, priv.times->Ti_Current));
            if(dloga > 0 && (dwk_i + dwk_j) < 0)
            {
                if((input.Mass + parts[other].Mass) > 0) {
                    visc = DMIN(visc, 0.5 * priv.fac_vsic_fix * vdotr2 /
                            (0.5 * (input.Mass + parts[other].Mass) * (dwk_i + dwk_j) * r * dloga));
                }
            }
        }
        const double hfc_visc = 0.5 * parts[other].Mass * visc * (dwk_i + dwk_j) / r;
        double hfc = hfc_visc;
        double rr1 = 1, rr2 = 1;

        if(HydroParams.DensityIndependentSphOn) {
            /*This enables the grad-h corrections*/
            rr1 = 0, rr2 = 0;
            /* leading-order term */
            hfc += parts[other].Mass *
                (dwk_i*p_over_rho2_i*EntVarPred/input.EntVarPred +
                dwk_j*p_over_rho2_j*input.EntVarPred/EntVarPred) / r;

            /* enable grad-h corrections only if contrastlimit is non negative */
            if(HydroParams.DensityContrastLimit >= 0) {
                rr1 = input.EgyRho / input.Density;
                rr2 = eomdensity / density_j;
                if(HydroParams.DensityContrastLimit > 0) {
                    /* apply the limit if it is enabled > 0*/
                    rr1 = DMIN(rr1, HydroParams.DensityContrastLimit);
                    rr2 = DMIN(rr2, HydroParams.DensityContrastLimit);
                }
            }
        }

        /* grad-h corrections: enabled if DensityIndependentSphOn = 0, or DensityConstrastLimit >= 0 */
        /* Formulation derived from the Lagrangian */
        hfc += parts[other].Mass * (p_over_rho2_i*input.SPH_DhsmlDensityFactor * dwk_i * rr1
                    + p_over_rho2_j*sphp_j->DhsmlEgyDensityFactor * dwk_j * rr2) / r;

        for(int d = 0; d < 3; d ++)
            output->Acc[d] += (-hfc * dist[d]);
        output->DtEntropy += (0.5 * hfc_visc * vdotr2);
    }
};

class HydroTopTreeWalk: public TopTreeWalk<HydroQuery, HydroPriv, NGB_TREEFIND_SYMMETRIC> { using TopTreeWalk::TopTreeWalk; };

class HydroTreeWalk: public TreeWalk<HydroTreeWalk, HydroQuery, HydroResult, HydroLocalTreeWalk, HydroTopTreeWalk, HydroPriv, HydroOutput> {
    public:
    HydroTreeWalk(const char * const i_ev_label, const ForceTree * const i_tree, const HydroPriv& i_priv, const HydroOutput& i_out):
    TreeWalk(i_ev_label, i_tree, i_priv, i_out) {}

    bool haswork(const particle_data& particle)
    {
        return particle.Type == 0;
    }

    void postprocess(const int i, struct particle_data * const parts)
    {
        if(parts[i].Type != 0)
            return;
        /* Translate energy change rate into entropy change rate */
        SphP[parts[i].PI].DtEntropy *= GAMMA_MINUS1 / (priv.hubble_a2 * pow(SphP[parts[i].PI].Density, GAMMA_MINUS1));
        /* if we have winds, we decouple particles briefly if delaytime>0 */
        if(winds_is_particle_decoupled(i))
            winds_decoupled_hydro(i, priv.atime);
    }
};

/*! This function is the driver routine for the calculation of hydrodynamical
 *  force and rate of change of entropy due to shock heating for all active
 *  particles .
 */
void
hydro_force(const ActiveParticles * act, const double atime, MyFloat * EntVarPred, DriftKickTimes& times,  Cosmology * CP, const ForceTree * const tree)
{
    if(!tree->hmax_computed_flag)
        endrun(5, "Hydro called before hmax computed\n");

    HydroPriv priv(tree->BoxSize, EntVarPred, atime, &times, CP);
    HydroOutput output;
    HydroTreeWalk tw("HYDRO", tree, priv, output);

    walltime_measure("/SPH/Hydro/Init");

    tw.run(act->ActiveParticle, act->NumActiveParticle, PartManager->Base);

    /* collect some timing information */
    double timeall = walltime_measure(WALLTIME_IGNORE);
    double timecomp = tw.timecomp0 + tw.timecomp1 + tw.timecomp2 + tw.timecomp3;

    walltime_add("/SPH/Hydro/WalkTop", tw.timecomp0);
    walltime_add("/SPH/Hydro/WalkPrim", tw.timecomp1);
    walltime_add("/SPH/Hydro/WalkSec", tw.timecomp2);
    walltime_add("/SPH/Hydro/PostPre", tw.timecomp3);
    // walltime_add("/SPH/Hydro/Compute", timecomp);
    walltime_add("/SPH/Hydro/Wait", tw.timewait1);
    walltime_add("/SPH/Hydro/Reduce", tw.timecommsumm);
    walltime_add("/SPH/Hydro/Misc", timeall - (timecomp + tw.timewait1 + tw.timecommsumm));

    tw.print_stats(MPI_COMM_WORLD);
}
