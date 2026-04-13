#ifndef HYDRATREE2_HPP
#define HYDRATREE2_HPP

#include "partmanager.h"
#include "timestep.h"
#include "localtreewalk2.h"
#include "physconst.h"
#include "slotsmanager.h"
#include "treewalk2.h"
#include "timebinmgr.h"
#include "density2.h"
#include "winds.h"
#include "utils/mymalloc.h"
#include "hydra2.h"
#include "densitykernel.hpp"

/* Function to get the center of mass density and HSML correction factor for an SPH particle with index i.
 * Encodes the main difference between pressure-entropy SPH and regular SPH.
 * This could be a template but that seems too much effort.*/
static inline MYCUDAFN MyFloat SPH_EOMDensity(const struct sph_particle_data * const pi, const bool DensityIndependentSphOn)
{
    if(DensityIndependentSphOn)
        return pi->EgyWtDensity;
    else
        return pi->Density;
}

/* Compute pressure using the predicted density (EgyWtDensity if Pressure-Entropy SPH,
 * Density otherwise) and predicted entropy*/
static MYCUDAFN double
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
    double * PressurePred;
    KickFactorData kf;
    double drifts[TIMEBINS+1];
    /* HydroParams*/
    double ArtBulkViscConst;
    double DensityContrastLimit;
    bool DensityIndependentSphOn;
    /* Wind model parameters*/
    double WindSpeed;
    double WindFreeTravelDensThresh;
    /* Pointer to the SPH particle data array.*/
    sph_particle_data * SphParts;

    HydroPriv(const double BoxSize, MyFloat * i_EntVarPred, const double i_atime, DriftKickTimes * const times, TimeBinMgr * timebinmgr, Cosmology * CP, hydro_params HydroPar) :
    ParamTypeBase(BoxSize), atime(i_atime), hubble(hubble_function(CP, atime)),
    EntVarPred(i_EntVarPred), fac_mu(pow(atime, 3 * (GAMMA - 1) / 2) / atime), fac_vsic_fix(hubble * pow(atime, 3 * GAMMA_MINUS1)),
    hubble_a2(hubble * atime * atime), kf(times, timebinmgr),
    ArtBulkViscConst(HydroPar.ArtBulkViscConst), DensityContrastLimit(HydroPar.DensityContrastLimit), DensityIndependentSphOn(HydroPar.DensityIndependentSphOn),
    WindSpeed(winds_get_speed()), WindFreeTravelDensThresh(winds_get_dens_thresh()),
    SphParts(reinterpret_cast<sph_particle_data *>(SlotsManager->info[0].ptr))
    {
        /* Cache the pressure for speed*/
        PressurePred = NULL;
        /* Compute pressure for particles used in density: if almost all particles are active, just pre-compute it and avoid thread contention.
        * For very small numbers of particles the memset is more expensive than just doing the exponential math,
        * so we don't pre-compute at all.*/
        if(EntVarPred) {
            PressurePred = (double *) mymanagedmalloc("PressurePred", SlotsManager->info[0].size * sizeof(double));
            /* Do it in slot order for memory locality*/
            #pragma omp parallel for
            for(int i = 0; i < SlotsManager->info[0].size; i++) {
                PressurePred[i] = PressurePredict(SPH_EOMDensity(&SphParts[i], DensityIndependentSphOn), EntVarPred[i]);
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
                drifts[i] = timebinmgr->get_exact_drift_factor(times->Ti_lastactivedrift[i], times->Ti_Current);
        }
    }
    ~HydroPriv()
    {
        if(PressurePred)
            myfree(PressurePred);
    }
};

class HydroOutput {
    public:
    /* Pointer to the SPH particle data array*/
    sph_particle_data * SphParts;

    HydroOutput(slots_manager_type * SlotsManager): SphParts(reinterpret_cast<sph_particle_data *>(SlotsManager->info[0].ptr)) {}

    MYCUDAFN void postprocess(const int i, particle_data * const parts, const HydroPriv * priv)
    {
        if(parts[i].Type != 0)
            return;
        sph_particle_data * sphp = &SphParts[parts[i].PI];
        /* Translate energy change rate into entropy change rate */
        SphParts[parts[i].PI].DtEntropy *= GAMMA_MINUS1 / (priv->hubble_a2 * pow(sphp->Density, GAMMA_MINUS1));
        /* if we have winds, we decouple particles briefly if delaytime>0 */
        if(winds_is_particle_decoupled(sphp)) {
            for(int k = 0; k < 3; k++)
                sphp->HydroAccel[k] = 0;
            sphp->DtEntropy = 0;
            winds_decoupled_hydro(sphp, priv->atime, priv->WindSpeed, priv->WindFreeTravelDensThresh);
        }
    }
};

class HydroQuery : public TreeWalkQueryBase<HydroPriv> {
    public:
    /* This is set to Density if DensityIndependentSphOn is false*/
    MyFloat EgyRho;
    /* Only used for DensityIndependentSphOn = 1*/
    MyFloat EntVarPred;
    double Vel[3];
    MyFloat Hsml;
    MyFloat Mass;
    MyFloat Density;
    MyFloat Pressure;
    MyFloat F1;
    MyFloat SPH_DhsmlDensityFactor;
    int TimeBinHydro;
    MYCUDAFN HydroQuery(const particle_data& particle, const int * const i_NodeList, const int firstnode, const HydroPriv& priv):
    TreeWalkQueryBase(particle, i_NodeList, firstnode, priv), EgyRho(priv.SphParts[particle.PI].EgyWtDensity),
    Hsml(particle.Hsml), Mass(particle.Mass), Density(priv.SphParts[particle.PI].Density), SPH_DhsmlDensityFactor(priv.SphParts[particle.PI].DhsmlEgyDensityFactor), TimeBinHydro(particle.TimeBinHydro)
    {
        sph_particle_data& sphp_i = priv.SphParts[particle.PI];

        if(priv.DensityIndependentSphOn)
            EgyRho = sphp_i.EgyWtDensity;
        else
            EgyRho = sphp_i.Density;
        priv.kf.SPH_VelPred(particle, sphp_i, Vel);
        if(priv.EntVarPred)
            EntVarPred = priv.EntVarPred[particle.PI];
        else
            EntVarPred = priv.kf.SPH_EntVarPred(particle, sphp_i);

        const double eomdensity_i = SPH_EOMDensity(&sphp_i, priv.DensityIndependentSphOn);
        if(priv.PressurePred)
            Pressure = priv.PressurePred[particle.PI];
        else
            Pressure = PressurePredict(eomdensity_i, EntVarPred);
        /* calculation of F1 */
        const double soundspeed_i = sqrt(GAMMA * Pressure / eomdensity_i);
        F1 = fabs(sphp_i.DivVel) /
            (fabs(sphp_i.DivVel) + sphp_i.CurlVel +
             0.0001 * soundspeed_i / Hsml / priv.fac_mu);
    };

    static MYCUDAFN bool haswork(const particle_data& particle)
    {
        if(!TreeWalkQueryBase::haswork(particle))
            return false;
        return particle.Type == 0;
    };
};

class HydroResult: public TreeWalkResultBase<HydroQuery, HydroOutput> {
    public:
    MyFloat Acc[3] = {0};
    MyFloat DtEntropy = 0;
    MyFloat MaxSignalVel = 0;
    MYCUDAFN HydroResult(const HydroQuery query): TreeWalkResultBase(query), DtEntropy(0), MaxSignalVel(0)
    {
        /* Note that if DensityIndependentSphOn is false, query.EgyRho is just Density*/
        MaxSignalVel = sqrt(GAMMA * query.Pressure / query.EgyRho);
    }

    template<TreeWalkReduceMode mode>
    MYCUDAFN void reduce(int place, const HydroOutput * output, struct particle_data * const parts)
    {
        TreeWalkResultBase::reduce<mode>(place, output, parts);
        struct sph_particle_data * sphpart = &output->SphParts[parts[place].PI];
        for(int k = 0; k < 3; k++)
            TREEWALK_REDUCE(sphpart->HydroAccel[k], Acc[k]);

        TREEWALK_REDUCE(sphpart->DtEntropy, DtEntropy);

        /* Need TREEWALK_PRIMARY or sphpart->MaxSignalVel < MaxSignalVel*/
        if constexpr(mode == TREEWALK_PRIMARY)
            sphpart->MaxSignalVel = MaxSignalVel;
        else if(sphpart->MaxSignalVel < MaxSignalVel)
               sphpart->MaxSignalVel = MaxSignalVel;
    }
};

/* Find the density predicted forward to the current drift time.
 * The Density in the SPHP struct is evaluated at the last time
 * the particle was active. Good for both EgyWtDensity and Density,
 * cube of the change in Hsml in drift.c. */
MYCUDAFN static inline double
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
template <typename DensityKernel>
class HydroLocalTreeWalk: public LocalNgbTreeWalk<HydroLocalTreeWalk<DensityKernel>, HydroQuery, HydroResult, HydroPriv, NGB_TREEFIND_SYMMETRIC, GASMASK>
{
    public:
    double p_over_rho2_i;
    double soundspeed_i;
    DensityKernel kernel_i;

    MYCUDAFN HydroLocalTreeWalk(const NODE * const Nodes, const HydroQuery& input): LocalNgbTreeWalk<HydroLocalTreeWalk<DensityKernel>, HydroQuery, HydroResult, HydroPriv, NGB_TREEFIND_SYMMETRIC, GASMASK>(Nodes, input), kernel_i(input.Hsml)
    {
        /* initialize variables before SPH loop is started.
         * Note that EgyRho is Density if DensityIndependentSphOn = 0 */
        soundspeed_i = sqrt(GAMMA * input.Pressure / input.EgyRho);
        p_over_rho2_i = input.Pressure / (input.EgyRho * input.EgyRho);
    }

    /*! This function is the 'core' of the SPH force computation. A target
     *  particle is specified which may either be local, or reside in the
     *  communication buffer.     *
     * @param input  Query data
     * @param output Result accumulator
     */
    MYCUDAFN void ngbiter(const HydroQuery& input, const particle_data& particle, HydroResult * output, const HydroPriv& priv)
    {
        double dist[3];
        double r2 = this->get_distance(input, particle, priv.BoxSize, dist);

        /* Check we are within the density kernel*/
        if(r2 <= 0 || !(r2 < input.Hsml * input.Hsml || r2 < particle.Hsml * particle.Hsml))
            return;

        /* Wind particles do not interact hydrodynamically: don't produce hydro acceleration
         * or change the signalvel.*/
        const sph_particle_data& sphp_j = priv.SphParts[particle.PI];

        if(winds_is_particle_decoupled(&sphp_j))
            return;

        DensityKernel kernel_j(particle.Hsml);

        MyFloat VelPred[3];
        priv.kf.SPH_VelPred(particle, sphp_j, VelPred);

        double EntVarPred;
        if(priv.EntVarPred)
            EntVarPred = priv.EntVarPred[particle.PI];
        else
            EntVarPred = priv.kf.SPH_EntVarPred(particle, sphp_j);

        /* Predict densities. Note that for active timebins the density is up to date so SPH_DensityPred is just returns the current densities.
         * This improves on the technique used in Gadget-2 by being a linear prediction that does not become pathological in deep timebins.*/
        const int bin = particle.TimeBinHydro;
        const double density_j = SPH_DensityPred(sphp_j.Density, sphp_j.DivVel, priv.drifts[bin]);
        const double eomdensity_j = SPH_DensityPred(SPH_EOMDensity(&sphp_j, priv.DensityIndependentSphOn), sphp_j.DivVel, priv.drifts[bin]);

        /* Compute pressure lazily*/
        double Pressure_j;

        if(priv.PressurePred)
            Pressure_j = priv.PressurePred[particle.PI];
        else
            Pressure_j = PressurePredict(eomdensity_j, EntVarPred);

        const double p_over_rho2_j = Pressure_j / (eomdensity_j * eomdensity_j);
        const double soundspeed_j = sqrt(GAMMA * Pressure_j / eomdensity_j);

        double dv[3];
        for(int d = 0; d < 3; d++) {
            dv[d] = input.Vel[d] - VelPred[d];
        }

        const double vdotr = dot_product(dist, dv);
        const double vdotr2 = vdotr + priv.hubble_a2 * r2;

        const double r = sqrt(r2);
        const double dwk_i = kernel_i.dwk(r / kernel_i.H);
        const double dwk_j = kernel_j.dwk(r / kernel_j.H);

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
            const double f2 = fabs(sphp_j.DivVel) / (fabs(sphp_j.DivVel) +
                    sphp_j.CurlVel + 0.0001 * soundspeed_j / priv.fac_mu / particle.Hsml);

            /*Gadget-2 paper, eq. 14*/
            visc = 0.25 * priv.ArtBulkViscConst * vsig * (-mu_ij) / rho_ij * (input.F1 + f2);
            /* .... end artificial viscosity evaluation */
            /* now make sure that viscous acceleration is not too large */

            /*XXX: why is this dloga ?*/
            double dloga = 2 * fmax(priv.kf.dloga[input.TimeBinHydro], priv.kf.dloga[particle.TimeBinHydro]);
            if(dloga > 0 && (dwk_i + dwk_j) < 0)
            {
                if((input.Mass + particle.Mass) > 0) {
                    visc = fmin(visc, 0.5 * priv.fac_vsic_fix * vdotr2 /
                            (0.5 * (input.Mass + particle.Mass) * (dwk_i + dwk_j) * r * dloga));
                }
            }
        }
        const double hfc_visc = 0.5 * particle.Mass * visc * (dwk_i + dwk_j) / r;
        double hfc = hfc_visc;
        double rr1 = 1, rr2 = 1;

        if(priv.DensityIndependentSphOn) {
            /*This enables the grad-h corrections*/
            rr1 = 0, rr2 = 0;
            /* leading-order term */
            hfc += particle.Mass *
                (dwk_i*p_over_rho2_i*EntVarPred/input.EntVarPred +
                dwk_j*p_over_rho2_j*input.EntVarPred/EntVarPred) / r;

            /* enable grad-h corrections only if contrastlimit is non negative */
            if(priv.DensityContrastLimit >= 0) {
                rr1 = input.EgyRho / input.Density;
                rr2 = eomdensity_j / density_j;
                /* Apply the contrastlimit only if it is strictly greater than zero*/
                if(priv.DensityContrastLimit > 0) {
                    rr1 = fmin(rr1, priv.DensityContrastLimit);
                    rr2 = fmin(rr2, priv.DensityContrastLimit);
                }
            }
        }

        /* grad-h corrections: enabled if DensityIndependentSphOn = 0, or DensityContrastLimit >= 0 */
        /* Formulation derived from the Lagrangian */
        hfc += particle.Mass * (p_over_rho2_i*input.SPH_DhsmlDensityFactor * dwk_i * rr1
                    + p_over_rho2_j*sphp_j.DhsmlEgyDensityFactor * dwk_j * rr2) / r;

        for(int d = 0; d < 3; d ++)
            output->Acc[d] += (-hfc * dist[d]);
        output->DtEntropy += (0.5 * hfc_visc * vdotr2);
    }
};

class HydroTopTreeWalk: public TopTreeWalk<HydroQuery, HydroPriv, NGB_TREEFIND_SYMMETRIC> { using TopTreeWalk::TopTreeWalk; };

template <typename TreeWalkType>
void do_hydro_walk(const ActiveParticles * act, const ForceTree * tree, HydroPriv * priv, HydroOutput * output)
{
    TreeWalkType tw("HYDRO", tree, *priv, output);
    tw.run(act->ActiveParticle, act->NumActiveParticle, PartManager->Base, MPI_COMM_WORLD);
    tw.print_stats("/SPH/Hydro", MPI_COMM_WORLD);
}

#ifdef USE_CUDA
void hydro_force_cuda(const ActiveParticles * act, const ForceTree * tree, HydroPriv * priv, HydroOutput * output, enum DensityKernelType DensityKernelType);
#endif

#endif
