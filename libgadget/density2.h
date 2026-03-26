#ifndef DENSITY2_H
#define DENSITY2_H
#include <math.h>
#include "partmanager.h"
#include "cosmology.h"
#include "timestep.h"
#include "timefac.h"
#include "forcetree.h"
#include "densitykernel.h"
#include "physconst.h"

struct density_params
{
    double DensityResolutionEta;		/*!< SPH resolution eta. See Price 2011. eq 12*/
    double MaxNumNgbDeviation;	/*!< Maximum allowed deviation neighbour number */

    /* These are for black hole neighbour finding and so belong in the density module, not the black hole module.*/
    double BlackHoleNgbFactor;	/*!< Factor by which the normal SPH neighbour should be increased/decreased */
    double BlackHoleMaxAccretionRadius;

    enum DensityKernelType DensityKernelType;  /* 0 for Cubic Spline,  (recmd NumNgb = 33)
                               1 for Quintic spline (recmd  NumNgb = 97) */

    /*!< minimum allowed SPH smoothing length in units of gravitational softening length */
    double MinGasHsmlFractional;
    /* In internal units*/
    double MinGasHsml;
};

/*Set/Get the parameters of the density module*/
void set_density_params(ParameterSet * ps);
struct density_params get_densitypar(void);

/*Set cooling module parameters from a density_params struct for the tests*/
void set_densitypar(struct density_params dp);

/* This routine computes the particle densities. If update_hsml is true
 * it runs multiple times, changing the smoothing length until
 * there are enough neighbours. If update_hsml is false (when initializing the EgyWtDensity)
 * it just computes densities.
 * If DoEgyDensity is true it also computes the entropy-weighted density for
 * pressure-entropy SPH. */
void density(const ActiveParticles * act, int update_hsml, int DoEgyDensity, int BlackHoleOn, DriftKickTimes& times, TimeBinMgr * timebinmgr, Cosmology * CP, MyFloat ** EntVarPred, MyFloat * GradRho_mag, const ForceTree * const tree, bool UseGPU=false);

/* Get the desired nuber of neighbours for the supplied kernel*/
double GetNumNgb(enum DensityKernelType KernelType);

/* Get the current density kernel type*/
enum DensityKernelType GetDensityKernelType(void);

/* Structure storing the pre-computed kick factors which
 * used for making the predicted velocities.*/
class KickFactorData
{
    public:
    double FgravkickB;
    double gravkicks[TIMEBINS+1];
    double hydrokicks[TIMEBINS+1];
    double dloga[TIMEBINS+1];

    /* Initialise the grav and hydrokick arrays for the current kick times.*/
    MYCUDAFN
    KickFactorData(const DriftKickTimes * const times, Cosmology * CP, TimeBinMgr * timebins)
    {
        int i;
        /* Factor this out since all particles have the same PM kick time*/
        FgravkickB = get_exact_gravkick_factor(CP, times->PM_kick, times->Ti_Current);
        memset(gravkicks, 0, sizeof(gravkicks[0])*(TIMEBINS+1));
        memset(hydrokicks, 0, sizeof(hydrokicks[0])*(TIMEBINS+1));
        memset(dloga, 0, sizeof(dloga[0])*(TIMEBINS+1));
        /* Compute the factors to move a current kick times velocity to the drift time velocity.
         * We need to do the computation for all timebins up to the maximum because even inactive
         * particles may have interactions. */
        #pragma omp parallel for
        for(i = times->mintimebin; i <= TIMEBINS; i++)
        {
            gravkicks[i] = get_exact_gravkick_factor(CP, times->Ti_kick[i], times->Ti_Current);
            hydrokicks[i] = get_exact_hydrokick_factor(CP, times->Ti_kick[i], times->Ti_Current);
            dloga[i] = timebins->dloga_from_dti(times->Ti_Current - times->Ti_kick[i], times->Ti_Current);
        }
    }

    /* Get the predicted velocity for a particle
     * at the current Force computation time ti,
     * which always coincides with the Drift inttime.
     * For hydro forces.*/
    MYCUDAFN
    void
    SPH_VelPred(const particle_data& particle, const sph_particle_data& sph_data, MyFloat * VelPred) const
    {
        /* Notice that the kick time for gravity and hydro may be different! So the prediction is also different*/
        for(int j = 0; j < 3; j++) {
            VelPred[j] = particle.Vel[j] + gravkicks[particle.TimeBinGravity] * particle.FullTreeGravAccel[j]
                + particle.GravPM[j] * FgravkickB + hydrokicks[particle.TimeBinHydro] * sph_data.HydroAccel[j];
        }
    }

    /* Get the predicted velocity for a particle
     * at the current Force computation time ti,
     * which always coincides with the Drift inttime.
     * For hydro forces.*/
    MYCUDAFN
    void
    DM_VelPred(const particle_data& particle, MyFloat * VelPred) const
    {
        int j;
        for(j = 0; j < 3; j++)
            VelPred[j] = particle.Vel[j] + gravkicks[particle.TimeBinGravity] * particle.FullTreeGravAccel[j]+ particle.GravPM[j] * FgravkickB;
    }

    /* The evolved entropy at drift time: evolved dlog a.
    * Used to predict pressure and entropy for SPH */
    MYCUDAFN MyFloat
    SPH_EntVarPred(const particle_data& particle, const sph_particle_data& sph_part)
    {
            double EntVarPred = sph_part.Entropy + sph_part.DtEntropy * dloga[particle.TimeBinHydro];
            /*Entropy limiter for the predicted entropy: makes sure entropy stays positive. */
            if(EntVarPred < 0.05*sph_part.Entropy)
                EntVarPred = 0.05 * sph_part.Entropy;
            /* Just in case*/
            if(EntVarPred <= 0)
                return 0;
            EntVarPred = exp(1./GAMMA * log(EntVarPred));
    //         EntVarPred = pow(EntVarPred, 1/GAMMA);
            return EntVarPred;
    };
};

/* Set the initial smoothing length for gas and BH. Used on first timestep in init()*/
void set_init_hsml(ForceTree * tree, DomainDecomp * ddecomp, const double MeanGasSeparation, struct part_manager_type * const PartManager);

#endif
