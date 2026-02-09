#ifndef DENSITY_H
#define DENSITY_H

#include "forcetree.h"
#include "timestep.h"
#include "densitykernel.h"
#include "utils/paramset.h"
#include "timebinmgr.h"
#include "density2.h"

struct sph_pred_data
{
    /*!< Predicted entropy at current particle drift time for SPH computation*/
    MyFloat * EntVarPred;
};

/* Structure storing the pre-computed kick factors which
 * used for making the predicted velocities.*/
struct kick_factor_data
{
    double FgravkickB;
    double gravkicks[TIMEBINS+1];
    double hydrokicks[TIMEBINS+1];
};

/*Set cooling module parameters from a density_params struct for the tests*/
void set_densitypar_old(struct density_params dp);

/* This routine computes the particle densities. If update_hsml is true
 * it runs multiple times, changing the smoothing length until
 * there are enough neighbours. If update_hsml is false (when initializing the EgyWtDensity)
 * it just computes densities.
 * If DoEgyDensity is true it also computes the entropy-weighted density for
 * pressure-entropy SPH. */
void density_old(const ActiveParticles * act, int update_hsml, int DoEgyDensity, int BlackHoleOn, const DriftKickTimes times, Cosmology * CP, struct sph_pred_data * SPH_predicted, MyFloat * GradRho_mag, const ForceTree * const tree);

void slots_free_sph_pred_data(struct sph_pred_data * sph_pred);

/* Predicted quantity computation used in hydro*/
MyFloat SPH_EntVarPred(const int i, const DriftKickTimes * times);
void SPH_VelPred(int i, MyFloat * VelPred, const struct kick_factor_data * kf);
/* Predicted velocity for dark matter, ignoring the hydro component.*/
void DM_VelPred(int i, MyFloat * VelPred, const struct kick_factor_data * kf);
/* Initialise the grav and hydrokick arrays for the current kick times.*/
void init_kick_factor_data(struct kick_factor_data * kf, const DriftKickTimes * const times, Cosmology * CP);

#endif
