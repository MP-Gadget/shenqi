#ifndef HYDRA_H
#define HYDRA_H

#include "forcetree.h"
#include "timestep.h"
#include "density.h"
#include "utils/paramset.h"
#include "hydra2.h"

/*Function to compute hydro accelerations and adiabatic entropy change*/
void hydro_force_old(const ActiveParticles * act, const double atime, struct sph_pred_data * SPH_predicted, const DriftKickTimes times,  Cosmology * CP, const ForceTree * const tree);

/*Set cooling module parameters from a density_params struct for the tests*/
void set_hydropar_old(struct hydro_params dp);
#endif
