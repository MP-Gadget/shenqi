#ifndef HYDRA2_H
#define HYDRA2_H

#include "forcetree.h"
#include "timestep.h"
#include "density.h"
#include "utils/paramset.h"

/*Function to compute hydro accelerations and adiabatic entropy change*/
void hydro_force(const ActiveParticles * act, const double atime, MyFloat * EntVarPred, const DriftKickTimes& times,  Cosmology * CP, const ForceTree * const tree);

struct hydro_params
{
    /* Enables density independent (Pressure-entropy) SPH */
    int DensityIndependentSphOn;
    /* limit of density contrast ratio for hydro force calculation (only effective with Density Indep. Sph) */
    double DensityContrastLimit;
    /*!< Sets the parameter \f$\alpha\f$ of the artificial viscosity */
    double ArtBulkViscConst;
};

void set_hydro_params(ParameterSet * ps);
struct hydro_params get_hydropar(void);
/*Set cooling module parameters from a density_params struct for the tests*/
void set_hydropar(struct hydro_params dp);

/* Gets whether we are using Density Independent Sph*/
int DensityIndependentSphOn(void);

#endif
