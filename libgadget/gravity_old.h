#ifndef GRAVSHORT_OLD_H
#define GRAVSHORT_OLD_H

#include "forcetree.h"
#include "petapm.h"
#include "timestep.h"
#include "gravity.h"

/* Fill the short-range gravity table*/
void gravshort_fill_ntab(const enum ShortRangeForceWindowType ShortRangeForceWindowType, const double Asmth);

/* Apply the short-range window function, which includes the smoothing kernel.*/
int grav_apply_short_range_window(double r, double * fac, double * pot, const double cellsize);

/* Set up the module*/
void set_gravshort_tree_params_old(ParameterSet * ps);
/* Helpers for the tests*/
void set_gravshort_treepar_old(struct gravshort_tree_params tree_params);
struct gravshort_tree_params get_gravshort_treepar_old(void);

void grav_short_pair(const ActiveParticles * act, PetaPM * pm, ForceTree * tree, double Rcut, double rho0);
void grav_short_tree_old(const ActiveParticles * act, PetaPM * pm, ForceTree * tree, MyFloat (* AccelStore)[3], double rho0, inttime_t Ti_Current);

#endif
