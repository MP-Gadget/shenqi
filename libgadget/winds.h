#ifndef WINDS_H
#define WINDS_H

#include "forcetree.h"
#include "utils/paramset.h"
#include "utils/system.h"
#include "types.h"
#include "physconst.h"
/*
 * Enumeration of the supported wind models.
 * Wind models may be combined.
 * SH03, VS08 and OFJT10 are supported.
 * */
enum WindModel {
    WIND_SUBGRID = 1,/* If this is true, winds are spawned from the star forming gas.
                      * If false, they are spawned from neighbours of the star particle.*/
    WIND_DECOUPLE_SPH = 2,/* Specifies that wind particles are created temporarily decoupled from the gas dynamics */
    WIND_USE_HALO = 4,/* Wind speeds depend on the halo circular velocity*/
    WIND_FIXED_EFFICIENCY = 8, /* Winds have a fixed efficiency and thus fixed wind speed*/
    WIND_ISOTROPIC = 512, /* Has no effect: only isotropic winds are implemented*/
};

/*Set the parameters of the wind model*/
void set_winds_params(ParameterSet * ps);

/*Initialize the wind model from the SFR module*/
void init_winds(double FactorSN, double EgySpecSN, double PhysDensThresh, double UnitTime_in_s);

/* Get the wind speed and the wind density threshold*/
double winds_get_speed(void);
double winds_get_dens_thresh(void);

/*Evolve a wind particle, reducing its DelayTime*/
void winds_evolve(int i, double a3inv, double hubble, TimeBinMgr * timebinmgr);

/*do the treewalk for the wind model*/
void winds_and_feedback(int * NewStars, const int64_t NumNewStars, const double Time, RandTable * rnd, ForceTree * tree, DomainDecomp * ddecomp);

/*Make a wind particle at the site of recent star formation.*/
int winds_make_after_sf(int i, double sm, double vdisp, double atime, const RandTable * const rnd);

/* Make winds for the subgrid model, after computing the velocity dispersion. */
void winds_subgrid(int * MaybeWind, int NumMaybeWind, const double Time, MyFloat * StellarMasses, const RandTable * const rnd);

/* Tests whether winds spawn from gas or stars*/
int winds_are_subgrid(void);

/*Tests whether a given particle has been made a wind particle and is hydrodynamically decoupled*/
int winds_is_particle_decoupled(int i);

/*Tests whether a given particle has been made a wind particle and is hydrodynamically decoupled.
 * Version using only the sph slot, suitable for CUDA code.*/
MYCUDAFN static inline int
winds_is_particle_decoupled(const sph_particle_data * const sph_data)
{
    return (sph_data->DelayTime > 0);
}

/* Sets the MaxSignalVel for a decoupled wind particle.*/
MYCUDAFN static inline void
winds_decoupled_hydro(sph_particle_data * sphp, const double atime, const double WindSpeed, const double WindFreeTravelDensThresh)
{
    double windspeed = WindSpeed * atime;
    const double fac_mu = pow(atime, 3 * (GAMMA - 1) / 2) / atime;
    windspeed *= fac_mu;
    double hsml_c = cbrt(WindFreeTravelDensThresh /sphp->Density) * atime;
    sphp->MaxSignalVel = hsml_c * fmax(2 * windspeed, sphp->MaxSignalVel);
}

/* Returns 1 if the winds ever decouple, 0 otherwise*/
int winds_ever_decouple(void);

#endif
