#ifndef THERMALVEL_H
#define THERMALVEL_H

#include <boost/math/interpolators/makima.hpp>
#include <boost/random/ranlux.hpp>
#include <boost/random/uniform_01.hpp>
#include <vector>
/*Length of the table*/
#define MAX_FERMI_DIRAC          17.0
#define LENGTH_FERMI_DIRAC_TABLE 2000

struct thermalvel
{
    double fermi_dirac_vel[LENGTH_FERMI_DIRAC_TABLE];
    double fermi_dirac_cumprob[LENGTH_FERMI_DIRAC_TABLE];
    double m_vamp;
    boost::math::interpolators::makima<std::vector<double>> * fd_intp;
};

/*Single parameter is the amplitude of the random velocities. All the physics is in here.
 * max_fd and min_fd give the maximum and minimum velocities to integrate over.
 * Note these values are dimensionless*/
/*Returns total fraction of the Fermi-Dirac distribution between max_fd and min_fd*/
double
init_thermalvel(struct thermalvel * thermals, const double v_amp, double max_fd, const double min_fd);

/*Add a randomly generated thermal speed in v_amp*(min_fd, max_fd) to a 3-velocity.*/
void
add_thermal_speeds(struct thermalvel * thermals, boost::random::ranlux48 *g_rng, float Vel[]);

/*Amplitude of the random velocity for neutrinos*/
double
NU_V0(const double Time, const double kBTNubyMNu, const double UnitVelocity_in_cm_per_s);

/*Amplitude of the random velocity for WDM*/
double
WDM_V0(const double Time, const double WDM_therm_mass, const double Omega_CDM, const double HubbleParam, const double UnitVelocity_in_cm_per_s);

unsigned int *
init_rng(int Seed, int Nmesh);


#endif
