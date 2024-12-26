/*Tests for the thermal velocity module, ported from S-GenIC.*/
#define BOOST_TEST_MODULE thermal

#include "booststub.h"

#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <libgadget/config.h>
#include <libgenic/thermal.h>

/*Check that the neutrino velocity NU_V0 is sensible*/
BOOST_AUTO_TEST_CASE(test_mean_velocity)
{
    /*Check has units of velocity*/
    BOOST_TEST(NU_V0(1, 1, 1e3) == 100*NU_V0(1, 1, 1e5), tt::tolerance(1e-8));
    /*Check scales linearly with neutrino mass*/
    BOOST_TEST(10*NU_V0(1, 0.1, 1e5) == NU_V0(1, 1, 1e5), tt::tolerance(1e-8));
    /*Check scales as z (gadget's cosmological velocity unit is accounted for outside)*/
    BOOST_TEST(0.5*NU_V0(0.5, 1, 1e5) == NU_V0(1, 1, 1e5), tt::tolerance(1e-8));
}

BOOST_AUTO_TEST_CASE(test_thermal_vel)
{
    /*Seed table with velocity of 100 km/s*/
    struct thermalvel nu_vels;
    init_thermalvel(&nu_vels, 100, 5000/100, 0);

    /*Test getting the distribution*/
    BOOST_TEST(nu_vels.fermi_dirac_vel[0] == 0);
    BOOST_TEST(nu_vels.fermi_dirac_vel[LENGTH_FERMI_DIRAC_TABLE - 1] ==  MAX_FERMI_DIRAC);

    /*Number verified by mathematica*/
    int ii = 0;
    while(nu_vels.fermi_dirac_cumprob[ii] < 0.5) {
        ii++;
    }
    BOOST_TEST(nu_vels.fermi_dirac_vel[ii] == 2.839075, tt::tolerance(0.002));
    /*Check some statistical properties (max, min, mean)*/
    double mean = 0;
    double max = 0;
    double min = 1e10;
    int nsample;
    float Vel[3] = {0};
    int64_t MaxID = 100000;
    gsl_rng * g_rng = gsl_rng_alloc(gsl_rng_ranlxd1);
    for (nsample=0; nsample < MaxID; nsample++)
    {
        add_thermal_speeds(&nu_vels, g_rng, Vel);
        double v2 = sqrt(Vel[0]*Vel[0]+Vel[1]*Vel[1]+Vel[2]*Vel[2]);
        if(v2 > max)
            max = v2;
        if(v2 < min)
            min = v2;
        mean+=v2;
        memset(Vel, 0, 3*sizeof(float));
    }
    gsl_rng_free(g_rng);
    mean/=nsample;
    /*Mean should be roughly 3*zeta(4)/zeta(3)*7/8/(3/4)* m_vamp*/
    BOOST_TEST(fabs(mean - 3*pow(M_PI,4)/90./1.202057*(7./8)/(3/4.)*100) < 1);
    BOOST_TEST(min > 0);
    BOOST_TEST( max < MAX_FERMI_DIRAC*100);
}
