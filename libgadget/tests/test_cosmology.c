/*Tests for the cosmology module, ported from N-GenIC.*/
#define BOOST_TEST_MODULE cosmology

#include "booststub.h"

#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <boost/math/special_functions/hypergeometric_pFq.hpp>
#include <libgadget/physconst.h>
#include <libgadget/cosmology.h>

/*Get integer from real time*/
double loga_from_ti(int64_t ti)
{
    return ti;
}

/*Neutrinos are tested elsewhere*/
void init_omega_nu(_omega_nu * const omnu, const double MNu[], const double a0, const double HubbleParam, const double tcmb0) {}

double get_omega_nu(const _omega_nu * omnu, const double a)
{
    return 0;
}
double get_omega_nu_nopart(const _omega_nu * omnu, const double a)
{
    return 0;
}

void init_hybrid_nu(_hybrid_nu * const hybnu, const double mnu[], const double vcrit, const double light, const double nu_crit_time, const double kBtnu){ }

void setup_cosmology(Cosmology * CP, double Omega0, double OmegaBaryon, double H0)
{
    CP->CMBTemperature = 2.7255;
    CP->Omega0 = Omega0;
    CP->OmegaLambda = 1- CP->Omega0;
    CP->OmegaBaryon = OmegaBaryon;
    CP->HubbleParam = H0;
    CP->RadiationOn = 1;
    CP->Omega_fld = 0; /*Energy density of dark energy fluid at z=0*/
    CP->w0_fld = -1; /*Dark energy equation of state parameter*/
    CP->wa_fld = 0; /*Dark energy equation of state evolution parameter*/
    CP->Omega_ur = 0;
    CP->MNu[0] = CP->MNu[1] = CP->MNu[2] = 0;
    struct UnitSystem units = get_unitsystem(3.085678e21, 1.989e43, 1e5);
    /*Do the main cosmology initialisation*/
    init_cosmology(CP,0.01, units);
}

static inline double radgrow(double aa, double omegar) {
    return omegar + 1.5 * 1. * aa;
}

//Omega_L + Omega_M = 1 => D+ ~ Gauss hypergeometric function
static inline double growth(double aa, double omegam) {
    double omegal = 1 - omegam;
    return aa * boost::math::hypergeometric_pFq({1./3, 1.}, {11./6.}, -omegal/omegam * pow(aa, 3));
}

BOOST_AUTO_TEST_CASE(test_cosmology)
{
    Cosmology CP = {0};
    //Check that we get the right scalings for total matter domination.
    //Cosmology(double HubbleParam, double Omega, double OmegaLambda, double MNu, int Hierarchy, bool NoRadiation)
    setup_cosmology(&CP, 1., 0.0455, 0.7);
    /*Check some of the setup worked*/
    BOOST_TEST(CP.OmegaK < 1e-5);
    BOOST_TEST(CP.OmegaG == 5.045e-5, tt::tolerance(2e-3));
    /*Check the hubble function is sane*/
    CP.RadiationOn = 0;
    BOOST_TEST(hubble_function(&CP, 1) == CP.Hubble, tt::tolerance(1e-5));
    CP.RadiationOn = 1;
    BOOST_TEST(hubble_function(&CP, 1) == CP.Hubble* sqrt(1+CP.OmegaG), tt::tolerance(1e-7));

    BOOST_TEST(CP.Hubble == 0.1, tt::tolerance(1e-6));
    BOOST_TEST(hubble_function(&CP, 1) == CP.Hubble, tt::tolerance(3e-5));
    BOOST_TEST(hubble_function(&CP, 0.1) == hubble_function(&CP, 1)/pow(0.1,3/2.), tt::tolerance(1e-2));
    BOOST_TEST(GrowthFactor(&CP, 0.5,1.) == 0.5, tt::tolerance(2e-4));
    //Check that the velocity correction d ln D1/d lna is constant
    BOOST_TEST(1.0 == F_Omega(&CP, 1.5), tt::tolerance(1e-1));
    BOOST_TEST(1.0 == F_Omega(&CP, 2) , tt::tolerance(1e-2));
    //Check radiation against exact solution from gr-qc/0504089
    BOOST_TEST(1/GrowthFactor(&CP, 0.05,1.) == radgrow(1., CP.OmegaG)/radgrow(0.05, CP.OmegaG), tt::tolerance(1e-3));
    BOOST_TEST(GrowthFactor(&CP, 0.01,0.001) == radgrow(0.01, CP.OmegaG)/radgrow(0.001, CP.OmegaG), tt::tolerance(1e-3));

    //Check against exact solutions from gr-qc/0504089: No radiation!
    //Note that the GSL hyperg needs the last argument to be < 1
    double omegam = 0.5;
    setup_cosmology(&CP, omegam, 0.0455, 0.7);
    CP.RadiationOn = 0;
    //Check growth factor during matter domination
    BOOST_TEST(1/GrowthFactor(&CP, 0.5, 1.) == growth(1., omegam)/growth(0.5, omegam), tt::tolerance(1e-3));
    BOOST_TEST(GrowthFactor(&CP, 0.3, 0.15) == growth(0.3, omegam)/growth(0.15, omegam), tt::tolerance(1e-3));
    BOOST_TEST(1/GrowthFactor(&CP, 0.01, 1.) == growth(1, omegam)/growth(0.01, omegam), tt::tolerance(1e-3));
    BOOST_TEST(0.01*log(GrowthFactor(&CP, 0.01+1e-5,0.01-1e-5))/2e-5 == F_Omega(&CP, 0.01), tt::tolerance(1e-3));
}
