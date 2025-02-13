#define BOOST_TEST_MODULE omega_nu_single

#include "booststub.h"
#include <libgadget/omega_nu_single.h>
#include <libgadget/physconst.h>
#include <libgadget/timefac.h>
#include <boost/math/quadrature/gauss_kronrod.hpp>

#define  T_CMB0      2.7255	/* present-day CMB temperature, from Fixsen 2009 */

/* A test case that checks initialisation. */
BOOST_AUTO_TEST_CASE(test_rho_nu_init)
{
    double mnu = 0.06;
    _rho_nu_single rho_nu_tab;
    /*Initialise*/
    rho_nu_init(&rho_nu_tab, 0.01, mnu, BOLEVK*TNUCMB*T_CMB0);
    /*Check everything initialised ok*/
    BOOST_TEST(rho_nu_tab.mnu == mnu);
    BOOST_TEST(rho_nu_tab.interp);
}

/*Check massless neutrinos work*/
#define STEFAN_BOLTZMANN 5.670373e-5
#define OMEGAR (4*STEFAN_BOLTZMANN*8*M_PI*GRAVITY/(3*LIGHTCGS*LIGHTCGS*LIGHTCGS*HUBBLE*HUBBLE*HubbleParam*HubbleParam)*pow(T_CMB0,4))

/* Check that the table gives the right answer. */
BOOST_AUTO_TEST_CASE(test_omega_nu_single)
{
    double mnu = 0.5;
    double HubbleParam = 0.7;
    _omega_nu omnu;
    /*Initialise*/
    double MNu[3] = {mnu, mnu, 0};
    init_omega_nu(&omnu, MNu, 0.01, 0.7,T_CMB0);
    BOOST_TEST(omnu.RhoNuTab[0].mnu == mnu);
    /*This is the critical density at z=0:
     * we allow it to change by what is (at the time of writing) the uncertainty on G.*/
    BOOST_TEST(omnu.rhocrit == 1.8784e-29*HubbleParam*HubbleParam, tt::tolerance(5e-3));
    /*Check everything initialised ok*/
    double omnuz0 = omega_nu_single(&omnu, 1, 0);
    /*Check redshift scaling*/
    BOOST_TEST(omnuz0/pow(0.5,3) == omega_nu_single(&omnu, 0.5, 0), tt::tolerance(5e-3));
    /*Check not just a^-3 scaling*/
    BOOST_TEST(omnuz0/pow(0.01,3) <  omega_nu_single(&omnu, 0.01, 0));
    /*Check that we have correctly accounted for neutrino decoupling*/
    BOOST_TEST(omnuz0 == mnu/93.14/HubbleParam/HubbleParam, tt::tolerance(1e-3));
    /*Check degenerate species works*/
    BOOST_TEST(omega_nu_single(&omnu, 0.1, 1) == omega_nu_single(&omnu, 0.1, 0));
    /*Check we get it right for massless neutrinos*/
    double omnunomassz0 = omega_nu_single(&omnu, 1, 2);
    BOOST_TEST(omnunomassz0 == OMEGAR*7./8.*pow(pow(4/11.,1/3.)*1.00381,4), tt::tolerance(5e-3));
    BOOST_TEST(omnunomassz0/pow(0.5,4) == omega_nu_single(&omnu, 0.5, 2));
    /*Check that we return something vaguely sensible for very early times*/
    BOOST_TEST(omega_nu_single(&omnu,1e-4,0) > omega_nu_single(&omnu, 1,0)/pow(1e-4,3));
}


double get_rho_nu_conversion();

/*Note q carries units of eV/c. kT/c has units of eV/c.
 * M_nu has units of eV  Here c=1. */
double rho_nu_int(double q, void * params);

double do_exact_rho_nu_integration(double a, double mnu, double rhocrit)
{
    double kTnu = BOLEVK*TNUCMB*T_CMB0;
    double amnu = mnu * a;

    auto rho_nu_int = [amnu, kTnu](const double q) {
            double epsilon = sqrt(q*q+amnu*amnu);
            double f0 = 1./(exp(q/kTnu)+1);
            return q*q*epsilon*f0;
    };

    // Oscillatory integral!
    double result = boost::math::quadrature::gauss_kronrod<double, 61>::integrate(rho_nu_int, 0, 500 * kTnu);
    result *= get_rho_nu_conversion()/pow(a,4)/rhocrit;
    return result;
}

/*Check exact integration against the interpolation table*/
BOOST_AUTO_TEST_CASE(test_omega_nu_single_exact)
{
    double mnu = 0.05;
    double hubble = 0.7;
    int i;
    _omega_nu omnu;
    /*Initialise*/
    double MNu[3] = {mnu, mnu, mnu};
    init_omega_nu(&omnu, MNu, 0.01, hubble,T_CMB0);
    double omnuz0 = omega_nu_single(&omnu, 1, 0);
    double rhocrit = omnu.rhocrit;
    BOOST_TEST(omnuz0 == do_exact_rho_nu_integration(1, mnu, rhocrit), tt::tolerance(1e-6));
    for(i=1; i< 123; i++) {
        double a = 0.01 + i/123.;
        omnuz0 = omega_nu_single(&omnu, a, 0);
        double omexact = do_exact_rho_nu_integration(a, mnu, rhocrit);
        if(fabs(omnuz0 - omexact) > 1e-6 * omnuz0)
            printf("a=%g %g %g %g\n",a, omnuz0, omexact, omnuz0/omexact-1);
        BOOST_TEST(omexact == omnuz0, tt::tolerance(1e-6));
    }
}

BOOST_AUTO_TEST_CASE(test_omega_nu_init_degenerate)
{
    /*Check we correctly initialise omega_nu with degenerate neutrinos*/
    _omega_nu omnu;
    /*Initialise*/
    double MNu[3] = {0.2,0.2,0.2};
    init_omega_nu(&omnu, MNu, 0.01, 0.7,T_CMB0);
    /*Check that we initialised the right number of arrays*/
    BOOST_TEST(omnu.nu_degeneracies[0] == 3);
    BOOST_TEST(omnu.nu_degeneracies[1] == 0);
    BOOST_TEST(omnu.RhoNuTab[0].interp);
    BOOST_TEST(omnu.RhoNuTab[1].interp == nullptr);
}

BOOST_AUTO_TEST_CASE(test_omega_nu_init_nondeg)
{
    /*Now check that it works with a non-degenerate set*/
    _omega_nu omnu;
    /*Initialise*/
    double MNu[3] = {0.2,0.1,0.3};
    int i;
    init_omega_nu(&omnu, MNu, 0.01, 0.7,T_CMB0);
    /*Check that we initialised the right number of arrays*/
    for(i=0; i<3; i++) {
        BOOST_TEST(omnu.nu_degeneracies[i] == 1);
        BOOST_TEST(omnu.RhoNuTab[i].interp);
    }
}

BOOST_AUTO_TEST_CASE(test_get_omega_nu)
{
    /*Check that we get the right answer from get_omega_nu, in terms of rho_nu*/
    _omega_nu omnu;
    /*Initialise*/
    double MNu[3] = {0.2,0.1,0.3};
    init_omega_nu(&omnu, MNu, 0.01, 0.7,T_CMB0);
    double total =0;
    int i;
    for(i=0; i<3; i++) {
        total += omega_nu_single(&omnu, 0.5, i);
    }
    BOOST_TEST(get_omega_nu(&omnu, 0.5) == total, tt::tolerance(1e-6));
}

BOOST_AUTO_TEST_CASE(test_get_omegag)
{
    /*Check that we get the right answer from get_omegag*/
    _omega_nu omnu;
    /*Initialise*/
    double MNu[3] = {0.2,0.1,0.3};
    const double HubbleParam = 0.7;
    init_omega_nu(&omnu, MNu, 0.01, HubbleParam,T_CMB0);
    const double omegag = OMEGAR/pow(0.5,4);
    BOOST_TEST(get_omegag(&omnu, 0.5) == omegag, tt::tolerance(1e-6));
}

/*Test integrate the fermi-dirac kernel between 0 and qc*/
BOOST_AUTO_TEST_CASE(test_nufrac_low)
{
    BOOST_TEST(nufrac_low(0) == 0);
    /*Mathematica integral: 1.*Integrate[x*x/(Exp[x] + 1), {x, 0, 0.5}]/(3*Zeta[3]/2)*/
    BOOST_TEST(nufrac_low(1) == 0.0595634, tt::tolerance(1e-5));
    BOOST_TEST(nufrac_low(0.5) == 0.00941738, tt::tolerance(1e-5));
}

BOOST_AUTO_TEST_CASE(test_hybrid_neutrinos)
{
    /*Check that we get the right answer from get_omegag*/
    _omega_nu omnu;
    /*Initialise*/
    double MNu[3] = {0.2,0.2,0.2};
    const double HubbleParam = 0.7;
    init_omega_nu(&omnu, MNu, 0.01, HubbleParam,T_CMB0);
    init_hybrid_nu(&omnu.hybnu, MNu, 700, 299792, 0.5,omnu.kBtnu);
    /*Check that the fraction of omega change over the jump*/
    double nufrac_part = nufrac_low(700/299792.*0.2/omnu.kBtnu);
    BOOST_TEST(particle_nu_fraction(&omnu.hybnu, 0.50001, 0) == nufrac_part, tt::tolerance(1e-5));
    BOOST_TEST(particle_nu_fraction(&omnu.hybnu, 0.49999, 0) == 0);
    BOOST_TEST(get_omega_nu_nopart(&omnu, 0.499999)*(1-nufrac_part) == get_omega_nu_nopart(&omnu, 0.500001), tt::tolerance(1e-4));
    /*Ditto omega_nu_single*/
    BOOST_TEST(omega_nu_single(&omnu, 0.499999, 0)*(1-nufrac_part) == omega_nu_single(&omnu, 0.500001, 0), tt::tolerance(1e-4));
}
