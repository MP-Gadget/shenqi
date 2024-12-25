#define BOOST_TEST_MODULE timefac

#include "booststub.h"

#include <libgadget/timefac.h>
#include <boost/math/quadrature/tanh_sinh.hpp>
#include <boost/math/quadrature/gauss_kronrod.hpp>

#define AMIN 0.005
#define AMAX 1.0
#define LTIMEBINS 29

/*Hubble function at scale factor a, in dimensions of All.Hubble*/
double hubble_function(const Cosmology * CP, double a)
{
    double hubble_a;
    /* lambda and matter */
    hubble_a = 1 - CP->Omega0;
    hubble_a += CP->Omega0 / (a * a * a);
    /*Radiation*/
    hubble_a += 5.045e-5*(1+7./8.*pow(pow(4/11.,1/3.)*1.00328,4)*3) / (a * a * a * a);
    /* Hard-code default Gadget unit system. */
    hubble_a = 0.1 * sqrt(hubble_a);
    return (hubble_a);
}

/*Get integer from real time*/
double loga_from_ti(inttime_t ti)
{
    double logDTime = (log(AMAX) - log(AMIN)) / (1 << LTIMEBINS);
    return log(AMIN) + ti * logDTime;
}

/*Get integer from real time*/
static inline inttime_t get_ti(double aa)
{
    double logDTime = (log(AMAX) - log(AMIN)) / (1 << LTIMEBINS);
    return (log(aa) - log(AMIN))/logDTime;
}

double exact_drift_factor(const Cosmology * const CP, const double a1, const double a2, const int exp)
{
    auto integrand = [&CP, exp](double a) {
        const double h = hubble_function(CP, a);
        return 1 / (h * std::pow(a,exp));
    };
    // Perform the integration
    // double error;
    // result = boost::math::quadrature::gauss_kronrod<double, 61>::integrate(integrand, a1, a2, 5, boost::math::tools::root_epsilon<double>(), &error);
    // message(1, "exact error: %g\n", error);
    double result = boost::math::quadrature::gauss_kronrod<double, 61>::integrate(integrand, a1, a2, 15);
    return result;
}

BOOST_AUTO_TEST_CASE(test_drift_factor)
{
    /*Initialise the table: default values from z=200 to z=0*/
    Cosmology CP;
    CP.Omega0 = 1.;
    /* Check default scaling: for total matter domination
     * we should have a drift factor like 1/sqrt(a)*/
    BOOST_TEST(get_exact_drift_factor(&CP, get_ti(0.8), get_ti(0.85))  == - 2/0.1*(1/sqrt(0.85) - 1/sqrt(0.8)), tt::tolerance(6e-5));
    /*Test the kick table*/
    BOOST_TEST(get_exact_gravkick_factor(&CP, get_ti(0.8), get_ti(0.85)) == 2/0.1*(sqrt(0.85) - sqrt(0.8)), tt::tolerance(6e-5));

    //Chosen so we get the same bin
    BOOST_TEST(get_exact_drift_factor(&CP, get_ti(0.8), get_ti(0.8003)) == - 2/0.1*(1/sqrt(0.8003) - 1/sqrt(0.8)), tt::tolerance(6e-6));
    //Now choose a more realistic cosmology
    CP.Omega0 = 0.25;
    /*Check late and early times*/
    BOOST_TEST(get_exact_drift_factor(&CP, get_ti(0.95), get_ti(0.98)) == exact_drift_factor(&CP, 0.95, 0.98,3), tt::tolerance(5e-5));
    BOOST_TEST(get_exact_drift_factor(&CP, get_ti(0.05), get_ti(0.06)) == exact_drift_factor(&CP, 0.05, 0.06,3), tt::tolerance(5e-5));
    /*Check boundary conditions*/
    double logDtime = (log(AMAX)-log(AMIN))/(1<<LTIMEBINS);
    BOOST_TEST(get_exact_drift_factor(&CP, ((1<<LTIMEBINS)-1), 1<<LTIMEBINS) == exact_drift_factor(&CP, AMAX-logDtime, AMAX,3), tt::tolerance(5e-5));
    BOOST_TEST(get_exact_drift_factor(&CP, 0, 1) == exact_drift_factor(&CP, 1.0 - exp(log(AMAX)-log(AMIN))/(1<<LTIMEBINS), 1.0,3), tt::tolerance(0.4));
    /*Gravkick*/
    BOOST_TEST(get_exact_gravkick_factor(&CP, get_ti(0.8), get_ti(0.85)) == exact_drift_factor(&CP, 0.8, 0.85, 2), tt::tolerance(5e-5));
    BOOST_TEST(get_exact_gravkick_factor(&CP, get_ti(0.05), get_ti(0.06)) == exact_drift_factor(&CP, 0.05, 0.06, 2), tt::tolerance(5e-5));

    /*Test the hydrokick table: always the same as drift*/
    BOOST_TEST(get_exact_hydrokick_factor(&CP, get_ti(0.8), get_ti(0.85)) == get_exact_drift_factor(&CP, get_ti(0.8), get_ti(0.85)), tt::tolerance(5e-5));

}
