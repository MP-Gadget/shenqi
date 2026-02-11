#include "physconst.h"
#include "timefac.h"
#include "timebinmgr.h"
#include <functional>
#include <boost/math/quadrature/gauss_kronrod.hpp>

// Function to compute a factor using Gauss-Kronrod adaptive integration
static double get_exact_factor(const Cosmology * const CP, const inttime_t t0, const inttime_t t1, const std::function<double(double)> func)
{
    if (t0 == t1)
        return 0;

    // Calculate the scale factors
    const double a0 = std::exp(loga_from_ti(t0));
    const double a1 = std::exp(loga_from_ti(t1));
    // Gauss-Kronrod integration for smooth functions. Boost uses by default the machine precision for accuracy and a max depth of 15.
    return boost::math::quadrature::gauss_kronrod<double, 61>::integrate(func, a0, a1);
}

/*Get the exact drift factor*/
double get_exact_drift_factor(Cosmology * CP, inttime_t ti0, inttime_t ti1)
{
    // Define the integrand as a lambda function, for the drift table.
    auto drift_integ = [CP](const double a) {
        double h = hubble_function(CP, a);
        return 1 / (h * a * a * a);
    };
    return get_exact_factor(CP, ti0, ti1, drift_integ);
}

/*Get the exact drift factor*/
double get_exact_gravkick_factor(Cosmology * CP, inttime_t ti0, inttime_t ti1)
{
    /* Integrand for the gravkick table*/
    auto gravkick_integ = [CP](const double a) {
        double h = hubble_function(CP, a);
        return 1 / (h * a * a);
    };
    return get_exact_factor(CP, ti0, ti1, gravkick_integ);
}

double get_exact_hydrokick_factor(Cosmology * CP, inttime_t ti0, inttime_t ti1)
{
    /* Integrand for the hydrokick table.
     * Note this is the same function as drift.*/
    auto hydrokick_integ = [CP](const double a) {
        double h = hubble_function(CP, a);
        return 1 / (h * pow(a, 3 * GAMMA_MINUS1) * a);
    };
    return get_exact_factor(CP, ti0, ti1, hydrokick_integ);
}

/* Function to compute comoving distance using the adaptive integrator */
double compute_comoving_distance(Cosmology *CP, double a0, double a1, const double UnitVelocity_in_cm_per_s)
{
    // relative error tolerance
    // double epsrel = 1e-8;
    /* Integrand for comoving distance */
    auto comoving_distance_integ = [CP](double a) {
        double h = hubble_function(CP, a);
        return 1. / (h * a * a);
    };

    // Call the generic adaptive integration function
    const double result = boost::math::quadrature::gauss_kronrod<double, 61>::integrate(comoving_distance_integ, a0, a1);
    // Convert the result using the provided units
    return (LIGHTCGS / UnitVelocity_in_cm_per_s) * result;
}
