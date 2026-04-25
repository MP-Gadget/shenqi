/*Tests for the timebinmgr module, to ensure we got the bitwise arithmetic right.*/
#define BOOST_TEST_MODULE timebinmgr

#include "booststub.h"
#include <libgadget/timebinmgr.h>

#define TIMEMAX 1.0
#define TIMEIC 0.1
#define OutputListLength 4
double outs[4] = {TIMEIC, 0.2, 0.8, TIMEMAX};
double logouts[4];

static void
setup(void)
{
    int i;
    for(i = 0; i < OutputListLength; i ++)
        logouts[i] = log(outs[i]);
    set_sync_params_test(OutputListLength, outs);
}

/*tbm has no state*/
/*First test conversions between float and integer timelines*/
BOOST_AUTO_TEST_CASE(test_conversions)
{
    setup();
    //Cosmology is used to init Excursion set syncpoints, they are not needed here since that flag is off
    TimeBinMgr tbm(NULL, TIMEIC, TIMEMAX, 0.0, false);

    /*Convert an integer to and from loga*/
    /* double loga_from_ti(unsigned int ti); */
    BOOST_TEST(tbm.loga_from_ti(0) == logouts[0]);
    BOOST_TEST(tbm.loga_from_ti(TIMEBASE) == logouts[1]);
    BOOST_TEST(tbm.loga_from_ti(TIMEBASE-1) == (logouts[0] + (logouts[1]-logouts[0])*(TIMEBASE-1)/TIMEBASE), tt::tolerance(1e-6));
    BOOST_TEST(tbm.loga_from_ti(TIMEBASE+1) == (logouts[1] + (logouts[2]-logouts[1])/TIMEBASE), tt::tolerance(1e-6));
    BOOST_TEST(tbm.loga_from_ti(2*TIMEBASE) == logouts[2]);
    /* unsigned int ti_from_loga(double loga); */
    BOOST_TEST(tbm.ti_from_loga(logouts[0]) == 0);
    BOOST_TEST(tbm.ti_from_loga(logouts[1]) == TIMEBASE);
    BOOST_TEST(tbm.ti_from_loga(logouts[2]) == 2*TIMEBASE);
    double midpt = (logouts[2] + logouts[1])/2;
    BOOST_TEST(tbm.ti_from_loga(midpt) == TIMEBASE+TIMEBASE/2);
    BOOST_TEST(tbm.loga_from_ti(TIMEBASE+TIMEBASE/2) == midpt, tt::tolerance(1e-6));

    /*Check behaviour past end*/
    BOOST_TEST(tbm.ti_from_loga(0) == 3*TIMEBASE);
    BOOST_TEST(tbm.loga_from_ti(tbm.ti_from_loga(log(0.1))) == log(0.1), tt::tolerance(1e-6));

    /*! this function returns the next output time after ti_curr.*/
    BOOST_TEST(tbm.find_next_sync_point(0)->ti == TIMEBASE);
    BOOST_TEST(tbm.find_next_sync_point(TIMEBASE)->ti == 2 * TIMEBASE);
    BOOST_TEST(tbm.find_next_sync_point(TIMEBASE-1)->ti == TIMEBASE);
    BOOST_TEST(tbm.find_next_sync_point(TIMEBASE+1)->ti == 2*TIMEBASE);
    BOOST_TEST(tbm.find_next_sync_point(4 * TIMEBASE) == nullptr);

    BOOST_TEST(tbm.find_current_sync_point(0)->ti == 0);
    BOOST_TEST(tbm.find_current_sync_point(TIMEBASE)->ti == TIMEBASE);
    BOOST_TEST(tbm.find_current_sync_point(-1)  == nullptr);
    BOOST_TEST(tbm.find_current_sync_point(TIMEBASE-1) == nullptr);

    BOOST_TEST(tbm.find_current_sync_point(0)->write_snapshot  == 1);
    BOOST_TEST(tbm.find_current_sync_point(TIMEBASE)->write_snapshot  == 1);
    BOOST_TEST(tbm.find_current_sync_point(2 * TIMEBASE)->write_snapshot  == 1);
    BOOST_TEST(tbm.find_current_sync_point(3 * TIMEBASE)->write_snapshot == 1);
}

BOOST_AUTO_TEST_CASE(test_skip_first)
{
    setup();

    TimeBinMgr tbm(NULL, TIMEIC, TIMEMAX, TIMEIC, false);
    BOOST_TEST(tbm.find_current_sync_point(0)->write_snapshot == 0);

    TimeBinMgr tbm2(NULL, TIMEIC, TIMEMAX, 0.0, false);
    BOOST_TEST(tbm2.find_current_sync_point(0)->write_snapshot == 1);
}

BOOST_AUTO_TEST_CASE(test_dloga)
{
    setup();

    TimeBinMgr tbm(NULL, TIMEIC, TIMEMAX, 0.0, 0);

    inttime_t Ti_Current = tbm.ti_from_loga(log(0.55));
    /* unsigned int dti_from_dloga(double loga); */
    /* double dloga_from_dti(unsigned int ti); */

    /*Get dloga from a timebin*/
    /* double get_dloga_for_bin(int timebin); */
    BOOST_TEST(tbm.get_dloga_for_bin(0, Ti_Current ) < 1e-6);
    BOOST_TEST(tbm.get_dloga_for_bin(TIMEBINS, Ti_Current ) == logouts[2]-logouts[1], tt::tolerance(1e-6));
    BOOST_TEST(tbm.get_dloga_for_bin(TIMEBINS-2, Ti_Current) == (logouts[2]-logouts[1])/4, tt::tolerance(1e-6));

    /*Enforce that an integer time is a power of two*/
    /* unsigned int round_down_power_of_two(unsigned int ti); */
    BOOST_TEST(round_down_power_of_two(TIMEBASE)==TIMEBASE);
    BOOST_TEST(round_down_power_of_two(TIMEBASE+1)==TIMEBASE);
    BOOST_TEST(round_down_power_of_two(TIMEBASE-1)==TIMEBASE/2);
}


#include <boost/math/quadrature/gauss_kronrod.hpp>

#define AMIN 0.005
#define AMAX 1.0

/*Get integer from real time*/
static inline inttime_t get_ti(double aa, TimeBinMgr * tbm)
{
    return tbm->ti_from_loga(log(aa));
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
    Cosmology CP = {0};
    CP.CMBTemperature = 2.7255;
    CP.Omega0 = 1.0;
    CP.OmegaLambda = 1- CP.Omega0;
    CP.OmegaBaryon = 0.0455;
    CP.HubbleParam = 0.7;
    CP.RadiationOn = 1;
    CP.Omega_fld = 0; /*Energy density of dark energy fluid at z=0*/
    CP.w0_fld = -1; /*Dark energy equation of state parameter*/
    CP.wa_fld = 0; /*Dark energy equation of state evolution parameter*/
    CP.Omega_ur = 0;
    CP.MNu[0] = CP.MNu[1] = CP.MNu[2] = 0;

    struct UnitSystem units = get_unitsystem(3.085678e21, 1.989e43, 1e5);
    init_cosmology(&CP, AMIN, units);

    /*Initialise the table: default values from z=200 to z=0*/
    TimeBinMgr tbm(&CP, AMIN, AMAX, 0.0, false);

    /* Check default scaling: for total matter domination
     * we should have a drift factor like 1/sqrt(a)*/
    BOOST_TEST(tbm.get_exact_drift_factor(get_ti(0.8, &tbm), get_ti(0.85, &tbm))  == - 2/0.1*(1/sqrt(0.85) - 1/sqrt(0.8)), tt::tolerance(6e-5));
    /*Test the kick table*/
    BOOST_TEST(tbm.get_exact_gravkick_factor(get_ti(0.8, &tbm), get_ti(0.85, &tbm)) == 2/0.1*(sqrt(0.85) - sqrt(0.8)), tt::tolerance(6e-5));

    //Chosen so we get the same bin
    BOOST_TEST(tbm.get_exact_drift_factor(get_ti(0.8, &tbm), get_ti(0.8003, &tbm)) == - 2/0.1*(1/sqrt(0.8003) - 1/sqrt(0.8)), tt::tolerance(6e-5));

    //Now choose a more realistic cosmology
    CP.Omega0 = 0.25;
    CP.OmegaLambda = 1- CP.Omega0;
    init_cosmology(&CP, AMIN, units);
    TimeBinMgr tbm2(&CP, AMIN, AMAX, 0.0, false);

    /*Check late and early times*/
    BOOST_TEST(tbm2.get_exact_drift_factor(get_ti(0.95, &tbm2), get_ti(0.98, &tbm2)) == exact_drift_factor(&CP, 0.95, 0.98,3), tt::tolerance(5e-5));
    BOOST_TEST(tbm2.get_exact_drift_factor(get_ti(0.05, &tbm2), get_ti(0.06, &tbm2)) == exact_drift_factor(&CP, 0.05, 0.06,3), tt::tolerance(5e-5));
    /*Gravkick*/
    BOOST_TEST(tbm2.get_exact_gravkick_factor(get_ti(0.8, &tbm2), get_ti(0.85, &tbm2)) == exact_drift_factor(&CP, 0.8, 0.85, 2), tt::tolerance(5e-5));
    BOOST_TEST(tbm2.get_exact_gravkick_factor(get_ti(0.05, &tbm2), get_ti(0.06, &tbm2)) == exact_drift_factor(&CP, 0.05, 0.06, 2), tt::tolerance(5e-5));

    /*Test the hydrokick table: always the same as drift*/
    BOOST_TEST(tbm2.get_exact_hydrokick_factor(get_ti(0.8, &tbm2), get_ti(0.85, &tbm2)) == tbm2.get_exact_drift_factor(get_ti(0.8, &tbm2), get_ti(0.85, &tbm2)), tt::tolerance(5e-5));
}
