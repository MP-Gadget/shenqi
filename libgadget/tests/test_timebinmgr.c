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

/*timebinmgr has no state*/
/*First test conversions between float and integer timelines*/
BOOST_AUTO_TEST_CASE(test_conversions)
{
    setup();
    //Cosmology is used to init Excursion set syncpoints, they are not needed here since that flag is off
    setup_sync_points(NULL, TIMEIC, TIMEMAX, 0.0, 0);

    /*Convert an integer to and from loga*/
    /* double loga_from_ti(unsigned int ti); */
    BOOST_TEST(loga_from_ti(0) == logouts[0]);
    BOOST_TEST(loga_from_ti(TIMEBASE) == logouts[1]);
    BOOST_TEST(loga_from_ti(TIMEBASE-1) == (logouts[0] + (logouts[1]-logouts[0])*(TIMEBASE-1)/TIMEBASE), tt::tolerance(1e-6));
    BOOST_TEST(loga_from_ti(TIMEBASE+1) == (logouts[1] + (logouts[2]-logouts[1])/TIMEBASE), tt::tolerance(1e-6));
    BOOST_TEST(loga_from_ti(2*TIMEBASE) == logouts[2]);
    /* unsigned int ti_from_loga(double loga); */
    BOOST_TEST(ti_from_loga(logouts[0]) == 0);
    BOOST_TEST(ti_from_loga(logouts[1]) == TIMEBASE);
    BOOST_TEST(ti_from_loga(logouts[2]) == 2*TIMEBASE);
    double midpt = (logouts[2] + logouts[1])/2;
    BOOST_TEST(ti_from_loga(midpt) == TIMEBASE+TIMEBASE/2);
    BOOST_TEST(loga_from_ti(TIMEBASE+TIMEBASE/2) == midpt, tt::tolerance(1e-6));

    /*Check behaviour past end*/
    BOOST_TEST(ti_from_loga(0) == 3*TIMEBASE);
    BOOST_TEST(loga_from_ti(ti_from_loga(log(0.1))) == log(0.1), tt::tolerance(1e-6));

    /*! this function returns the next output time after ti_curr.*/
    BOOST_TEST(find_next_sync_point(0)->ti == TIMEBASE);
    BOOST_TEST(find_next_sync_point(TIMEBASE)->ti == 2 * TIMEBASE);
    BOOST_TEST(find_next_sync_point(TIMEBASE-1)->ti == TIMEBASE);
    BOOST_TEST(find_next_sync_point(TIMEBASE+1)->ti == 2*TIMEBASE);
    BOOST_TEST(find_next_sync_point(4 * TIMEBASE) == nullptr);

    BOOST_TEST(find_current_sync_point(0)->ti == 0);
    BOOST_TEST(find_current_sync_point(TIMEBASE)->ti == TIMEBASE);
    BOOST_TEST(find_current_sync_point(-1)  == nullptr);
    BOOST_TEST(find_current_sync_point(TIMEBASE-1) == nullptr);

    BOOST_TEST(find_current_sync_point(0)->write_snapshot  == 1);
    BOOST_TEST(find_current_sync_point(TIMEBASE)->write_snapshot  == 1);
    BOOST_TEST(find_current_sync_point(2 * TIMEBASE)->write_snapshot  == 1);
    BOOST_TEST(find_current_sync_point(3 * TIMEBASE)->write_snapshot == 1);
}

BOOST_AUTO_TEST_CASE(test_skip_first)
{
    setup();
    setup_sync_points(NULL, TIMEIC, TIMEMAX, TIMEIC, 0);
    BOOST_TEST(find_current_sync_point(0)->write_snapshot == 0);

    setup_sync_points(NULL, TIMEIC, TIMEMAX, 0.0, 0);
    BOOST_TEST(find_current_sync_point(0)->write_snapshot == 1);
}

BOOST_AUTO_TEST_CASE(test_dloga)
{
    setup();

    setup_sync_points(NULL, TIMEIC, TIMEMAX, 0.0, 0);

    inttime_t Ti_Current = ti_from_loga(log(0.55));
    /* unsigned int dti_from_dloga(double loga); */
    /* double dloga_from_dti(unsigned int ti); */

    /*Get dloga from a timebin*/
    /* double get_dloga_for_bin(int timebin); */
    BOOST_TEST(get_dloga_for_bin(0, Ti_Current ) < 1e-6);
    BOOST_TEST(get_dloga_for_bin(TIMEBINS, Ti_Current ) == logouts[2]-logouts[1], tt::tolerance(1e-6));
    BOOST_TEST(get_dloga_for_bin(TIMEBINS-2, Ti_Current) == (logouts[2]-logouts[1])/4, tt::tolerance(1e-6));

    /*Enforce that an integer time is a power of two*/
    /* unsigned int round_down_power_of_two(unsigned int ti); */
    BOOST_TEST(round_down_power_of_two(TIMEBASE)==TIMEBASE);
    BOOST_TEST(round_down_power_of_two(TIMEBASE+1)==TIMEBASE);
    BOOST_TEST(round_down_power_of_two(TIMEBASE-1)==TIMEBASE/2);
}
