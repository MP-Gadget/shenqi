#ifndef TIMEBINMGR_H
#define TIMEBINMGR_H

#include <string>
/* This file manages the integer timeline,
 * and converts from integers ti to double loga.*/

/*!< The simulated timespan is mapped onto the integer interval [0,TIMEBASE],
 *   where TIMEBASE needs to be a power of 2. Note that (1<<28) corresponds
 *   to 2^29.
 *   We allow some bits at the top of the integer timeline for snapshot outputs.
 *   Note that because each snapshot uses TIMEBASE on the integer timeline, the conversion
 *   factor between loga and ti is not constant across snapshots.
 */
#define TIMEBINS 46
#define TIMEBASE (1Lu<<TIMEBINS)
#define MAXSNAPSHOTS (1Lu<<(62-TIMEBINS))

#include "types.h"
#include "utils/paramset.h"
#include "cosmology.h"
#include "physconst.h"
#include <functional>
#include <boost/math/quadrature/gauss_kronrod.hpp>

typedef struct SyncPoint SyncPoint;

struct SyncPoint
{
    double a;
    double loga;
    bool write_snapshot;
    bool write_fof;
    bool calc_uvbg;  //! Calculate the UV background
    bool write_plane;  //! Write a plane
    int plane_snapnum;  //! The snapshot number for the plane
    inttime_t ti;
};

/*Get the dti from the timebin*/
MYCUDAFN static inline inttime_t dti_from_timebin(int bin) {
    /*Casts to work around bug in intel compiler 18.0*/
    return bin > 0 ? (1Lu << (uint64_t) bin) : 0;
}

/*! table with desired sync points. All forces and phase space variables are synchonized to the same order. */
class TimeBinMgr {
    public:
    Cosmology * CP;
    SyncPoint * SyncPoints;
    int64_t NSyncPoints;    /* number of times stored in table of desired sync points */

    TimeBinMgr (Cosmology * CP, double TimeIC, double TimeMax, double no_snapshot_until_time, bool SnapshotWithFOF);

    TimeBinMgr (): CP(NULL), SyncPoints(NULL), NSyncPoints(0) {};

    /*! this function returns the next output time that is in the future of
    *  ti_curr; if none is find it return NULL, indication the run shall terminate.
    */
    SyncPoint *
    find_next_sync_point(inttime_t ti)
    {
        int64_t i;
        for(i = 0; i < NSyncPoints; i ++) {
            if(SyncPoints[i].ti > ti) {
                return &SyncPoints[i];
            }
        }
        return NULL;
    }

    /* This function finds if ti is a sync point; if so returns the sync point;
    * otherwise, NULL. We check if we shall write a snapshot with this. */
    SyncPoint *
    find_current_sync_point(inttime_t ti)
    {
        int64_t i;
        for(i = 0; i < NSyncPoints; i ++) {
            if(SyncPoints[i].ti == ti) {
                return &SyncPoints[i];
            }
        }
        return NULL;
    }

    /*Convert an integer to and from loga*/
    MYCUDAFN double
    loga_from_ti(inttime_t ti)
    {
        inttime_t lastsnap = ti >> TIMEBINS;
        if(lastsnap > NSyncPoints) {
            lastsnap = NSyncPoints -1;
        }
        double last = SyncPoints[lastsnap].loga;
        inttime_t dti = ti & (TIMEBASE - 1);
        double logDTime = Dloga_interval_ti(ti);
        return last + dti * logDTime;
    }

    MYCUDAFN inttime_t
    ti_from_loga(double loga)
    {
        inttime_t i, ti;
        /* First syncpoint is simulation start*/
        for(i = 1; i < NSyncPoints - 1; i++)
        {
            if(SyncPoints[i].loga > loga)
                break;
        }
        /*If loop didn't trigger, i == All.NSyncPointTimes-1*/
        double logDTime = (SyncPoints[i].loga - SyncPoints[i-1].loga)/TIMEBASE;
        ti = (i-1) << TIMEBINS;
        /* Note this means if we overrun the end of the timeline,
        * we still get something reasonable*/
        ti += (loga - SyncPoints[i-1].loga)/logDTime;
        return ti;
    }

    /*Convert changes in loga to and from ti*/
    MYCUDAFN inttime_t
    dti_from_dloga(double loga, const inttime_t Ti_Current)
    {
        inttime_t ti = ti_from_loga(loga_from_ti(Ti_Current));
        inttime_t tip = ti_from_loga(loga+loga_from_ti(Ti_Current));
        return tip - ti;
    }

    MYCUDAFN double dloga_from_dti(inttime_t dti, const inttime_t Ti_Current)
    {
        double Dloga = Dloga_interval_ti(Ti_Current);
        int sign = 1;
        if(dti < 0) {
            dti = -dti;
            sign = -1;
        }
        if((uint64_t) dti > TIMEBASE) {
            dti = TIMEBASE;
        }
        return Dloga * dti * sign;
    }
    /*Get dloga from a timebin*/
    MYCUDAFN double get_dloga_for_bin(int timebin, const inttime_t Ti_Current)
    {
        double logDTime = Dloga_interval_ti(Ti_Current);
        return dti_from_timebin(timebin) * logDTime;
    }

    /* Get the current scale factor*/
    MYCUDAFN double
    get_atime(const inttime_t Ti_Current) {
        return exp(loga_from_ti(Ti_Current));
    }

    /*Get the exact drift factor*/
    double get_exact_drift_factor(inttime_t ti0, inttime_t ti1)
    {
        Cosmology *CP = this->CP;
        // Define the integrand as a lambda function, for the drift table.
        auto drift_integ = [CP](const double a) {
            double h = hubble_function(CP, a);
            return 1 / (h * a * a * a);
        };
        return get_exact_factor(ti0, ti1, drift_integ);
    }

    /*Get the exact drift factor*/
    double get_exact_gravkick_factor(inttime_t ti0, inttime_t ti1)
    {
        Cosmology *CP = this->CP;
        /* Integrand for the gravkick table*/
        auto gravkick_integ = [CP](const double a) {
            double h = hubble_function(CP, a);
            return 1 / (h * a * a);
        };
        return get_exact_factor(ti0, ti1, gravkick_integ);
    }

    double get_exact_hydrokick_factor(inttime_t ti0, inttime_t ti1)
    {
        Cosmology *CP = this->CP;
        /* Integrand for the hydrokick table.
        * Note this is the same function as drift.*/
        auto hydrokick_integ = [CP](const double a) {
            double h = hubble_function(CP, a);
            return 1 / (h * pow(a, 3 * GAMMA_MINUS1) * a);
        };
        return get_exact_factor(ti0, ti1, hydrokick_integ);
    }

    private:
    /* Each integer time stores in the first 10 bits the snapshot number.
    * Then the rest of the bits are the standard integer timeline,
    * which should be a power-of-two hierarchy. We use this bit trick to speed up
    * the dloga look up. But the additional math makes this quite fragile. */

    /*Gets Dloga / ti for the current integer timeline.
    * Valid up to the next snapshot, after which it will change*/
    MYCUDAFN double Dloga_interval_ti(inttime_t ti)
    {
        /* FIXME: This uses the bit tricks because it has to be fast
        * -- till we clean up the calls to loga_from_ti; then we can avoid bit tricks. */

        inttime_t lastsnap = ti >> TIMEBINS;

        if(lastsnap >= NSyncPoints - 1) {
            /* stop advancing loga after the last sync point. */
            return 0;
        }
        double lastoutput = SyncPoints[lastsnap].loga;
        return (SyncPoints[lastsnap+1].loga - lastoutput)/TIMEBASE;
    }

    // Function to compute a factor using Gauss-Kronrod adaptive integration
    double get_exact_factor(const inttime_t t0, const inttime_t t1, const std::function<double(double)> func)
    {
        if (t0 == t1)
            return 0;

        // Calculate the scale factors
        const double a0 = std::exp(loga_from_ti(t0));
        const double a1 = std::exp(loga_from_ti(t1));
        // Gauss-Kronrod integration for smooth functions. Boost uses by default the machine precision for accuracy and a max depth of 15.
        return boost::math::quadrature::gauss_kronrod<double, 61>::integrate(func, a0, a1);
    }
};

/* Function to compute comoving distance using the adaptive integrator */
double compute_comoving_distance(Cosmology * CP, double a0, double a1, const double UnitVelocity_in_cm_per_s);

/* Enforce that an integer timestep is a power
 * of two subdivision of TIMEBASE, rounding down
 * to the first power of two less than the ti passed in.
 * Note TIMEBASE is the maximum value returned.*/
inttime_t round_down_power_of_two(inttime_t ti);

/*! This function parses a string containing a comma-separated list of variables,
 *  each of which is interpreted as a double.
 *  The purpose is to read an array of output times into the code.
 *  So specifying the output list now looks like:
 *  OutputList  0.1,0.3,0.5,1.0
 *
 *  We sort the input after reading it, so that the initial list need not be sorted.
 *  This function could be repurposed for reading generic arrays in future.
 */
std::vector<double> BuildOutputList(std::string outputliststr);

void set_sync_params_test(int OutputListLength, double * OutputListTimes);
void set_sync_params(ParameterSet * ps);
void setup_sync_points(Cosmology * CP, double TimeIC, double TimeMax, double no_snapshot_until_time, int SnapshotWithFOF);

#endif
