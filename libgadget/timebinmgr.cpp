#include <mpi.h>
#include <math.h>
#include <vector>
#include <algorithm>
#include <boost/math/quadrature/gauss_kronrod.hpp>

#include "timebinmgr.h"
#include "utils/endrun.h"
#include "cosmology.h"

static struct sync_params
{
    double ExcursionSetZStart;
    double ExcursionSetZStop;
    double UVBGTimestep;
    int ExcursionSetReionOn;
    std::vector<double> OutputListTimes;
    std::vector<double> PlaneOutputListTimes;
} Sync;

//set the other sync params we can't get using the action
void set_sync_params(ParameterSet * ps)
{
    Sync.ExcursionSetReionOn = param_get_int(ps,"ExcursionSetReionOn");
    Sync.ExcursionSetZStart = param_get_double(ps,"ExcursionSetZStart");
    Sync.ExcursionSetZStop = param_get_double(ps,"ExcursionSetZStop");
    Sync.UVBGTimestep = param_get_double(ps,"UVBGTimestep");

    Sync.OutputListTimes = BuildOutputList<double>(param_get_string(ps, "OutputList"));
    Sync.PlaneOutputListTimes = BuildOutputList<double>(param_get_string(ps, "PlaneOutputList"));
}

#define SEC_PER_MEGAYEAR 3.155e13

//time_to_present in Myr for excursion set syncpoints
static double time_to_present(const double a, const Cosmology * const CP)
{
    const double hubble = CP->Hubble / CP->UnitTime_in_s * SEC_PER_MEGAYEAR * CP->HubbleParam;

    // Define the integrand as a lambda function
    auto integrand_time_to_present = [CP](const double a) {
        const double h = hubble_function(CP, a);
        return 1 / a / h;
    };

    // Perform the Tanh-Sinh adaptive integration
    const double result = boost::math::quadrature::gauss_kronrod<double, 61>::integrate(integrand_time_to_present, a, 1.0);

    //convert to Myr and multiply by h
    const double time = result / (hubble/CP->Hubble);
    // return time to present as a function of redshift
    return time;
}

/* For the tests*/
void set_sync_params_test(int OutputListLength, double * OutputListTimes)
{
    for(int i = 0; i < OutputListLength; i++)
        Sync.OutputListTimes.push_back(OutputListTimes[i]);
}

/* This function compiles
 *
 * Sync.OutputListTimes, All.TimeIC, All.TimeMax
 *
 * into a list of SyncPoint objects.
 *
 * A SyncPoint is a time step where all state variables are at the same time on the
 * KkdkK timeline.
 *
 * TimeIC and TimeMax are used to ensure restarting from snapshot obtains exactly identical
 * integer stamps.
 **/
TimeBinMgr::TimeBinMgr(Cosmology * CP, double TimeIC, double TimeMax, double no_snapshot_until_time, bool SnapshotWithFOF)
{
    this->CP = CP;

    std::vector<SyncPoint> SyncPoints;
    /* Set up first entry: no output here. */
    SyncPoint tmpsync;
    tmpsync.loga = log(TimeIC);
    SyncPoints.push_back(tmpsync);

    // set up UVBG syncpoints at given intervals
    if(Sync.ExcursionSetReionOn) {
        /* Excursion set sync points ensure that the reionization excursion set model is run frequently*/
        const double ExcursionSet_delta_a = 0.0001;
        const double a_end = 1/(1+Sync.ExcursionSetZStop) < TimeMax ? 1/(1+Sync.ExcursionSetZStop) : TimeMax;
        double uv_a = 1/(1+Sync.ExcursionSetZStart) > TimeIC ? 1/(1+Sync.ExcursionSetZStart) : TimeIC;
        while (uv_a <= a_end) {
            SyncPoint tmpsync;
            tmpsync.loga = log(uv_a);
            tmpsync.calc_uvbg = true;
            SyncPoints.push_back(tmpsync);
            //message(0,"added UVBG syncpoint at a = %.3f z = %.3f, Nsync = %ld\n",uv_a,1/uv_a - 1,SyncPoints.size());
            // TODO(smutch): OK - this is ridiculous (sorry!), but I just wanted to quickly hack something...
            // TODO(jdavies): fix low-z where delta_a > 10Myr
            double lbt = time_to_present(uv_a,CP);
            double delta_lbt = 0.0;
            while ((delta_lbt <= Sync.UVBGTimestep) && (uv_a <= TimeMax)) {
                uv_a += ExcursionSet_delta_a;
                delta_lbt = lbt - time_to_present(uv_a,CP);
                //message(0,"trying UVBG syncpoint at a = %.3e, z = %.3e, delta_lbt = %.3e\n",uv_a,1/uv_a - 1,delta_lbt);
            }
        }
        message(0,"Added %lu Syncpoints for the excursion Set\n",SyncPoints.size()-1);
    }

    tmpsync.loga = log(TimeMax);
    tmpsync.write_snapshot = true;
    tmpsync.write_fof = true;
    SyncPoints.push_back(tmpsync);

    /* Finds the first SyncPoint with loga >= the given loga, keeping SyncPoints sorted. */
    auto by_loga = [](const SyncPoint& s, double loga) { return s.loga < loga; };

    for(size_t i = 0; i < Sync.OutputListTimes.size(); i ++) {
        double a = Sync.OutputListTimes[i];
        double loga = log(a);

        if(a < TimeIC || a > TimeMax) {
            /*If the user inputs syncpoints outside the scope of the simulation, it can mess
             *with the timebins, which causes errors when calculating densities from the ICs,
             *so we exclude them here*/
            continue;
        }

        auto it = std::lower_bound(SyncPoints.begin(), SyncPoints.end(), loga, by_loga);
        if(it == SyncPoints.end() || loga != it->loga) {
            /* insert a blank item; */
            SyncPoint tmpsync;
            tmpsync.loga = loga;
            it = SyncPoints.insert(it, tmpsync);
        }
        if(it->loga > log(no_snapshot_until_time)) {
            it->write_snapshot = true;
            if(SnapshotWithFOF)
                it->write_fof = true;
        }
        it->plane_snapnum = -1;
    }

    /* Now insert the plane outputs*/
    for(size_t i = 0; i < Sync.PlaneOutputListTimes.size(); i ++) {
        double a = Sync.PlaneOutputListTimes[i];
        double loga = log(a);
        if(a < TimeIC || a > TimeMax) {
            /*If the user inputs syncpoints outside the scope of the simulation, it can mess
             *with the timebins, which causes errors when calculating densities from the ICs,
             *so we exclude them here*/
            continue;
        }

        auto it = std::lower_bound(SyncPoints.begin(), SyncPoints.end(), loga, by_loga);
        // to avoid setting sync points too close to each other (which can cause bad timestep errors)
        if(it == SyncPoints.end() || fabs(loga - it->loga) > 1e-4) {
            /* insert a blank item with no snapshot output. */
            SyncPoint tmpsync;
            tmpsync.loga = loga;
            it = SyncPoints.insert(it, tmpsync);
        }
        it->write_plane = 1;
        it->plane_snapnum = i;
    }

    // This avoids the memory access overhead of std::vector and avoids copying.
    this->NSyncPoints = SyncPoints.size();
    this->SyncPoints = std::make_unique<SyncPoint[]>(SyncPoints.size());
    std::copy(SyncPoints.begin(), SyncPoints.end(), this->SyncPoints.get());
}

/* Function to compute comoving distance using the adaptive integrator */
double compute_comoving_distance(Cosmology * CP, double a0, double a1, const double UnitVelocity_in_cm_per_s)
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

inttime_t
round_down_power_of_two(inttime_t dti)
{
    /* make dti a power 2 subdivision */
    inttime_t ti_min = TIMEBASE;
    int sign = 1;
    if(dti < 0) {
        dti = -dti;
        sign = -1;
    }
    while(ti_min > dti)
        ti_min >>= 1;
    return ti_min * sign;
}
