#include <mpi.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <vector>
#include <sstream>
#include <boost/math/quadrature/gauss_kronrod.hpp>

#include "timebinmgr.h"
#include "utils/endrun.h"
#include "utils/mymalloc.h"
#include "utils/openmpsort.h"

#include "cosmology.h"

#define MAXTIMES 1024

static struct sync_params
{
    double ExcursionSetZStart;
    double ExcursionSetZStop;
    double UVBGTimestep;
    int ExcursionSetReionOn;
    std::vector<double> OutputListTimes;
    std::vector<double> PlaneOutputListTimes;
} Sync;

/*! This function parses a string containing a comma-separated list of variables,
 *  each of which is interpreted as a double.
 *  The purpose is to read an array of output times into the code.
 *  So specifying the output list now looks like:
 *  OutputList  0.1,0.3,0.5,1.0
 *
 *  We sort the input after reading it, so that the initial list need not be sorted.
 *  This function could be repurposed for reading generic arrays in future.
 */
std::vector<double> BuildOutputList(std::string outputliststr)
{
    std::vector<double> outputlist;
    if(outputliststr.empty()) {
        return outputlist;
    }
    /* Note TimeInit and TimeMax not yet initialised here*/
    std::istringstream ss(outputliststr);
    std::string token;
    while (std::getline(ss, token, ',')) {
         std::string_view sv = token;
         if (!sv.empty() && sv.front() == '"')
             sv.remove_prefix(1);

        double a;
        try {
            a = std::stod(std::string(sv));
        } catch(const std::exception & e) {
              endrun(1, "Could not parse '%s' in OutputList: %s\n", token.c_str(), e.what());
        }
        if (a < 0.0)
            endrun(1, "Requesting a negative output scaling factor a = %g\n", a);
        outputlist.push_back(a);
    }
    std::sort(outputlist.begin(), outputlist.end());
    return outputlist;
}

//set the other sync params we can't get using the action
void set_sync_params(ParameterSet * ps){
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    if(ThisTask==0)
    {
        Sync.ExcursionSetReionOn = param_get_int(ps,"ExcursionSetReionOn");
        Sync.ExcursionSetZStart = param_get_double(ps,"ExcursionSetZStart");
        Sync.ExcursionSetZStop = param_get_double(ps,"ExcursionSetZStop");
        Sync.UVBGTimestep = param_get_double(ps,"UVBGTimestep");

        Sync.OutputListTimes = BuildOutputList(param_get_string(ps, "OutputList"));
        Sync.PlaneOutputListTimes = BuildOutputList(param_get_string(ps, "PlaneOutputList"));
    }

    // 1. Broadcast the POD members together
    MPI_Bcast(&Sync, offsetof(sync_params, OutputListTimes), MPI_BYTE, 0, MPI_COMM_WORLD);

    // 2. Broadcast the vectors
    size_t len = Sync.OutputListTimes.size();
    MPI_Bcast(&len, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    Sync.OutputListTimes.resize(len);
    MPI_Bcast(Sync.OutputListTimes.data(), len, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    len = Sync.PlaneOutputListTimes.size();
    MPI_Bcast(&len, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    Sync.PlaneOutputListTimes.resize(len);
    MPI_Bcast(Sync.PlaneOutputListTimes.data(), len, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    return;
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
    int64_t NSyncPointsAlloc = Sync.OutputListTimes.size() + Sync.PlaneOutputListTimes.size() + 2;

    /* Excursion set sync points ensure that the reionization excursion set model is run frequently*/
    const double ExcursionSet_delta_a = 0.0001;
    const double a_end = 1/(1+Sync.ExcursionSetZStop) < TimeMax ? 1/(1+Sync.ExcursionSetZStop) : TimeMax;

    if(Sync.ExcursionSetReionOn) {
        double uv_a = 1/(1+Sync.ExcursionSetZStart) > TimeIC ? 1/(1+Sync.ExcursionSetZStart) : TimeIC;
        while (uv_a <= a_end) {
            NSyncPointsAlloc++;
            double lbt = time_to_present(uv_a,CP);
            double delta_lbt = 0.0;
            while ((delta_lbt <= Sync.UVBGTimestep) && (uv_a <= TimeMax)) {
                uv_a += ExcursionSet_delta_a;
                delta_lbt = lbt - time_to_present(uv_a,CP);
            }
        }
    }
    //z=20 to z=4 is ~150 syncpoints at 10 Myr spaces
    SyncPoints = (SyncPoint *) mymalloc("SyncPoints", sizeof(SyncPoint) * NSyncPointsAlloc);

    /* Set up first and last entry to SyncPoints; TODO we can insert many more! */
    //NOTE(jdavies): these first syncpoints need to be in order

    SyncPoints[0].a = TimeIC;
    SyncPoints[0].loga = log(TimeIC);
    SyncPoints[0].write_snapshot = false; /* by default no output here. */
    SyncPoints[0].write_fof = false;
    SyncPoints[0].calc_uvbg = false;
    SyncPoints[0].write_plane = false;
    SyncPoints[0].plane_snapnum = -1;
    NSyncPoints = 1;

    // set up UVBG syncpoints at given intervals
    if(Sync.ExcursionSetReionOn) {
        double uv_a = 1/(1+Sync.ExcursionSetZStart) > TimeIC ? 1/(1+Sync.ExcursionSetZStart) : TimeIC;
        while (uv_a <= a_end) {
            SyncPoints[NSyncPoints].a = uv_a;
            SyncPoints[NSyncPoints].loga = log(uv_a);
            SyncPoints[NSyncPoints].write_snapshot = 0;
            SyncPoints[NSyncPoints].write_fof = 0;
            SyncPoints[NSyncPoints].calc_uvbg = 1;
            NSyncPoints++;
            if(NSyncPoints > NSyncPointsAlloc)
                endrun(1, "Tried to generate %ld syncpoints, %ld allocated\n", NSyncPoints, NSyncPointsAlloc);
            //message(0,"added UVBG syncpoint at a = %.3f z = %.3f, Nsync = %ld\n",uv_a,1/uv_a - 1,NSyncPoints);
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
        message(0,"Added %ld Syncpoints for the excursion Set\n",NSyncPoints-1);
    }

    SyncPoints[NSyncPoints].a = TimeMax;
    SyncPoints[NSyncPoints].loga = log(TimeMax);
    SyncPoints[NSyncPoints].write_snapshot = true;
    SyncPoints[NSyncPoints].calc_uvbg = false;
    SyncPoints[NSyncPoints].write_fof = true;
    SyncPoints[NSyncPoints].write_plane = false;
    SyncPoints[NSyncPoints].plane_snapnum = -1;
    NSyncPoints++;

    /* we do an insertion sort here. A heap is faster but who cares the speed for this? */
    for(size_t i = 0; i < Sync.OutputListTimes.size(); i ++) {
        // print outputlisttime and index
        // message(0, "outIdx: %d, outtime: %g, planeoutIdx: %d, planeouttime: %g.\n", outIdx, Sync.OutputListTimes[outIdx], planeoutIdx, Sync.PlaneOutputListTimes[planeoutIdx]);
        int64_t j = 0;
        double a = Sync.OutputListTimes[i];
        double loga = log(a);

        if(a < TimeIC || a > TimeMax) {
            /*If the user inputs syncpoints outside the scope of the simulation, it can mess
             *with the timebins, which causes errors when calculating densities from the ICs,
             *so we exclude them here*/
            continue;
        }

        for(j = 0; j < NSyncPoints; j ++) {
            if(a <= SyncPoints[j].a) {
                break;
            }
        }
        /* found, so loga >= SyncPoints[j].loga */
        if(a == SyncPoints[j].a) {
            /* requesting output on an existing entry, e.g. TimeInit or duplicated entry */
        } else {
            /* insert the item; */
            memmove(&SyncPoints[j + 1], &SyncPoints[j], sizeof(SyncPoints[0]) * (NSyncPoints - j));
            memset(&SyncPoints[j], 0, sizeof(SyncPoints[0]));
            SyncPoints[j].a = a;
            SyncPoints[j].loga = loga;
            NSyncPoints ++;
            //message(0,"added outlist syncpoint at a = %.3f, j = %d, Ns = %ld\n",a,j,NSyncPoints);
        }
        if(SyncPoints[j].a > no_snapshot_until_time) {
            SyncPoints[j].write_snapshot = 1;
            if(SnapshotWithFOF)
                SyncPoints[j].write_fof = 1;
        }
        SyncPoints[j].plane_snapnum = -1;
    }

    /* Now insert the plane outputs*/
    for(size_t i = 0; i < Sync.PlaneOutputListTimes.size(); i ++) {
        int64_t j = 0;
        double a = Sync.PlaneOutputListTimes[i];
        double loga = log(a);
        if(a < TimeIC || a > TimeMax) {
            /*If the user inputs syncpoints outside the scope of the simulation, it can mess
             *with the timebins, which causes errors when calculating densities from the ICs,
             *so we exclude them here*/
            continue;
        }

        for(j = 0; j < NSyncPoints; j ++) {
            if(a <= SyncPoints[j].a) {
                break;
            }
        }
        /* found, so loga >= SyncPoints[j].loga */
        // to avoid setting sync points too close to each other (which can cause bad timestep errors)
        if(fabs(loga - SyncPoints[j].loga) > 1e-4) {
            /* insert a blank item with no snapshot output. */
            memmove(&SyncPoints[j + 1], &SyncPoints[j], sizeof(SyncPoints[0]) * (NSyncPoints - j));
            memset(&SyncPoints[j], 0, sizeof(SyncPoints[0]));
            SyncPoints[j].a = a;
            SyncPoints[j].loga = loga;
            NSyncPoints ++;
            //message(0,"added outlist syncpoint at a = %.3f, j = %d, Ns = %ld\n",a,j,NSyncPoints);
        }
        SyncPoints[j].write_plane = 1;
        SyncPoints[j].plane_snapnum = i;
    }

    for(int i = 0; i < NSyncPoints; i++) {
        SyncPoints[i].ti = (i * 1L) << (TIMEBINS);
    }
    if(NSyncPoints > NSyncPointsAlloc)
        endrun(1, "Tried to generate %ld syncpoints, %ld allocated\n", NSyncPoints, NSyncPointsAlloc);

    //message(1,"NSyncPoints = %ld, OutputListLength = %ld , timemax = %.3f\n",NSyncPoints,Sync.OutputListLength,TimeMax);
    /*for(i = 0; i < NSyncPoints; i++) {
        message(1,"Out: %g %ld\n", exp(SyncPoints[i].loga), SyncPoints[i].ti);
    }*/
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
