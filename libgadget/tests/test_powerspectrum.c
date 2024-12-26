/* This file tests the power spectrum routines only.
 * The actual PM code is too complicated for now. */
#define BOOST_TEST_MODULE powerspectrum

#include "booststub.h"

#include <bigfile-mpi.h>
#include <libgadget/powerspectrum.h>

#define NUM_THREADS 4

/*Test the total powerspectrum on one processor only*/
BOOST_AUTO_TEST_CASE(test_total_powerspectrum)
{
    /*Check allocation*/
    int nmpi;
    Power PowerSpectrum;
    MPI_Comm_size(MPI_COMM_WORLD, &nmpi);

    powerspectrum_alloc(&PowerSpectrum,15,NUM_THREADS, 0, 3.085678e24);
    BOOST_TEST(PowerSpectrum.Nmodes);
    BOOST_TEST(PowerSpectrum.Power);
    BOOST_TEST(PowerSpectrum.kk);
    powerspectrum_zero(&PowerSpectrum);
    BOOST_TEST(PowerSpectrum.Nmodes[0] == 0);
    BOOST_TEST(PowerSpectrum.Nmodes[PowerSpectrum.size-1] == 0);

    //Construct input power (this would be done by the power spectrum routine in petapm)
    int ii, th;
    for(ii=0; ii<15; ii++) {
        for(th = 0; th < NUM_THREADS; th++) {
            PowerSpectrum.Nmodes[ii+PowerSpectrum.size*th] = ii;
            PowerSpectrum.Power[ii+PowerSpectrum.size*th] = ii*sin(ii)*sin(ii);
            PowerSpectrum.kk[ii+PowerSpectrum.size*th] = ii*ii;
        }
    }
    PowerSpectrum.Norm = 1;
    /*Now every thread and every MPI has the same data. Sum it.*/
    powerspectrum_sum(&PowerSpectrum);

    /*Check summation was done correctly*/
    BOOST_TEST(PowerSpectrum.Nmodes[0] == NUM_THREADS*nmpi);
    BOOST_TEST(PowerSpectrum.Nmodes[13] == NUM_THREADS*nmpi*14);

    BOOST_TEST(PowerSpectrum.Power[0] == sin(1)*sin(1), tt::tolerance(1e-5));
    BOOST_TEST(PowerSpectrum.Power[12] == sin(13)*sin(13), tt::tolerance(1e-5));
    BOOST_TEST(PowerSpectrum.kk[12] == 2 * M_PI *13, tt::tolerance(1e-5));
    BOOST_TEST(PowerSpectrum.kk[0] == 2 * M_PI, tt::tolerance(1e-5));
}
