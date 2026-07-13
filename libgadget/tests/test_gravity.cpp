/*Simple test for gravitational force accuracy.*/
#define BOOST_TEST_MODULE gravity
#include "booststub.h"

#include <time.h>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <omp.h>

#include <libgadget/utils/mymalloc.h>
#include <libgadget/utils/system.h>
#include <libgadget/utils/endrun.h>
#include <libgadget/partmanager.h>
#include <libgadget/walltime.h>
#include <libgadget/domain.h>
#include <libgadget/forcetree.h>
#include <libgadget/gravity.h>
#include <libgadget/petapm.h>
#include <libgadget/timestep.h>
#include <libgadget/physconst.h>

static struct ClockTable CT;
static const double G = 43.0071;

static void setup(void) {
    walltime_init(&CT);
    /*Set up the important parts of the All structure.*/
    /*Particles should not be outside p_i*/
    PartManager->BoxSize = 8;
    PartManager->NumPart = 16*16*16;
    struct DomainParams dp = {0};
    dp.DomainOverDecompositionFactor = 2;
    dp.TopNodeAllocFactor = 1.;
    dp.SetAsideFactor = 1;
    set_domain_par(dp);
    /* The FFT backend is chosen by the PETAPM_TEST_FFTW environment
     * variable so both code paths can be exercised. */
    enum PetaPMBackend backend = getenv("PETAPM_TEST_FFTW") ? PETAPM_BACKEND_FFTW : PETAPM_BACKEND_HEFFTE;
    petapm_module_init(omp_get_max_threads(), 0, backend);
    init_forcetree_params(0.7);
}

/* Accumulate the acceleration on the particle at ipos from a unit mass at jpos,
 * shifted by offset. Returns without accumulating for coincident points, so
 * self-interactions are skipped.*/
static void
grav_force(const double * ipos, const double * jpos, const double * offset, double * accn)
{
    double r2 = 0;
    int d;
    double dist[3];
    for(d = 0; d < 3; d ++) {
        /* the distance vector points to the source at jpos */
        dist[d] = offset[d] + ipos[d] - jpos[d];
        r2 += dist[d] * dist[d];
    }

    /* Skip self-interactions*/
    if(r2 == 0)
        return;

    const double r = sqrt(r2);

    const double h = FORCE_SOFTENING();

    double fac = 1 / (r2 * r);
    if(r < h) {
        double h_inv = 1.0 / h;
        double h3_inv = h_inv * h_inv * h_inv;
        double u = r * h_inv;
        if(u < 0.5)
            fac = 1. * h3_inv * (10.666666666667 + u * u * (32.0 * u - 38.4));
        else
            fac =
                1. * h3_inv * (21.333333333333 - 48.0 * u +
                        38.4 * u * u - 10.666666666667 * u * u * u - 0.066666666667 / (u * u * u));
    }

    for(d = 0; d < 3; d ++)
        accn[d] += - dist[d] * fac * G;
}

void check_accns(double * meanerr_tot, double * maxerr_tot, double *PairAccn, double meanacc)
{
    double meanerr=0, maxerr=-1;
    int64_t i;
    /* This checks that the short-range force accuracy is being correctly estimated.*/
    #pragma omp parallel for reduction(+: meanerr) reduction(max: maxerr)
    for(i = 0; i < PartManager->NumPart; i++)
    {
        int k;
        for(k=0; k<3; k++) {
            double err = fabs((PairAccn[3*i+k] - (PartManager->Base[i].GravPM[k] + PartManager->Base[i].FullTreeGravAccel[k]))/meanacc);
            meanerr += err;
            if(maxerr < err)
                maxerr = err;
        }
    }
    MPI_Allreduce(&meanerr, meanerr_tot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&maxerr, maxerr_tot, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    int64_t tot_npart;
    MPI_Allreduce(&PartManager->NumPart, &tot_npart, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);

    *meanerr_tot/= (tot_npart*3.);
}

static void find_means(double * meangrav, double * suppmean, double * suppaccns)
{
    int i;
    double meanacc = 0, meanforce = 0;
    #pragma omp parallel for reduction(+: meanacc) reduction(+: meanforce)
    for(i = 0; i < PartManager->NumPart; i++)
    {
        int k;
        for(k=0; k<3; k++) {
            if(suppaccns)
                meanacc += fabs(suppaccns[3*i+k]);
            meanforce += fabs(PartManager->Base[i].GravPM[k] + PartManager->Base[i].FullTreeGravAccel[k]);
        }
    }
    int64_t tot_npart;
    MPI_Allreduce(&PartManager->NumPart, &tot_npart, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);
    if(suppaccns) {
        MPI_Allreduce(&meanacc, suppmean, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        *suppmean/= (tot_npart*3.);
    }
    MPI_Allreduce(&meanforce, meangrav, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    *meangrav/= (tot_npart*3.);
}


/* This checks the force on each local particle using a direct summation
 * over the global particle set (of which every rank has a copy in gpos):
 * very slow, but accurate.
 * Periodic boundary conditions are included by mirroring the box.*/
static void force_direct(double * accn, const double * gpos, const int64_t nglobal)
{
    memset(accn, 0, 3 * sizeof(double) * PartManager->NumPart);
    /* Checked that increasing the number of mirror boxes has no visible effect on the computed force accuracy*/
    int repeat = 1;
    int64_t i;
    /* (slowly) compute gravitational force, accounting for periodicity by just inventing extra boxes on either side.*/
    #pragma omp parallel for
    for(i = 0; i < PartManager->NumPart; i++) {
        int xx, yy, zz;
        for(xx=-repeat; xx <= repeat; xx++)
            for(yy=-repeat; yy <= repeat; yy++)
                for(zz=-repeat; zz <= repeat; zz++)
                {
                    double offset[3] = {PartManager->BoxSize * xx, PartManager->BoxSize * yy, PartManager->BoxSize * zz};
                    int64_t j;
                    for(j = 0; j < nglobal; j++)
                        grav_force(PartManager->Base[i].Pos, gpos + 3*j, offset, accn + 3*i);
                }
    }
}

static int check_against_force_direct(double ErrTolForceAcc, const double * gpos, const int64_t nglobal)
{
    double * accn = mymalloc("accelerations", double, 3*PartManager->NumPart);
    force_direct(accn, gpos, nglobal);
    double meanerr=0, maxerr=-1, meanacc=0, meanforce=0;
    find_means(&meanacc, &meanforce, accn);
    check_accns(&meanerr, &maxerr, accn, meanacc);
    myfree(accn);
    message(0, "Mean rel err is: %g max rel err is %g, meanacc %g mean grav force %g\n", meanerr, maxerr, meanacc, meanforce);
    /*Make some statements about the force error*/
    BOOST_TEST(maxerr < 3*ErrTolForceAcc);
    BOOST_TEST(meanerr < 0.8*ErrTolForceAcc);

    return 0;
}

/* Distribute the global particle set (gpos, identical on every rank) among the
 * ranks: each rank keeps a contiguous slice, with globally unique IDs recording
 * the index into gpos.*/
static void setup_particles(const double * gpos, const int64_t nglobal)
{
    int ThisTask, NTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    MPI_Comm_size(MPI_COMM_WORLD, &NTask);
    const int64_t start = ThisTask * nglobal / NTask;
    const int64_t end = (ThisTask + 1) * nglobal / NTask;
    PartManager->NumPart = end - start;
    int64_t i;
    #pragma omp parallel for
    for(i=0; i<PartManager->NumPart; i++) {
        int j;
        for(j=0; j<3; j++)
            PartManager->Base[i].Pos[j] = gpos[3*(start+i)+j];
        PartManager->Base[i].Type = 1;
        PartManager->Base[i].Mass = 1;
        PartManager->Base[i].ID = start+i+1;
        PartManager->Base[i].TimeBinHydro = 0;
        PartManager->Base[i].TimeBinGravity = 0;
        PartManager->Base[i].IsGarbage = 0;
    }
}

static void do_force_test(int Nmesh, double Asmth, double ErrTolForceAcc, int direct, const double * gpos, const int64_t nglobal)
{
    setup_particles(gpos, nglobal);

    DomainDecomp ddecomp = {0};
    domain_decompose_full(&ddecomp, MPI_COMM_WORLD);

    PetaPM pm = {0};
    gravpm_init_periodic(&pm, PartManager->BoxSize, Asmth, Nmesh, G);
    /* Setup cosmology*/
    Cosmology CP ={0};
    CP.MNu[0] = CP.MNu[1] = CP.MNu[2] = 0;
    CP.OmegaCDM = 0.3;
    CP.CMBTemperature = 2.72;
    CP.HubbleParam = 0.7;
    CP.Omega0 = 0.3;
    CP.OmegaBaryon = 0.045;
    CP.OmegaLambda = 0.7;
    struct UnitSystem units = get_unitsystem(3.085678e21, 1.989e43, 1e5);
    init_cosmology(&CP, 0.01, units);

    gravpm_force(&pm, &ddecomp, &CP, 0.1, CM_PER_MPC/1000., ".", 0.01);
    ForceTree Tree = {0};
    force_tree_full(&Tree, &ddecomp, 1, "");
    const double rho0 = CP.Omega0 * 3 * CP.Hubble * CP.Hubble / (8 * M_PI * G);

    /* Barnes-Hut on first iteration*/
    struct gravshort_tree_params treeacc = {0};
    treeacc.BHOpeningAngle = 0.175;
    treeacc.TreeUseBH = 2;
    treeacc.Rcut = 7;
    treeacc.ErrTolForceAcc = ErrTolForceAcc;
    treeacc.FractionalGravitySoftening = 1./30.;
    treeacc.MaxExportBufferBytes = 1024 * 1024 * 1024;
    treeacc.ShortRangeForceWindowType = SHORTRANGE_FORCE_WINDOW_TYPE_EXACT;

    set_gravshort_treepar(treeacc);
    /* Softening must use the global particle number so it is independent of the number of ranks*/
    gravshort_set_softenings(PartManager->BoxSize / cbrt(nglobal));

    /* Twice so the opening angle is consistent*/
    ActiveParticles act = init_empty_active_particles(PartManager);
    grav_short_tree(&act, &pm, &Tree, NULL, rho0, 0);
    grav_short_tree(&act, &pm, &Tree, NULL, rho0, 0);

    force_tree_free(&Tree);
    petapm_destroy(&pm);
    domain_free(&ddecomp);
    if(direct)
        check_against_force_direct(ErrTolForceAcc, gpos, nglobal);
}

BOOST_AUTO_TEST_CASE(test_force_flat)
{
    setup();
    /*Set up the particle data*/
    int numpart = PartManager->NumPart;
    int ncbrt = cbrt(numpart);
    particle_alloc_memory(PartManager, 8, numpart);
    /* Create a regular grid of particles, 16x16x16, all of type 1,
     * in a box 8 kpc across.*/
    double * gpos = mymalloc("globalpos", double, 3*numpart);
    int i;
    #pragma omp parallel for
    for(i=0; i<numpart; i++) {
        gpos[3*i] = (PartManager->BoxSize/ncbrt) * (i/ncbrt/ncbrt);
        gpos[3*i+1] = (PartManager->BoxSize/ncbrt) * ((i/ncbrt) % ncbrt);
        gpos[3*i+2] = (PartManager->BoxSize/ncbrt) * (i % ncbrt);
    }
    do_force_test(48, 1.5, 0.002, 0, gpos, numpart);
    /* For a homogeneous mass distribution, the force should be zero*/
    double meanerr=0, maxerr=-1;
    #pragma omp parallel for reduction(+: meanerr) reduction(max: maxerr)
    for(i = 0; i < PartManager->NumPart; i++)
    {
        int k;
        for(k=0; k<3; k++) {
            double err = fabs((PartManager->Base[i].GravPM[k] + PartManager->Base[i].FullTreeGravAccel[k]));
            meanerr += err;
            if(maxerr < err)
                maxerr = err;
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, &meanerr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &maxerr, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    int64_t tot_npart;
    MPI_Allreduce(&PartManager->NumPart, &tot_npart, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);
    meanerr/= (tot_npart*3.);

    message(0, "Max force %g, mean grav force %g\n", maxerr, meanerr);
    /*Make some statements about the force error*/
    BOOST_TEST(maxerr < 0.015);
    BOOST_TEST(meanerr < 0.005);
    myfree(gpos);
    myfree(PartManager->Base);
}

BOOST_AUTO_TEST_CASE(test_force_close)
{
    setup();
    /*Set up the particle data*/
    int numpart = PartManager->NumPart;
    int ncbrt = cbrt(numpart);
    double close = 5000;
    particle_alloc_memory(PartManager, 8, numpart);
    /* Create particles clustered in one place, all of type 1.*/
    double * gpos = mymalloc("globalpos", double, 3*numpart);
    int i;
    #pragma omp parallel for
    for(i=0; i<numpart; i++) {
        gpos[3*i] = 4. + (i/ncbrt/ncbrt)/close;
        gpos[3*i+1] = 4. + ((i/ncbrt) % ncbrt) /close;
        gpos[3*i+2] = 4. + (i % ncbrt)/close;
    }
    do_force_test(48, 1.5, 0.002, 1, gpos, numpart);
    myfree(gpos);
    myfree(PartManager->Base);
}

void do_random_test(boost::random::mt19937 & r, const int numpart)
{
    boost::random::uniform_real_distribution<double> dist(0, 1);
    /* Create a random particle distribution, all of type 1,
     * in a box 8 kpc across. Every rank draws the same sequence,
     * so gpos is identical everywhere.*/
    double * gpos = mymalloc("globalpos", double, 3*numpart);
    int i;
    for(i=0; i<numpart/4; i++) {
        int j;
        for(j=0; j<3; j++)
            gpos[3*i+j] = PartManager->BoxSize * dist(r);
    }
    for(i=numpart/4; i<3*numpart/4; i++) {
        int j;
        for(j=0; j<3; j++)
            gpos[3*i+j] = PartManager->BoxSize/2 + PartManager->BoxSize/8 * exp(pow(dist(r)-0.5,2));
    }
    for(i=3*numpart/4; i<numpart; i++) {
        int j;
        for(j=0; j<3; j++)
            gpos[3*i+j] = PartManager->BoxSize*0.1 + PartManager->BoxSize/32 * exp(pow(dist(r)-0.5,2));
    }
    do_force_test(48, 1.5, 0.002, 1, gpos, numpart);
    myfree(gpos);
}

BOOST_AUTO_TEST_CASE(test_force_random)
{
    setup();
    /*Set up the particle data*/
    int numpart = PartManager->NumPart;
    auto r = boost::random::mt19937(0);
    particle_alloc_memory(PartManager, 8, numpart);
    int i;
    for(i=0; i<2; i++) {
        do_random_test(r, numpart);
    }
    myfree(PartManager->Base);
}
