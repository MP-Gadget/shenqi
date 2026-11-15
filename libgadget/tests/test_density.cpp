/*Simple test for the density treewalk*/
#define BOOST_TEST_MODULE density
#include "booststub.h"

#include <optional>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real_distribution.hpp>

#include <libgadget/partmanager.h>
#include <libgadget/walltime.h>
#include <libgadget/slotsmanager.h>
#include <libgadget/utils/mymalloc.h>
#include <libgadget/density2.h>
#include <libgadget/domain.h>
#include <libgadget/forcetree.h>
#include <libgadget/timestep.h>
#include <libgadget/gravity.h>

static struct ClockTable CT;

/* The true struct for the state variable*/
struct density_testdata
{
    struct sph_pred_data sph_pred;
    struct density_params dp;
    TimeBinMgr timebinmgr;
};

/* nstar is the number of star particles to leave room for: the stellar density test needs them.*/
static struct density_testdata setup_density(const int64_t nstar = 0)
{
    struct density_testdata data = {0};
    /* Needed so the integer timeline works*/
    data.timebinmgr = TimeBinMgr(NULL,0.01, 0.1, 0.0, false);

    /*Reserve space for the slots*/
    slots_init(0.01 * PartManager->MaxPart, SlotsManager);
    slots_set_enabled(0, sizeof(struct sph_particle_data), SlotsManager);
    slots_set_enabled(5, sizeof(struct sph_particle_data), SlotsManager);
    int64_t atleast[6] = {0};
    atleast[0] = pow(32,3);
    atleast[5] = 2;
    if(nstar > 0) {
        slots_set_enabled(4, sizeof(struct star_particle_data), SlotsManager);
        atleast[4] = nstar;
    }
    int64_t maxpart = 0;
    int i;
    for(i = 0; i < 6; i++)
        maxpart+=atleast[i];
    const double BoxSize = 8;
    particle_alloc_memory(PartManager, BoxSize, maxpart);
    slots_reserve(1, atleast, SlotsManager);
    walltime_init(&CT);
    init_forcetree_params(0.7);
    data.sph_pred.EntVarPred = NULL;
    /*Set up the domain decomposition parameters*/
    struct DomainParams dmp = {0};
    dmp.DomainOverDecompositionFactor = 2;
    dmp.TopNodeAllocFactor = 1.;
    dmp.SetAsideFactor = 1;
    set_domain_par(dmp);
    data.dp.DensityResolutionEta = 1.;
    data.dp.BlackHoleNgbFactor = 2;
    data.dp.MaxNumNgbDeviation = 0.5;
    data.dp.DensityKernelType = DENSITY_KERNEL_CUBIC_SPLINE;
    data.dp.MinGasHsmlFractional = 0.006;
    data.dp.BlackHoleMaxAccretionRadius = 99999.;
    struct gravshort_tree_params tree_params = {0};
    tree_params.FractionalGravitySoftening = 1;
    set_gravshort_treepar(tree_params);
    gravshort_set_softenings(1);
    data.dp.MinGasHsml = 0.006 * (FORCE_SOFTENING()/2.8);
    set_densitypar(data.dp);
    return data;
}

/* Perform some simple checks on the densities*/
static void check_densities(double MinGasHsml)
{
    int i;
    double maxHsml=PartManager->Base[0].Hsml, minHsml= PartManager->Base[0].Hsml;
    for(i=0; i<PartManager->NumPart; i++) {
        BOOST_TEST(std::isfinite(PartManager->Base[i].Hsml));
        BOOST_TEST(std::isfinite(SPHP(i).Density));
        BOOST_TEST(SPHP(i).Density > 0);
        if(PartManager->Base[i].Hsml < minHsml)
            minHsml = PartManager->Base[i].Hsml;
        if(PartManager->Base[i].Hsml > maxHsml)
            maxHsml = PartManager->Base[i].Hsml;
    }
    BOOST_TEST(std::isfinite(minHsml));
    BOOST_TEST(minHsml >= MinGasHsml);
    BOOST_TEST(maxHsml <= PartManager->BoxSize);

}

/* Distribute the global particle set (gpos and ghsml, identical on every
 * rank) among the ranks: each rank keeps a contiguous slice, with globally
 * unique IDs recording the index into gpos. If lastisbh is set, the last
 * particle in the global set is a black hole: since it is globally last it is
 * also locally last, keeping the particles ordered by type as
 * slots_setup_topology requires.*/
static void setup_density_particles(const double * gpos, const double * ghsml, const int64_t nglobal, const int lastisbh)
{
    int ThisTask, NTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    MPI_Comm_size(MPI_COMM_WORLD, &NTask);
    const int64_t start = ThisTask * nglobal / NTask;
    const int64_t end = (ThisTask + 1) * nglobal / NTask;
    PartManager->NumPart = end - start;
    const int64_t npbh = (lastisbh && end == nglobal);
    int64_t i;
    #pragma omp parallel for
    for(i=0; i<PartManager->NumPart; i++) {
        const int64_t gid = start + i;
        memset(&PartManager->Base[i], 0, sizeof(struct particle_data));
        PartManager->Base[i].ID = gid + 1;
        PartManager->Base[i].Mass = 1;
        PartManager->Base[i].Hsml = ghsml[gid];
        int j;
        for(j=0; j<3; j++) {
            PartManager->Base[i].Pos[j] = gpos[3*gid+j];
            PartManager->Base[i].Vel[j] = 1.5;
        }
    }
    int64_t NLocal[6] = {0};
    NLocal[0] = PartManager->NumPart - npbh;
    NLocal[5] = npbh;
    slots_setup_topology(PartManager, NLocal, SlotsManager);
    slots_setup_id(PartManager, SlotsManager);
    #pragma omp parallel for
    for(i=0; i<NLocal[0]; i++) {
        SPHP(i).Entropy = 1;
        SPHP(i).DtEntropy = 0;
        SPHP(i).Density = 1;
    }
}

static void do_density_test(struct density_testdata * data, const int64_t nglobal, double expectedhsml, double hsmlerr)
{
    int i;
    DomainDecomp ddecomp = {0};
    domain_decompose_full(&ddecomp, MPI_COMM_WORLD);

    ActiveParticles act = init_empty_active_particles(PartManager);

    {
        /* The initial hsml guess uses the tree moments. This test has only gas and BH particles.*/
        ForceTreeMoments momtree(*PartManager, &ddecomp, nullptr, false, "");
        set_init_hsml(&momtree, PartManager->BoxSize, PartManager);
    }
    /*Time doing the density finding*/
    double start, end;
    start = MPI_Wtime();
    /*Find the density*/
    DriftKickTimes kick = {0};
    Cosmology CP = {0};
    CP.CMBTemperature = 2.7255;
    CP.Omega0 = 0.3;
    CP.OmegaLambda = 1- CP.Omega0;
    CP.OmegaBaryon = 0.045;
    CP.HubbleParam = 0.7;
    CP.RadiationOn = 0;
    CP.w0_fld = -1; /*Dark energy equation of state parameter*/
    /*Should be 0.1*/
    struct UnitSystem units = get_unitsystem(3.085678e21, 1.989e43, 1e5);
    init_cosmology(&CP,0.01, units);

    /* Rebuild without moments to check it works.
     * std::optional so the tree can be freed before the checks.*/
    std::optional<ForceTreeHmax> tree;
    tree.emplace(*PartManager, &ddecomp, GASMASK, "");
    density(&act, 1, 0, 0, kick, &data->timebinmgr, &CP, &(data->sph_pred.EntVarPred), NULL, &*tree);
    end = MPI_Wtime();
    double ms = (end - start)*1000;
    message(0, "Found densities in %.3g ms\n", ms);
    check_densities(data->dp.MinGasHsmlFractional);
    slots_free_sph_pred_data(&data->sph_pred);

    double avghsml = 0;
    #pragma omp parallel for reduction(+:avghsml)
    for(i=0; i<PartManager->NumPart; i++) {
        avghsml += PartManager->Base[i].Hsml;
    }
    MPI_Allreduce(MPI_IN_PLACE, &avghsml, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    message(0, "Average Hsml: %g Expected %g +- %g\n",avghsml/nglobal, expectedhsml, hsmlerr);
    BOOST_TEST(fabs(avghsml/nglobal - expectedhsml) < hsmlerr);
    /* Make MaxNumNgbDeviation smaller and check we get a consistent result.*/
    double * Hsml = mymalloc2("Hsml", double, PartManager->NumPart);
    #pragma omp parallel for
    for(i=0; i<PartManager->NumPart; i++) {
        Hsml[i] = PartManager->Base[i].Hsml;
    }
    data->dp.MaxNumNgbDeviation = 0.5;
    set_densitypar(data->dp);

    start = MPI_Wtime();
    /*Find the density*/
    density(&act, 1, 0, 0, kick, &data->timebinmgr, &CP, &(data->sph_pred.EntVarPred), NULL, &*tree);
    end = MPI_Wtime();
    slots_free_sph_pred_data(&data->sph_pred);

    ms = (end - start)*1000;
    message(0, "Found 1 dev densities in %.3g ms\n", ms);
    double diff = 0;
    double DesNumNgb = GetNumNgb(GetDensityKernelType());
    /* Free tree before checks so that we still recover if checks fail*/
    tree.reset();

    for(i=0; i<PartManager->NumPart; i++) {
        BOOST_TEST(fabs(Hsml[i]/PartManager->Base[i].Hsml-1) < data->dp.MaxNumNgbDeviation / DesNumNgb);
        if(fabs(Hsml[i] - PartManager->Base[i].Hsml) > diff)
            diff = fabs(Hsml[i] - PartManager->Base[i].Hsml);
    }
    message(0, "Max diff between Hsml: %g\n",diff);
    myfree(Hsml);

    check_densities(data->dp.MinGasHsmlFractional);
    domain_free(&ddecomp);
}

BOOST_AUTO_TEST_CASE(test_density_flat)
{
    struct density_testdata data = setup_density();
    /*Set up the particle data*/
    int ncbrt = 32;
    int numpart = ncbrt*ncbrt*ncbrt;
    /* Create a regular grid of particles, 32x32x32, all of type 0,
     * in a box 8 kpc across.*/
    double * gpos = mymalloc("globalpos", double, 3*numpart);
    double * ghsml = mymalloc("globalhsml", double, numpart);
    int i;
    #pragma omp parallel for
    for(i=0; i<numpart; i++) {
        ghsml[i] = 1.5*PartManager->BoxSize/cbrt(numpart);
        gpos[3*i] = (PartManager->BoxSize/ncbrt) * (i/ncbrt/ncbrt);
        gpos[3*i+1] = (PartManager->BoxSize/ncbrt) * ((i/ncbrt) % ncbrt);
        gpos[3*i+2] = (PartManager->BoxSize/ncbrt) * (i % ncbrt);
    }
    setup_density_particles(gpos, ghsml, numpart, 0);
    do_density_test(&data, numpart, 0.5, 5e-4);
    myfree(ghsml);
    myfree(gpos);
    slots_free(SlotsManager);
    myfree(PartManager->Base);
}

BOOST_AUTO_TEST_CASE(test_density_close)
{
    struct density_testdata data = setup_density();
    /*Set up the particle data*/
    int ncbrt = 32;
    int numpart = ncbrt*ncbrt*ncbrt;
    double close = 500.;
    int i;
    double * gpos = mymalloc("globalpos", double, 3*numpart);
    double * ghsml = mymalloc("globalhsml", double, numpart);
    /* A few particles scattered about the place so the tree is not sparse*/
    #pragma omp parallel for
    for(i=0; i<numpart/4; i++) {
        ghsml[i] = 4*PartManager->BoxSize/cbrt(numpart/8);
        gpos[3*i] = (PartManager->BoxSize/ncbrt) * (i/(ncbrt/2.)/(ncbrt/2.));
        gpos[3*i+1] = (PartManager->BoxSize/ncbrt) * ((i*2/ncbrt) % (ncbrt/2));
        gpos[3*i+2] = (PartManager->BoxSize/ncbrt) * (i % (ncbrt/2));
    }

    /* Create particles clustered in one place, all of type 0.
     * The last particle is a black hole.*/
    #pragma omp parallel for
    for(i=numpart/4; i<numpart; i++) {
        ghsml[i] = 2*ncbrt/close;
        gpos[3*i] = 4.1 + (i/ncbrt/ncbrt)/close;
        gpos[3*i+1] = 4.1 + ((i/ncbrt) % ncbrt) /close;
        gpos[3*i+2] = 4.1 + (i % ncbrt)/close;
    }
    setup_density_particles(gpos, ghsml, numpart, 1);
    do_density_test(&data, numpart, 0.131726, 1e-4);
    myfree(ghsml);
    myfree(gpos);
    slots_free(SlotsManager);
    myfree(PartManager->Base);
}

void do_random_test(struct density_testdata * data, boost::random::mt19937 &r, const int numpart)
{
    boost::random::uniform_real_distribution<double> dist(0.0, 1.0);
    /* Create a randomly spaced set of particles, all of type 0.
     * Every rank draws the same sequence, so gpos is identical everywhere.*/
    double * gpos = mymalloc("globalpos", double, 3*numpart);
    double * ghsml = mymalloc("globalhsml", double, numpart);
    int i;
    for(i=0; i<numpart/4; i++) {
        ghsml[i] = PartManager->BoxSize/cbrt(numpart);
        int j;
        for(j=0; j<3; j++)
            gpos[3*i+j] = PartManager->BoxSize * dist(r);
    }
    for(i=numpart/4; i<3*numpart/4; i++) {
        ghsml[i] = PartManager->BoxSize/cbrt(numpart);
        int j;
        for(j=0; j<3; j++)
            gpos[3*i+j] = PartManager->BoxSize/2 + PartManager->BoxSize/8 * exp(pow(dist(r)-0.5,2));
    }
    for(i=3*numpart/4; i<numpart; i++) {
        ghsml[i] = PartManager->BoxSize/cbrt(numpart);
        int j;
        for(j=0; j<3; j++)
            gpos[3*i+j] = PartManager->BoxSize*0.1 + PartManager->BoxSize/32 * exp(pow(dist(r)-0.5,2));
    }
    setup_density_particles(gpos, ghsml, numpart, 0);
    do_density_test(data, numpart, 0.187515, 1e-3);
    myfree(ghsml);
    myfree(gpos);
}

BOOST_AUTO_TEST_CASE(test_density_random)
{
    struct density_testdata data = setup_density();
    /*Set up the particle data*/
    int ncbrt = 32;
    boost::random::mt19937 r = boost::random::mt19937(0);
    int numpart = ncbrt*ncbrt*ncbrt;
    int i;
    for(i=0; i<2; i++) {
        do_random_test(&data, r, numpart);
    }
    slots_free(SlotsManager);
    myfree(PartManager->Base);
}

/* Find the gas densities twice from the same starting smoothing lengths: with the default
 * export buffer, and with one small enough that it fills up many times when there is more
 * than one rank. The size of the export buffer must not change the answer.
 * With one rank there are no exports and this repeats the first walk.*/
BOOST_AUTO_TEST_CASE(test_density_small_buffer)
{
    struct density_testdata data = setup_density();
    const int numpart = 32*32*32;
    /* The same clustered random distribution as do_random_test.*/
    boost::random::mt19937 r = boost::random::mt19937(0);
    boost::random::uniform_real_distribution<double> dist(0, 1);
    double * gpos = mymalloc("globalpos", double, 3*numpart);
    double * ghsml = mymalloc("globalhsml", double, numpart);
    for(int i = 0; i < numpart; i++) {
        ghsml[i] = PartManager->BoxSize/cbrt(numpart);
        for(int j = 0; j < 3; j++) {
            if(i < numpart/4)
                gpos[3*i+j] = PartManager->BoxSize * dist(r);
            else if(i < 3*numpart/4)
                gpos[3*i+j] = PartManager->BoxSize/2 + PartManager->BoxSize/8 * exp(pow(dist(r)-0.5,2));
            else
                gpos[3*i+j] = PartManager->BoxSize*0.1 + PartManager->BoxSize/32 * exp(pow(dist(r)-0.5,2));
        }
    }
    setup_density_particles(gpos, ghsml, numpart, 0);
    myfree(ghsml);
    myfree(gpos);

    DomainDecomp ddecomp = {0};
    domain_decompose_full(&ddecomp, MPI_COMM_WORLD);
    ActiveParticles act = init_empty_active_particles(PartManager);
    {
        ForceTreeMoments momtree(*PartManager, &ddecomp, nullptr, false, "");
        set_init_hsml(&momtree, PartManager->BoxSize, PartManager);
    }
    DriftKickTimes kick = {0};
    Cosmology CP = {0};
    CP.CMBTemperature = 2.7255;
    CP.Omega0 = 0.3;
    CP.OmegaLambda = 1- CP.Omega0;
    CP.OmegaBaryon = 0.045;
    CP.HubbleParam = 0.7;
    CP.RadiationOn = 0;
    CP.w0_fld = -1;
    struct UnitSystem units = get_unitsystem(3.085678e21, 1.989e43, 1e5);
    init_cosmology(&CP, 0.01, units);

    const int64_t NumPart = PartManager->NumPart;
    double * InitHsml = mymalloc("InitHsml", double, NumPart + 1);
    double * RefHsml = mymalloc("RefHsml", double, NumPart + 1);
    double * RefDensity = mymalloc("RefDensity", double, NumPart + 1);
    for(int64_t i = 0; i < NumPart; i++)
        InitHsml[i] = PartManager->Base[i].Hsml;

    /* First with the default export buffer. A new tree each time,
     * because the density walk updates hmax in the tree.*/
    {
        ForceTreeHmax tree(*PartManager, &ddecomp, GASMASK, "");
        density(&act, 1, 0, 0, kick, &data.timebinmgr, &CP, &(data.sph_pred.EntVarPred), NULL, &tree);
        slots_free_sph_pred_data(&data.sph_pred);
    }
    for(int64_t i = 0; i < NumPart; i++) {
        RefHsml[i] = PartManager->Base[i].Hsml;
        RefDensity[i] = SPHP(i).Density;
        PartManager->Base[i].Hsml = InitHsml[i];
    }
    /* Then with a small export buffer: room for a few hundred queries.*/
    {
        ForceTreeHmax tree(*PartManager, &ddecomp, GASMASK, "");
        density(&act, 1, 0, 0, kick, &data.timebinmgr, &CP, &(data.sph_pred.EntVarPred), NULL, &tree, false, 64 * 1024);
        slots_free_sph_pred_data(&data.sph_pred);
    }
    double maxhsmldiff = 0, maxdensdiff = 0;
    for(int64_t i = 0; i < NumPart; i++) {
        const double hdiff = fabs(PartManager->Base[i].Hsml / RefHsml[i] - 1);
        const double ddiff = fabs(SPHP(i).Density / RefDensity[i] - 1);
        BOOST_TEST(hdiff < 1e-6);
        BOOST_TEST(ddiff < 1e-6);
        maxhsmldiff = std::max(maxhsmldiff, hdiff);
        maxdensdiff = std::max(maxdensdiff, ddiff);
    }
    MPI_Allreduce(MPI_IN_PLACE, &maxhsmldiff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &maxdensdiff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    message(0, "Density with small export buffer: max relative diff in Hsml %g in Density %g\n", maxhsmldiff, maxdensdiff);
    check_densities(data.dp.MinGasHsmlFractional);

    myfree(RefDensity);
    myfree(RefHsml);
    myfree(InitHsml);
    domain_free(&ddecomp);
    slots_free(SlotsManager);
    myfree(PartManager->Base);
}

/* Set up a regular grid of gas with a coarser grid of stars offset from it.
 * Each rank keeps a contiguous slice of the gas, followed by a slice of the stars,
 * so that the particles are ordered by type as slots_setup_topology requires.
 * The gas has unit mass and density, so the SPH volume of each gas particle is one.*/
static void setup_stellar_density_particles(const int ngcbrt, const int nscbrt, const double starhsml)
{
    int ThisTask, NTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    MPI_Comm_size(MPI_COMM_WORLD, &NTask);
    const int64_t ngas = 1L * ngcbrt * ngcbrt * ngcbrt;
    const int64_t nstar = 1L * nscbrt * nscbrt * nscbrt;
    const double BoxSize = PartManager->BoxSize;
    const int64_t gstart = ThisTask * ngas / NTask;
    const int64_t gend = (ThisTask + 1) * ngas / NTask;
    const int64_t sstart = ThisTask * nstar / NTask;
    const int64_t send = (ThisTask + 1) * nstar / NTask;
    const int64_t nlocalgas = gend - gstart;
    PartManager->NumPart = nlocalgas + (send - sstart);
    #pragma omp parallel for
    for(int64_t i = 0; i < PartManager->NumPart; i++) {
        particle_data& part = PartManager->Base[i];
        memset(&part, 0, sizeof(struct particle_data));
        part.Mass = 1;
        if(i < nlocalgas) {
            const int64_t gid = gstart + i;
            part.ID = gid + 1;
            part.Hsml = 1.5 * BoxSize / ngcbrt;
            part.Pos[0] = (BoxSize/ngcbrt) * (gid/ngcbrt/ngcbrt);
            part.Pos[1] = (BoxSize/ngcbrt) * ((gid/ngcbrt) % ngcbrt);
            part.Pos[2] = (BoxSize/ngcbrt) * (gid % ngcbrt);
        }
        else {
            const int64_t sid = sstart + (i - nlocalgas);
            part.ID = ngas + sid + 1;
            part.Hsml = starhsml;
            /* Offset from the gas grid so that no gas particle is at zero distance or exactly on the edge of a kernel.*/
            part.Pos[0] = (BoxSize/nscbrt) * (sid/nscbrt/nscbrt) + 0.11 * BoxSize/ngcbrt;
            part.Pos[1] = (BoxSize/nscbrt) * ((sid/nscbrt) % nscbrt) + 0.07 * BoxSize/ngcbrt;
            part.Pos[2] = (BoxSize/nscbrt) * (sid % nscbrt) + 0.05 * BoxSize/ngcbrt;
        }
    }
    int64_t NLocal[6] = {0};
    NLocal[0] = nlocalgas;
    NLocal[4] = send - sstart;
    slots_setup_topology(PartManager, NLocal, SlotsManager);
    slots_setup_id(PartManager, SlotsManager);
}

/* Test the stellar density treewalk, which finds the smoothing length of each star and the SPH volume
 * of the gas inside it. Under MPI most stars are exported, so this also tests the distributed treewalk.
 * The walk is run twice: with the default export buffer, and with one small enough that it fills up
 * several times. The stellar density walk reduces its search radius as it finds neighbours, which
 * must not change the exports found by the later passes over the toptree. */
BOOST_AUTO_TEST_CASE(test_stellar_density)
{
    const int ngcbrt = 32, nscbrt = 16;
    const int64_t nstar = nscbrt * nscbrt * nscbrt;
    struct density_testdata data = setup_density(nstar);
    const double BoxSize = PartManager->BoxSize;
    /* Start well above the converged smoothing length, so that the first
     * toptree walk exports each star to more ranks than it finally needs.*/
    const double starhsml = 4 * BoxSize / ngcbrt;
    setup_stellar_density_particles(ngcbrt, nscbrt, starhsml);

    DomainDecomp ddecomp = {0};
    domain_decompose_full(&ddecomp, MPI_COMM_WORLD);
    ActiveParticles act = init_empty_active_particles(PartManager);

    /* The domain exchange has moved the particles, so set the slot data now.*/
    const int64_t NumPart = PartManager->NumPart;
    const int64_t nstarslot = SlotsManager->info[4].size;
    MyFloat * StarVolumeSPH = mymalloc("StarVolumeSPH", MyFloat, nstarslot + 1);
    MyFloat * RefVolumeSPH = mymalloc("RefVolumeSPH", MyFloat, nstarslot + 1);
    MyFloat * RefHsml = mymalloc("RefHsml", MyFloat, nstarslot + 1);
    /* Every star has significant mass return, so every star finds a density.*/
    MyFloat * MassReturn = mymalloc("MassReturn", MyFloat, nstarslot + 1);
    int64_t nlocalstar = 0;
    for(int64_t i = 0; i < NumPart; i++) {
        const particle_data& part = PartManager->Base[i];
        if(part.Type == 0)
            SPHP(i).Density = 1;
        else if(part.Type == 4) {
            STARP(i).TotalMassReturned = 0;
            MassReturn[part.PI] = 1;
            nlocalstar++;
        }
    }
    int64_t ntotstar;
    MPI_Allreduce(&nlocalstar, &ntotstar, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);
    BOOST_TEST(ntotstar == nstar);

    const int SPHWeighting = 0;
    const double MaxNgbDeviation = 0.5;
    const double DesNumNgb = GetNumNgb(GetDensityKernelType());
    std::optional<ForceTreeHmax> tree;
    tree.emplace(*PartManager, &ddecomp, GASMASK, "");

    /* First with the default export buffer.*/
    memset(StarVolumeSPH, 0, sizeof(MyFloat) * (nstarslot + 1));
    stellar_density(&act, StarVolumeSPH, MassReturn, &*tree, SPHWeighting, MaxNgbDeviation);
    for(int64_t i = 0; i < NumPart; i++) {
        const particle_data& part = PartManager->Base[i];
        if(part.Type != 4)
            continue;
        RefHsml[part.PI] = part.Hsml;
        RefVolumeSPH[part.PI] = StarVolumeSPH[part.PI];
    }

    /* Then from the same starting point with a small export buffer. This is room for a few hundred queries:
     * the minimum allowed is 20. With 4 ranks each rank exports more than a thousand queries,
     * so the buffer fills several times. With one rank there are no exports and this repeats the first walk.*/
    for(int64_t i = 0; i < NumPart; i++)
        if(PartManager->Base[i].Type == 4)
            PartManager->Base[i].Hsml = starhsml;
    memset(StarVolumeSPH, 0, sizeof(MyFloat) * (nstarslot + 1));
    stellar_density(&act, StarVolumeSPH, MassReturn, &*tree, SPHWeighting, MaxNgbDeviation, 64 * 1024);
    /* Free tree before checks so that we still recover if checks fail*/
    tree.reset();

    /* The gas has unit SPH volume and uniform number density, so without kernel weighting
     * the volume weight is the number of gas particles inside the smoothing length. */
    const double numdens = pow(ngcbrt / BoxSize, 3);
    double avghsml = 0, maxhsmldiff = 0, maxvoldiff = 0;
    for(int64_t i = 0; i < NumPart; i++) {
        const particle_data& part = PartManager->Base[i];
        if(part.Type != 4)
            continue;
        const double hsml = part.Hsml;
        const double vol = StarVolumeSPH[part.PI];
        BOOST_TEST(std::isfinite(hsml));
        BOOST_TEST(hsml > 0);
        BOOST_TEST(hsml < starhsml);
        BOOST_TEST(vol > 0);
        /* Counting particles on a grid inside a sphere is noisy, so this is a loose bound.*/
        BOOST_TEST(fabs(vol / (4 * M_PI / 3 * pow(hsml, 3) * numdens) - 1) < 0.25);
        /* The size of the export buffer must not change the answer.*/
        BOOST_TEST(fabs(hsml / RefHsml[part.PI] - 1) < 1e-6);
        BOOST_TEST(fabs(vol / RefVolumeSPH[part.PI] - 1) < 1e-6);
        maxhsmldiff = std::max(maxhsmldiff, fabs(hsml - RefHsml[part.PI]));
        maxvoldiff = std::max(maxvoldiff, fabs(vol - RefVolumeSPH[part.PI]));
        avghsml += hsml;
    }
    MPI_Allreduce(MPI_IN_PLACE, &avghsml, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    avghsml /= nstar;
    /* The kernel-weighted neighbour number is DesNumNgb for a uniform medium when
     * 4 pi/3 h^3 n = DesNumNgb, up to the discreteness of the grid.*/
    const double expectedhsml = cbrt(DesNumNgb / (4 * M_PI / 3 * numdens));
    message(0, "Stellar density: average Hsml %g expected %g. Max diff with small export buffer: Hsml %g volume %g\n", avghsml, expectedhsml, maxhsmldiff, maxvoldiff);
    BOOST_TEST(fabs(avghsml / expectedhsml - 1) < 0.05);

    myfree(MassReturn);
    myfree(RefHsml);
    myfree(RefVolumeSPH);
    myfree(StarVolumeSPH);
    domain_free(&ddecomp);
    slots_free(SlotsManager);
    myfree(PartManager->Base);
}
