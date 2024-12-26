/*Simple test for the density treewalk*/
#define BOOST_TEST_MODULE density
#include "booststub.h"

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real_distribution.hpp>

#include <libgadget/partmanager.h>
#include <libgadget/walltime.h>
#include <libgadget/slotsmanager.h>
#include <libgadget/utils/mymalloc.h>
#include <libgadget/density.h>
#include <libgadget/domain.h>
#include <libgadget/forcetree.h>
#include <libgadget/timestep.h>
#include <libgadget/gravity.h>

static struct ClockTable CT;

/* The true struct for the state variable*/
struct density_testdata
{
    struct sph_pred_data sph_pred;
    DomainDecomp ddecomp;
    struct density_params dp;
};

/*Make a simple trivial domain for all data on a single processor*/
void trivial_domain(DomainDecomp * ddecomp)
{
    /* The whole tree goes into one topnode.
     * Set up just enough of the TopNode structure that
     * domain_get_topleaf works*/
    ddecomp->domain_allocated_flag = 1;
    ddecomp->NTopNodes = 1;
    ddecomp->NTopLeaves = 1;
    ddecomp->TopNodes = (struct topnode_data *) mymalloc("topnode", sizeof(struct topnode_data));
    ddecomp->TopNodes[0].Daughter = -1;
    ddecomp->TopNodes[0].Leaf = 0;
    ddecomp->TopLeaves = (struct topleaf_data *) mymalloc("topleaf",sizeof(struct topleaf_data));
    ddecomp->TopLeaves[0].Task = 0;
    /*These are not used*/
    ddecomp->TopNodes[0].StartKey = 0;
    ddecomp->TopNodes[0].Shift = BITS_PER_DIMENSION * 3;
    /*To tell the code we are in serial*/
    ddecomp->Tasks = (struct task_data *) mymalloc("task",sizeof(struct task_data));
    ddecomp->Tasks[0].StartLeaf = 0;
    ddecomp->Tasks[0].EndLeaf = 1;
}

static void free_domain(DomainDecomp * ddecomp) {
    myfree(ddecomp->Tasks);
    myfree(ddecomp->TopLeaves);
    myfree(ddecomp->TopNodes);
    slots_free(SlotsManager);
    myfree(PartManager->Base);
}

static struct density_testdata setup_density(void)
{
    struct density_testdata data = {0};
    /* Needed so the integer timeline works*/
    setup_sync_points(NULL,0.01, 0.1, 0.0, 0);

    /*Reserve space for the slots*/
    slots_init(0.01 * PartManager->MaxPart, SlotsManager);
    slots_set_enabled(0, sizeof(struct sph_particle_data), SlotsManager);
    slots_set_enabled(5, sizeof(struct sph_particle_data), SlotsManager);
    int64_t atleast[6] = {0};
    atleast[0] = pow(32,3);
    atleast[5] = 2;
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
    /*Set up the top-level domain grid*/
    trivial_domain(&data.ddecomp);
    data.dp.DensityResolutionEta = 1.;
    data.dp.BlackHoleNgbFactor = 2;
    data.dp.MaxNumNgbDeviation = 2;
    data.dp.DensityKernelType = DENSITY_KERNEL_CUBIC_SPLINE;
    data.dp.MinGasHsmlFractional = 0.006;
    struct gravshort_tree_params tree_params = {0};
    tree_params.FractionalGravitySoftening = 1;
    set_gravshort_treepar(tree_params);
    gravshort_set_softenings(1);
    data.dp.BlackHoleMaxAccretionRadius = 99999.;

    set_densitypar(data.dp);
    return data;
}

/* Perform some simple checks on the densities*/
static void check_densities(double MinGasHsml)
{
    int i;
    double maxHsml=P[0].Hsml, minHsml= P[0].Hsml;
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

static void do_density_test(struct density_testdata * data, const int numpart, double expectedhsml, double hsmlerr)
{
    int i, npbh=0;
    #pragma omp parallel for reduction(+: npbh)
    for(i=0; i<numpart; i++) {
        int j;
        PartManager->Base[i].Mass = 1;
        PartManager->Base[i].TimeBinHydro = 0;
        PartManager->Base[i].TimeBinGravity = 0;
        PartManager->Base[i].Ti_drift = 0;
        for(j=0; j<3; j++)
            PartManager->Base[i].Vel[j] = 1.5;
        if(PartManager->Base[i].Type == 0) {
            SPHP(i).Entropy = 1;
            SPHP(i).DtEntropy = 0;
            SPHP(i).Density = 1;
        }
        if(PartManager->Base[i].Type == 5)
            npbh++;
    }

    SlotsManager->info[0].size = numpart-npbh;
    SlotsManager->info[5].size = npbh;
    PartManager->NumPart = numpart;
    ActiveParticles act = init_empty_active_particles(PartManager);
    DomainDecomp ddecomp = data->ddecomp;

    ForceTree tree = {0};
    /* Finds fathers for each gas and BH particle, so need BH*/
    force_tree_rebuild_mask(&tree, &ddecomp, GASMASK+BHMASK, NULL);
    set_init_hsml(&tree, &ddecomp, PartManager->BoxSize);
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

    /* Rebuild without moments to check it works*/
    force_tree_rebuild_mask(&tree, &ddecomp, GASMASK, NULL);
    density(&act, 1, 0, 0, kick, &CP, &data->sph_pred, NULL, &tree);
    end = MPI_Wtime();
    double ms = (end - start)*1000;
    message(0, "Found densities in %.3g ms\n", ms);
    check_densities(data->dp.MinGasHsmlFractional);
    slots_free_sph_pred_data(&data->sph_pred);

    double avghsml = 0;
    #pragma omp parallel for reduction(+:avghsml)
    for(i=0; i<numpart; i++) {
        avghsml += PartManager->Base[i].Hsml;
    }
    message(0, "Average Hsml: %g Expected %g +- %g\n",avghsml/numpart, expectedhsml, hsmlerr);
    BOOST_TEST(fabs(avghsml/numpart - expectedhsml) < hsmlerr);
    /* Make MaxNumNgbDeviation smaller and check we get a consistent result.*/
    double * Hsml = (double *) mymalloc2("Hsml", numpart * sizeof(double));
    #pragma omp parallel for
    for(i=0; i<numpart; i++) {
        Hsml[i] = PartManager->Base[i].Hsml;
    }
    data->dp.MaxNumNgbDeviation = 0.5;
    set_densitypar(data->dp);

    start = MPI_Wtime();
    /*Find the density*/
    density(&act, 1, 0, 0, kick, &CP, &data->sph_pred, NULL, &tree);
    end = MPI_Wtime();
    slots_free_sph_pred_data(&data->sph_pred);

    ms = (end - start)*1000;
    message(0, "Found 1 dev densities in %.3g ms\n", ms);
    double diff = 0;
    double DesNumNgb = GetNumNgb(GetDensityKernelType());
    /* Free tree before checks so that we still recover if checks fail*/
    force_tree_free(&tree);

    for(i=0; i<numpart; i++) {
        BOOST_TEST(fabs(Hsml[i]/PartManager->Base[i].Hsml-1) < data->dp.MaxNumNgbDeviation / DesNumNgb);
        if(fabs(Hsml[i] - PartManager->Base[i].Hsml) > diff)
            diff = fabs(Hsml[i] - PartManager->Base[i].Hsml);
    }
    message(0, "Max diff between Hsml: %g\n",diff);
    myfree(Hsml);

    check_densities(data->dp.MinGasHsmlFractional);
}

BOOST_AUTO_TEST_CASE(test_density_flat)
{
    struct density_testdata data = setup_density();
    /*Set up the particle data*/
    int ncbrt = 32;
    int numpart = ncbrt*ncbrt*ncbrt;
    /* Create a regular grid of particles, 8x8x8, all of type 1,
     * in a box 8 kpc across.*/
    int i;
    #pragma omp parallel for
    for(i=0; i<numpart; i++) {
        PartManager->Base[i].Type = 0;
        PartManager->Base[i].PI = i;
        PartManager->Base[i].Hsml = 1.5*PartManager->BoxSize/cbrt(numpart);
        PartManager->Base[i].Pos[0] = (PartManager->BoxSize/ncbrt) * (i/ncbrt/ncbrt);
        PartManager->Base[i].Pos[1] = (PartManager->BoxSize/ncbrt) * ((i/ncbrt) % ncbrt);
        PartManager->Base[i].Pos[2] = (PartManager->BoxSize/ncbrt) * (i % ncbrt);
    }
    do_density_test(&data, numpart, 0.501747, 1e-4);
    free_domain(&data.ddecomp);
}

BOOST_AUTO_TEST_CASE(test_density_close)
{
    struct density_testdata data = setup_density();
    /*Set up the particle data*/
    int ncbrt = 32;
    int numpart = ncbrt*ncbrt*ncbrt;
    double close = 500.;
    int i;
    /* A few particles scattered about the place so the tree is not sparse*/
    #pragma omp parallel for
    for(i=0; i<numpart/4; i++) {
        PartManager->Base[i].Type = 0;
        PartManager->Base[i].PI = i;
        PartManager->Base[i].Hsml = 4*PartManager->BoxSize/cbrt(numpart/8);
        PartManager->Base[i].Pos[0] = (PartManager->BoxSize/ncbrt) * (i/(ncbrt/2.)/(ncbrt/2.));
        PartManager->Base[i].Pos[1] = (PartManager->BoxSize/ncbrt) * ((i*2/ncbrt) % (ncbrt/2));
        PartManager->Base[i].Pos[2] = (PartManager->BoxSize/ncbrt) * (i % (ncbrt/2));
    }

    /* Create particles clustered in one place, all of type 0.*/
    #pragma omp parallel for
    for(i=numpart/4; i<numpart; i++) {
        PartManager->Base[i].Type = 0;
        PartManager->Base[i].PI = i;
        PartManager->Base[i].Hsml = 2*ncbrt/close;
        PartManager->Base[i].Pos[0] = 4.1 + (i/ncbrt/ncbrt)/close;
        PartManager->Base[i].Pos[1] = 4.1 + ((i/ncbrt) % ncbrt) /close;
        PartManager->Base[i].Pos[2] = 4.1 + (i % ncbrt)/close;
    }
    P[numpart-1].Type = 5;
    P[numpart-1].PI = 0;

    do_density_test(&data, numpart, 0.131726, 1e-4);
    free_domain(&data.ddecomp);
}

void do_random_test(struct density_testdata * data, boost::random::mt19937 &r, const int numpart)
{
    boost::random::uniform_real_distribution<double> dist(0.0, 1.0);
    /* Create a randomly space set of particles, 8x8x8, all of type 0. */
    int i;
    for(i=0; i<numpart/4; i++) {
        PartManager->Base[i].Type = 0;
        PartManager->Base[i].PI = i;
        PartManager->Base[i].Hsml = PartManager->BoxSize/cbrt(numpart);

        int j;
        for(j=0; j<3; j++)
            PartManager->Base[i].Pos[j] = PartManager->BoxSize * dist(r);
    }
    for(i=numpart/4; i<3*numpart/4; i++) {
        PartManager->Base[i].Type = 0;
        PartManager->Base[i].PI = i;
        PartManager->Base[i].Hsml = PartManager->BoxSize/cbrt(numpart);
        int j;
        for(j=0; j<3; j++)
            PartManager->Base[i].Pos[j] = PartManager->BoxSize/2 + PartManager->BoxSize/8 * exp(pow(dist(r)-0.5,2));
    }
    for(i=3*numpart/4; i<numpart; i++) {
        PartManager->Base[i].Type = 0;
        PartManager->Base[i].PI = i;
        PartManager->Base[i].Hsml = PartManager->BoxSize/cbrt(numpart);
        int j;
        for(j=0; j<3; j++)
            PartManager->Base[i].Pos[j] = PartManager->BoxSize*0.1 + PartManager->BoxSize/32 * exp(pow(dist(r)-0.5,2));
    }
    do_density_test(data, numpart, 0.187515, 1e-3);
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
    free_domain(&data.ddecomp);
}
