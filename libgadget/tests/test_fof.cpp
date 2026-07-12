/*Tests for the friends-of-friends halo finder*/
#define BOOST_TEST_MODULE fof
#include "booststub.h"

#include <cmath>
#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <vector>

#include <libgadget/fof.h>
#include <libgadget/walltime.h>
#include <libgadget/domain.h>
#include <libgadget/forcetree.h>
#include <libgadget/partmanager.h>

static struct ClockTable CT;

static void setup_domain_params(void)
{
    struct DomainParams dp = {0};
    dp.DomainOverDecompositionFactor = 1;
    dp.TopNodeAllocFactor = 1.;
    dp.SetAsideFactor = 1;
    set_domain_par(dp);
    init_forcetree_params(0.7);
}

static void setup_slots(void)
{
    slots_init(0.01 * PartManager->MaxPart, SlotsManager);
    slots_set_enabled(0, sizeof(struct sph_particle_data), SlotsManager);
    slots_set_enabled(4, sizeof(struct star_particle_data), SlotsManager);
    slots_set_enabled(5, sizeof(struct bh_particle_data), SlotsManager);

    int64_t newSlots[6] = {128, 0, 0, 0, 128, 128};
    slots_reserve(1, newSlots, SlotsManager);
}

/* Gather the distributed group catalogue onto every rank so that
 * global properties can be checked.*/
static std::vector<struct Group> gather_groups(FOFGroups * fof)
{
    int NTask;
    MPI_Comm_size(MPI_COMM_WORLD, &NTask);
    std::vector<int> counts(NTask), displs(NTask);
    int nbytes = fof->Ngroups * sizeof(struct Group);
    MPI_Allgather(&nbytes, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    int total = 0, i;
    for(i = 0; i < NTask; i++) {
        displs[i] = total;
        total += counts[i];
    }
    std::vector<struct Group> allgroups(total / sizeof(struct Group));
    MPI_Allgatherv(fof->Group, nbytes, MPI_BYTE, allgroups.data(), counts.data(), displs.data(), MPI_BYTE, MPI_COMM_WORLD);
    return allgroups;
}

/* Distance between two points accounting for periodic wrapping*/
static double periodic_dist(double a, double b, double BoxSize)
{
    double d = fabs(a - b);
    if(d > BoxSize/2)
        d = BoxSize - d;
    return d;
}

static int
setup_particles(int NumPart, double BoxSize)
{

    int ThisTask, NTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    MPI_Comm_size(MPI_COMM_WORLD, &NTask);

    particle_alloc_memory(PartManager, BoxSize, 1.5 * NumPart);
    PartManager->NumPart = NumPart;

    setup_slots();
    int i;
    #pragma omp parallel for
    for(i = 0; i < PartManager->NumPart; i ++) {
        memset(&PartManager->Base[i], 0, sizeof(struct particle_data));
        PartManager->Base[i].ID = (i+1) + PartManager->NumPart * ThisTask;
        /* DM only*/
        PartManager->Base[i].Type = 1;
        PartManager->Base[i].Mass = 1;
        int j;
        for(j=0; j<3; j++) {
            PartManager->Base[i].Pos[j] = BoxSize * (j+1) * PartManager->Base[i].ID / (PartManager->NumPart * NTask);
            while(PartManager->Base[i].Pos[j] > BoxSize)
                PartManager->Base[i].Pos[j] -= BoxSize;
            PartManager->Base[i].Vel[j] = j+1;
        }
    }
    fof_init(BoxSize/cbrt(PartManager->NumPart));
    return 0;
}

/* All particles are in a single wrapped diagonal line with inter-particle
 * spacings much smaller than the linking length, so they form one group
 * containing every particle in the box.*/
BOOST_AUTO_TEST_CASE(test_fof_line)
{
    int NTask;
    walltime_init(&CT);

    setup_domain_params();
    set_fof_testpar(1, 0.2, 5);

    MPI_Comm_size(MPI_COMM_WORLD, &NTask);
    int NumPart = 512*512 / NTask;
    const int64_t GlobalNumPart = 1L * NumPart * NTask;
    /* 20000 kpc*/
    double BoxSize = 20000;
    setup_particles(NumPart, BoxSize);

    /* Build a tree and domain decomposition*/
    DomainDecomp ddecomp = {0};
    domain_decompose_full(&ddecomp, MPI_COMM_WORLD);

    FOFGroups fof = fof_fof(&ddecomp, 1, MPI_COMM_WORLD);

    BOOST_TEST(fof.Group);
    BOOST_TEST(fof.TotNgroups == 1);

    std::vector<struct Group> allgroups = gather_groups(&fof);
    BOOST_TEST((int64_t) allgroups.size() == fof.TotNgroups);
    if(allgroups.size() == 1) {
        const struct Group& g = allgroups[0];
        /* The group contains every particle in the box*/
        BOOST_TEST(g.Length == GlobalNumPart);
        BOOST_TEST(g.base.GrNr == 1);
        /* The lowest particle ID is 1*/
        BOOST_TEST(g.base.MinID == 1);
        BOOST_TEST(g.LenType[1] == GlobalNumPart);
        BOOST_TEST(g.LenType[0] == 0);
        BOOST_TEST(fabs(g.Mass - GlobalNumPart) < 1e-3 * GlobalNumPart);
        BOOST_TEST(fabs(g.MassType[1] - g.Mass) < 1e-6 * g.Mass);
        /* All particles have the same velocity, which is thus the group velocity*/
        int j;
        for(j = 0; j < 3; j++)
            BOOST_TEST(fabs(g.Vel[j] - (j+1)) < 1e-5);
    }
    /* Every particle is in the group*/
    int i;
    for(i = 0; i < PartManager->NumPart; i++)
        BOOST_TEST(PartManager->Base[i].GrNr == 1);

    fof_finish(&fof);
    domain_free(&ddecomp);
    slots_free(SlotsManager);
    myfree(PartManager->Base);
    return;
}

/* A controlled configuration of well-separated halos with known properties.
 * Halo h has 5+h members in a small ball around a grid of centers, so every
 * halo has a distinct, known length, mass, center of mass and velocity.
 * Halo 0 sits at the box corner, so it tests the periodic wrapping of the
 * group center of mass, and has exactly FOFHaloMinLength members. Isolated
 * single particles and pairs are below FOFHaloMinLength and must not
 * form groups. */
#define NHALO 64
#define NSINGLE 64
#define NPAIR 64
/* A uniform background of isolated particles. These join no group, but keep
 * the particle number per rank large enough that the force tree, whose node
 * budget scales with the local particle number, can resolve the halos.*/
#define NBGGRID 16

static int halo_size(const int h)
{
    return 5 + h;
}

/* Global index of the first particle in halo h; halo particles
 * come before the singles and pairs.*/
static int64_t halo_first(const int h)
{
    return 5L*h + (h*1L*(h-1))/2;
}

static void halo_center(const int h, double * center, const double BoxSize)
{
    center[0] = (BoxSize/4.) * (h % 4);
    center[1] = (BoxSize/4.) * ((h/4) % 4);
    center[2] = (BoxSize/4.) * (h / 16);
}

/* The halo of a global particle index, or -1 if it is one of the
 * singles or pairs at the end of the global set.*/
static int halo_of(const int64_t gid)
{
    if(gid >= halo_first(NHALO))
        return -1;
    int h;
    for(h = 0; h < NHALO; h++)
        if(gid < halo_first(h+1))
            return h;
    return -1;
}

BOOST_AUTO_TEST_CASE(test_fof_halos)
{
    int ThisTask, NTask;
    walltime_init(&CT);

    setup_domain_params();
    set_fof_testpar(1, 0.2, 5);
    /* Mean separation 100 kpc, so the linking length is 20 kpc.*/
    fof_init(100);

    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    MPI_Comm_size(MPI_COMM_WORLD, &NTask);
    const double BoxSize = 8000;
    const int64_t nhalopart = halo_first(NHALO);
    const int64_t nglobal = nhalopart + NSINGLE + 2*NPAIR + NBGGRID*NBGGRID*NBGGRID;

    /* Generate the global particle set, identically on every rank.
     * Positions are wrapped into the box.*/
    double * gpos = (double *) malloc(3*nglobal*sizeof(double));
    double expcm[NHALO][3] = {{0}};
    int h;
    int64_t i;
    for(h = 0; h < NHALO; h++) {
        double center[3];
        halo_center(h, center, BoxSize);
        int k;
        for(k = 0; k < halo_size(h); k++) {
            /* Halo members sit on a sub-grid with a 4 kpc spacing, well
             * inside the 20 kpc linking length, so each halo is one group.
             * A regular grid bounds the tree depth needed to separate them.*/
            const double off[3] = {4.*(k % 4) - 6., 4.*((k/4) % 4) - 6., 4.*(k/16) - 6.};
            int j;
            for(j = 0; j < 3; j++) {
                expcm[h][j] += off[j] / halo_size(h);
                double pp = center[j] + off[j];
                if(pp < 0)
                    pp += BoxSize;
                if(pp >= BoxSize)
                    pp -= BoxSize;
                gpos[3*(halo_first(h) + k) + j] = pp;
            }
        }
        int j;
        for(j = 0; j < 3; j++) {
            expcm[h][j] += center[j];
            if(expcm[h][j] < 0)
                expcm[h][j] += BoxSize;
            if(expcm[h][j] >= BoxSize)
                expcm[h][j] -= BoxSize;
        }
    }
    /* Isolated single particles, more than 1000 kpc from everything else*/
    for(i = 0; i < NSINGLE; i++) {
        gpos[3*(nhalopart + i)] = (BoxSize/4.) * (i % 4) + BoxSize/8.;
        gpos[3*(nhalopart + i) + 1] = (BoxSize/4.) * ((i/4) % 4) + BoxSize/8.;
        gpos[3*(nhalopart + i) + 2] = (BoxSize/4.) * (i / 16) + BoxSize/8.;
    }
    /* Isolated pairs: linked to each other but below FOFHaloMinLength*/
    for(i = 0; i < NPAIR; i++) {
        int j;
        for(j = 0; j < 2; j++) {
            const int64_t gid = nhalopart + NSINGLE + 2*i + j;
            gpos[3*gid] = (BoxSize/4.) * (i % 4) + BoxSize/8.;
            gpos[3*gid + 1] = (BoxSize/4.) * ((i/4) % 4) + BoxSize/8.;
            gpos[3*gid + 2] = (BoxSize/4.) * (i / 16) + 5.*j;
        }
    }
    /* The isolated background grid, offset so it stays far from
     * the halos, singles and pairs.*/
    for(i = 0; i < NBGGRID*NBGGRID*NBGGRID; i++) {
        const int64_t gid = nhalopart + NSINGLE + 2*NPAIR + i;
        gpos[3*gid] = (BoxSize/NBGGRID) * (i % NBGGRID) + BoxSize/(2.*NBGGRID);
        gpos[3*gid + 1] = (BoxSize/NBGGRID) * ((i/NBGGRID) % NBGGRID) + BoxSize/(2.*NBGGRID);
        gpos[3*gid + 2] = (BoxSize/NBGGRID) * (i / (NBGGRID*NBGGRID)) + BoxSize/(2.*NBGGRID);
    }

    /* Each rank keeps a contiguous slice of the global set*/
    const int64_t start = ThisTask * nglobal / NTask;
    const int64_t end = (ThisTask + 1) * nglobal / NTask;
    particle_alloc_memory(PartManager, BoxSize, nglobal);
    PartManager->NumPart = end - start;
    setup_slots();
    #pragma omp parallel for
    for(i = 0; i < PartManager->NumPart; i++) {
        const int64_t gid = start + i;
        memset(&PartManager->Base[i], 0, sizeof(struct particle_data));
        PartManager->Base[i].ID = gid + 1;
        PartManager->Base[i].Type = 1;
        PartManager->Base[i].Mass = 1.5;
        const int hh = halo_of(gid);
        int j;
        for(j = 0; j < 3; j++) {
            PartManager->Base[i].Pos[j] = gpos[3*gid + j];
            /* A known mean velocity for each halo*/
            if(hh >= 0)
                PartManager->Base[i].Vel[j] = (j+1) * (hh+1);
        }
    }
    free(gpos);

    DomainDecomp ddecomp = {0};
    domain_decompose_full(&ddecomp, MPI_COMM_WORLD);

    FOFGroups fof = fof_fof(&ddecomp, 1, MPI_COMM_WORLD);

    /* The singles and pairs are below FOFHaloMinLength, so only
     * the halos form groups.*/
    BOOST_TEST(fof.TotNgroups == NHALO);

    std::vector<struct Group> allgroups = gather_groups(&fof);
    BOOST_TEST((int64_t) allgroups.size() == fof.TotNgroups);

    int64_t grnr_of_halo[NHALO];
    int found[NHALO] = {0};
    for(h = 0; h < NHALO; h++)
        grnr_of_halo[h] = -1;
    for(const struct Group& g : allgroups) {
        /* Every group is one of the halos: the smallest ID in halo h is
         * halo_first(h)+1 because the halo members are contiguous in ID.*/
        const int hh = halo_of(g.base.MinID - 1);
        BOOST_REQUIRE(hh >= 0);
        BOOST_TEST(g.base.MinID == halo_first(hh) + 1);
        found[hh]++;
        grnr_of_halo[hh] = g.base.GrNr;
        BOOST_TEST(g.Length == halo_size(hh));
        BOOST_TEST(g.LenType[1] == halo_size(hh));
        BOOST_TEST(g.LenType[0] == 0);
        BOOST_TEST(fabs(g.Mass - 1.5*halo_size(hh)) < 1e-6 * g.Mass);
        BOOST_TEST(fabs(g.MassType[1] - g.Mass) < 1e-6 * g.Mass);
        int j;
        for(j = 0; j < 3; j++) {
            /* All halo members have the same velocity, which is thus the group velocity*/
            BOOST_TEST(fabs(g.Vel[j] - (j+1)*(hh+1)) < 1e-5 * (hh+1));
            /* The center of mass, allowing for the periodic wrap:
             * halo 0 straddles the box corner.*/
            const double cm = g.CM[j] - PartManager->CurrentParticleOffset[j];
            BOOST_TEST(periodic_dist(cm, expcm[hh][j], BoxSize) < 0.01);
        }
    }
    /* Each halo was found exactly once*/
    for(h = 0; h < NHALO; h++)
        BOOST_TEST(found[h] == 1);
    /* Groups are numbered in order of decreasing length. The halo
     * lengths are all distinct, so the group numbers are fully determined:
     * the largest halo, h = NHALO-1, has GrNr 1.*/
    for(h = 0; h < NHALO; h++)
        BOOST_TEST(grnr_of_halo[h] == NHALO - h);

    /* Check the group number stored in the particles: halo members
     * store their halo's group, singles and pairs are not in groups.*/
    for(i = 0; i < PartManager->NumPart; i++) {
        const int hh = halo_of(PartManager->Base[i].ID - 1);
        if(hh >= 0)
            BOOST_TEST(PartManager->Base[i].GrNr == grnr_of_halo[hh]);
        else
            BOOST_TEST(PartManager->Base[i].GrNr == -1);
    }

    fof_finish(&fof);
    domain_free(&ddecomp);
    slots_free(SlotsManager);
    myfree(PartManager->Base);
}
