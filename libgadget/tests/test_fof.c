/*Simple test for the exchange function*/
#define BOOST_TEST_MODULE fof
#include "booststub.h"

#include <cmath>
#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real_distribution.hpp>

#define qsort_openmp qsort

#include <libgadget/fof.h>
#include <libgadget/walltime.h>
#include <libgadget/domain.h>
#include <libgadget/forcetree.h>
#include <libgadget/partmanager.h>

static struct ClockTable CT;

#define NUMPART1 8
static int
setup_particles(int NumPart, double BoxSize)
{

    int ThisTask, NTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    MPI_Comm_size(MPI_COMM_WORLD, &NTask);

    particle_alloc_memory(PartManager, BoxSize, 1.5 * NumPart);
    PartManager->NumPart = NumPart;

    slots_init(0.01 * PartManager->MaxPart, SlotsManager);
    slots_set_enabled(0, sizeof(struct sph_particle_data), SlotsManager);
    slots_set_enabled(4, sizeof(struct star_particle_data), SlotsManager);
    slots_set_enabled(5, sizeof(struct bh_particle_data), SlotsManager);

    int64_t newSlots[6] = {128, 0, 0, 0, 128, 128};
    slots_reserve(1, newSlots, SlotsManager);
    int i;
    #pragma omp parallel for
    for(i = 0; i < PartManager->NumPart; i ++) {
        PartManager->Base[i].ID = i + PartManager->NumPart * ThisTask;
        /* DM only*/
        PartManager->Base[i].Type = 1;
        PartManager->Base[i].Mass = 1;
        PartManager->Base[i].IsGarbage = 0;
        int j;
        for(j=0; j<3; j++) {
            PartManager->Base[i].Pos[j] = BoxSize * (j+1) * PartManager->Base[i].ID / (PartManager->NumPart * NTask);
            while(PartManager->Base[i].Pos[j] > BoxSize)
                PartManager->Base[i].Pos[j] -= BoxSize;
        }
    }
    fof_init(BoxSize/cbrt(PartManager->NumPart));
    /* TODO: Here create particles in some halo-like configuration*/
    return 0;
}

BOOST_AUTO_TEST_CASE(test_fof)
{
    int NTask;
    walltime_init(&CT);

    struct DomainParams dp = {0};
    dp.DomainOverDecompositionFactor = 1;
    dp.DomainUseGlobalSorting = 0;
    dp.TopNodeAllocFactor = 1.;
    dp.SetAsideFactor = 1;
    set_domain_par(dp);
    set_fof_testpar(1, 0.2, 5);
    init_forcetree_params(0.7);

    MPI_Comm_size(MPI_COMM_WORLD, &NTask);
    int NumPart = 512*512 / NTask;
    /* 20000 kpc*/
    double BoxSize = 20000;
    setup_particles(NumPart, BoxSize);

    /* Build a tree and domain decomposition*/
    DomainDecomp ddecomp = {0};
    domain_decompose_full(&ddecomp);

    FOFGroups fof = fof_fof(&ddecomp, 1, MPI_COMM_WORLD);

    /* Example assertion: this checks that the groups were allocated. */
    BOOST_TEST(fof.Group);
    BOOST_TEST(fof.TotNgroups == 1);
    /* Assert some more things about the particles,
     * maybe checking the halo properties*/

    fof_finish(&fof);
    domain_free(&ddecomp);
    slots_free(SlotsManager);
    myfree(PartManager->Base);
    return;
}
