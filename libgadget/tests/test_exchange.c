/*Simple test for the exchange function*/
#define BOOST_TEST_MODULE exchange
#include "booststub.h"

#define qsort_openmp qsort

#include <libgadget/exchange.h>
#include <libgadget/domain.h>
#include <libgadget/slotsmanager.h>
#include <libgadget/partmanager.h>
#include <libgadget/walltime.h>
#include <libgadget/utils/system.h>

int NTask, ThisTask;
int TotNumPart;

static struct ClockTable Clocks;

#define assert_all_true(x) BOOST_TEST(!MPIU_Any(!x, MPI_COMM_WORLD));

#define NUMPART1 8
static int
setup_particles(int64_t NType[6])
{
    walltime_init(&Clocks);
    MPI_Barrier(MPI_COMM_WORLD);
    particle_alloc_memory(PartManager, 8, 1024);
    int ptype;
    PartManager->NumPart = 0;
    for(ptype = 0; ptype < 6; ptype ++) {
        PartManager->NumPart += NType[ptype];
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    MPI_Comm_size(MPI_COMM_WORLD, &NTask);

    slots_init(0.01 * PartManager->MaxPart, SlotsManager);
    slots_set_enabled(0, sizeof(struct sph_particle_data), SlotsManager);
    slots_set_enabled(4, sizeof(struct star_particle_data), SlotsManager);
    slots_set_enabled(5, sizeof(struct bh_particle_data), SlotsManager);


    slots_reserve(1, NType, SlotsManager);

    slots_setup_topology(PartManager, NType, SlotsManager);

    int i;
    #pragma omp parallel for
    for(i = 0; i < PartManager->NumPart; i ++) {
        PartManager->Base[i].ID = i + PartManager->NumPart * ThisTask;
    }

    slots_setup_id(PartManager, SlotsManager);

    MPI_Allreduce(&PartManager->NumPart, &TotNumPart, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    return 0;
}

static int
teardown_particles(void)
{
    int TotNumPart2;

    int i;
    int nongarbage = 0, garbage = 0;
    for(i = 0; i < PartManager->NumPart; i ++) {
        if(!PartManager->Base[i].IsGarbage) {
            nongarbage++;
            BOOST_TEST (PartManager->Base[i].ID % NTask == 1Lu * ThisTask);
            continue;
        }
        else
            garbage++;
    }
    message(2, "curpart %d (np %ld) tot %d garbage %d\n", nongarbage, PartManager->NumPart, TotNumPart, garbage);
    MPI_Allreduce(&nongarbage, &TotNumPart2, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    BOOST_TEST(TotNumPart2 == TotNumPart);

    slots_free(SlotsManager);
    myfree(PartManager->Base);
    MPI_Barrier(MPI_COMM_WORLD);
    return 0;
}


static int
test_exchange_layout_func(int i, const void * userdata)
{
    return PartManager->Base[i].ID % NTask;
}

BOOST_AUTO_TEST_CASE(test_exchange)
{
    int64_t newSlots[6] = {NUMPART1, NUMPART1, NUMPART1, NUMPART1, NUMPART1, NUMPART1};

    setup_particles(newSlots);

    int fail = domain_exchange(&test_exchange_layout_func, NULL, NULL, PartManager, SlotsManager,10000, MPI_COMM_WORLD);

    assert_all_true(!fail);
#ifdef DEBUG
    slots_check_id_consistency(PartManager, SlotsManager);
#endif
    domain_test_id_uniqueness(PartManager);
    teardown_particles();
    return;
}

BOOST_AUTO_TEST_CASE(test_exchange_zero_slots)
{
    int64_t newSlots[6] = {NUMPART1, 0, NUMPART1, 0, NUMPART1, 0};

    setup_particles(newSlots);

    int fail = domain_exchange(&test_exchange_layout_func, NULL, NULL, PartManager, SlotsManager, 10000, MPI_COMM_WORLD);

    assert_all_true(!fail);
#ifdef DEBUG
    slots_check_id_consistency(PartManager, SlotsManager);
#endif
    domain_test_id_uniqueness(PartManager);

    teardown_particles();
    return;
}

BOOST_AUTO_TEST_CASE(test_exchange_with_garbage)
{
    int64_t newSlots[6] = {NUMPART1, NUMPART1, NUMPART1, NUMPART1, NUMPART1, NUMPART1};

    setup_particles(newSlots);

    slots_mark_garbage(0, PartManager, SlotsManager); /* watch out! this propagates the garbage flag to children */
    TotNumPart -= NTask;
    int fail = domain_exchange(&test_exchange_layout_func, NULL, NULL, PartManager, SlotsManager, 10000, MPI_COMM_WORLD);

    assert_all_true(!fail);

    domain_test_id_uniqueness(PartManager);
#ifdef DEBUG
    slots_check_id_consistency(PartManager, SlotsManager);
#endif
    teardown_particles();
    return;
}

static int
test_exchange_layout_func_uneven(int i, const void * userdata)
{
    if(PartManager->Base[i].Type == 0) return 0;

    return PartManager->Base[i].ID % NTask;
}

BOOST_AUTO_TEST_CASE(test_exchange_uneven)
{
    int64_t newSlots[6] = {NUMPART1, NUMPART1, NUMPART1, NUMPART1, NUMPART1, NUMPART1};

    setup_particles(newSlots);
    int i;

    /* this will trigger a slot growth on slot type 0 due to the inbalance */
    int fail = domain_exchange(&test_exchange_layout_func_uneven, NULL, NULL, PartManager, SlotsManager, 10000, MPI_COMM_WORLD);

    assert_all_true(!fail);

    if(ThisTask == 0) {
        /* the slot type must have grown automatically to handle the new particles. */
        BOOST_TEST(SlotsManager->info[0].size == NUMPART1 * NTask);
    }

#ifdef DEBUG
    slots_check_id_consistency(PartManager, SlotsManager);
#endif
    domain_test_id_uniqueness(PartManager);

    int TotNumPart2;

    int nongarbage = 0;
    for(i = 0; i < PartManager->NumPart; i ++) {
        if(!PartManager->Base[i].IsGarbage) {
            nongarbage++;
            if(PartManager->Base[i].Type == 0) {
                BOOST_TEST (ThisTask == 0);
            } else {
                BOOST_TEST(PartManager->Base[i].ID % NTask == 1Lu * ThisTask);
            }
            continue;
        }
    }
    MPI_Allreduce(&nongarbage, &TotNumPart2, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    BOOST_TEST(TotNumPart2 == TotNumPart);

    slots_free(SlotsManager);
    myfree(PartManager->Base);
    MPI_Barrier(MPI_COMM_WORLD);
    return;
}
