#define BOOST_TEST_MODULE slotsmanager

#include "booststub.h"

#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include <libgadget/partmanager.h>
#include <libgadget/domain.h>
#include <libgadget/slotsmanager.h>

struct part_manager_type PartManager[1] = {{0}};

static int
setup_particles(void)
{
    PartManager->MaxPart = 1024;
    PartManager->NumPart = 128 * 6;
    PartManager->BoxSize = 25000;

    int64_t newSlots[6] = {128, 128, 128, 128, 128, 128};

    PartManager->Base = (struct particle_data *) mymalloc("P", PartManager->MaxPart* sizeof(struct particle_data));
    memset(PartManager->Base, 0, sizeof(struct particle_data) * PartManager->MaxPart);

    slots_init(0.01 * PartManager->MaxPart, SlotsManager);
    int ptype;
    slots_set_enabled(0, sizeof(struct sph_particle_data), SlotsManager);
    slots_set_enabled(4, sizeof(struct star_particle_data), SlotsManager);
    slots_set_enabled(5, sizeof(struct bh_particle_data), SlotsManager);
    for(ptype = 1; ptype < 4; ptype++) {
        slots_set_enabled(ptype, sizeof(struct particle_data_ext), SlotsManager);
    }

    slots_reserve(1, newSlots, SlotsManager);

    slots_setup_topology(PartManager, newSlots, SlotsManager);

    int64_t i;
    #pragma omp parallel for
    for(i = 0; i < PartManager->NumPart; i ++) {
        int j;
        for(j = 0; j <3; j++)
            PartManager->Base[i].Pos[j] = i / PartManager->NumPart * PartManager->BoxSize;
        PartManager->Base[i].ID = i;
    }

    slots_setup_id(PartManager, SlotsManager);

    return 0;
}

static int
teardown_particles(void)
{
    slots_free(SlotsManager);
    myfree(PartManager->Base);
    return 0;
}

BOOST_AUTO_TEST_CASE(test_slots_gc)
{
    setup_particles();
    int i;
    int compact[6];
    for(i = 0; i < 6; i ++) {
        slots_mark_garbage(128 * i, PartManager, SlotsManager);
        compact[i] = 1;
    }
    slots_gc(compact, PartManager, SlotsManager);
    BOOST_TEST(PartManager->NumPart == 127 * i);

    BOOST_TEST(SlotsManager->info[0].size == 127);
    BOOST_TEST(SlotsManager->info[4].size == 127);
    BOOST_TEST(SlotsManager->info[5].size == 127);
#ifdef DEBUG
    slots_check_id_consistency(PartManager, SlotsManager);
#endif
    teardown_particles();
    return;
}

BOOST_AUTO_TEST_CASE(test_slots_gc_sorted)
{
    setup_particles();
    int i;
    for(i = 0; i < 6; i ++) {
        slots_mark_garbage(128 * i, PartManager, SlotsManager);
    }
    slots_gc_sorted(PartManager, SlotsManager);
    BOOST_TEST(PartManager->NumPart == 127 * i);

    BOOST_TEST(SlotsManager->info[0].size == 127);
    BOOST_TEST(SlotsManager->info[4].size == 127);
    BOOST_TEST(SlotsManager->info[5].size == 127);
    peano_t * Keys = (peano_t *) mymalloc("Keys", PartManager->NumPart * sizeof(peano_t));
    for(i = 0; i < PartManager->NumPart; i++) {
        Keys[i] = PEANO(PartManager->Base[i].Pos, PartManager->BoxSize);
        if(i >= 1) {
            BOOST_TEST(PartManager->Base[i].Type >=PartManager->Base[i-1].Type);
            if(PartManager->Base[i].Type == PartManager->Base[i-1].Type)
                BOOST_TEST(Keys[i] >= Keys[i-1]);
        }
    }
    myfree(Keys);
#ifdef DEBUG
    slots_check_id_consistency(PartManager, SlotsManager);
#endif
    teardown_particles();
    return;
}

BOOST_AUTO_TEST_CASE(test_slots_reserve)
{
    /* FIXME: these depends on the magic numbers in slots_reserve. After
     * moving those numbers to All.* we shall rework the code here. */
    setup_particles();

    int64_t newSlots[6] = {128, 128, 128, 128, 128, 128};
    int64_t oldSize[6];
    int ptype;
    for(ptype = 0; ptype < 6; ptype++) {
        oldSize[ptype] = SlotsManager->info[ptype].maxsize;
    }
    slots_reserve(1, newSlots, SlotsManager);

    /* shall not increase max size*/
    for(ptype = 0; ptype < 6; ptype++) {
        BOOST_TEST(oldSize[ptype] == SlotsManager->info[ptype].maxsize);
    }

    for(ptype = 0; ptype < 6; ptype++) {
        newSlots[ptype] += 1;
    }

    /* shall not increase max size; because it is small difference */
    slots_reserve(1, newSlots, SlotsManager);
    for(ptype = 0; ptype < 6; ptype++) {
        BOOST_TEST(oldSize[ptype] == SlotsManager->info[ptype].maxsize);
    }

    for(ptype = 0; ptype < 6; ptype++) {
        newSlots[ptype] += 8192;
    }

    /* shall increase max size; because it large difference */
    slots_reserve(1, newSlots, SlotsManager);

    for(ptype = 0; ptype < 6; ptype++) {
        BOOST_TEST(oldSize[ptype] < SlotsManager->info[ptype].maxsize);
    }
    teardown_particles();
}

/*Check that we behave correctly when the slot is empty*/
BOOST_AUTO_TEST_CASE(test_slots_zero)
{
    setup_particles();
    int i;
    int compact[6] = {1,0,0,0,1,1};
    for(i = 0; i < PartManager->NumPart; i ++) {
        slots_mark_garbage(i, PartManager, SlotsManager);
    }
    slots_gc(compact, PartManager, SlotsManager);
    BOOST_TEST(PartManager->NumPart == 0);
    BOOST_TEST(SlotsManager->info[0].size == 0);
    BOOST_TEST(SlotsManager->info[1].size == 128);
    BOOST_TEST(SlotsManager->info[4].size == 0);
    BOOST_TEST(SlotsManager->info[5].size == 0);

    teardown_particles();

    setup_particles();
    for(i = 0; i < PartManager->NumPart; i ++) {
        slots_mark_garbage(i, PartManager, SlotsManager);
    }
    slots_gc_sorted(PartManager, SlotsManager);
    BOOST_TEST(PartManager->NumPart == 0);
    BOOST_TEST(SlotsManager->info[0].size == 0);
    BOOST_TEST(SlotsManager->info[4].size == 0);
    BOOST_TEST(SlotsManager->info[5].size == 0);

    teardown_particles();

    return;

}

BOOST_AUTO_TEST_CASE(test_slots_fork)
{
    setup_particles();
    int i;
    for(i = 0; i < 6; i ++) {
        slots_split_particle(128 * i, 0, PartManager);
        slots_convert(128 * i, PartManager->Base[i * 128].Type, -1, PartManager, SlotsManager);

    }

    BOOST_TEST(PartManager->NumPart == 129 * i);

    BOOST_TEST(SlotsManager->info[0].size == 129);
    BOOST_TEST(SlotsManager->info[4].size == 129);
    BOOST_TEST(SlotsManager->info[5].size == 129);

    teardown_particles();
    return;
}

BOOST_AUTO_TEST_CASE(test_slots_convert)
{
    setup_particles();
    int i;
    for(i = 0; i < 6; i ++) {
        slots_convert(128 * i, PartManager->Base[i * 128].Type, -1, PartManager, SlotsManager);
    }

    BOOST_TEST(PartManager->NumPart == 128 * i);

    BOOST_TEST(SlotsManager->info[0].size == 129);
    BOOST_TEST(SlotsManager->info[4].size == 129);
    BOOST_TEST(SlotsManager->info[5].size == 129);

    teardown_particles();
    return;
}
