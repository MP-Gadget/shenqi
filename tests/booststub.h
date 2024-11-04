/* Boost::test common code for all tests*/
#include <boost/test/included/unit_test.hpp>

#include <mpi.h>
#include <libgadget/utils/mymalloc.h>
#include <libgadget/utils/endrun.h>

/* Generic MPI setup/teardown code*/
class MPIGlobalFixture {
public:
    MPIGlobalFixture() {};
    ~MPIGlobalFixture() {};
    void setup() {
        int thr;
        MPI_Init_thread(NULL, NULL, MPI_THREAD_FUNNELED, &thr);
        init_endrun(1);
        allocator_init(A_MAIN, "MAIN", 650 * 1024 * 1024, 0, NULL);
        allocator_init(A_TEMP, "TEMP", 8 * 1024 * 1024, 0, A_MAIN);
    }
    void teardown() {
        MPI_Finalize();
    }
};

BOOST_TEST_GLOBAL_FIXTURE(MPIGlobalFixture);

namespace tt = boost::test_tools;
