#define BOOST_TEST_MODULE memory

#include "booststub.h"

#include <libgadget/utils/memory.h>

static void
test_allocator(Allocator * A0)
{
    int * p1 = (int *) allocator_alloc_bot(A0, "M+1", 1024*sizeof(int));
    int * p2 = (int *) allocator_alloc_bot(A0, "M+2", 2048*sizeof(int));

    int * q1 = (int *) allocator_alloc_top(A0, "M-1", 1024*sizeof(int));
    int * q2 = (int *) allocator_alloc_top(A0, "M-2", 2048*sizeof(int));

    p1[0] = 1;
    p2[1] = 1;
    allocator_print(A0);

    q2[2000] = 1;
    q2 = (int *) allocator_realloc(A0, q2, 3072*sizeof(int));
    BOOST_TEST(q2[2000] == 1);
    /*Realloc to something smaller*/
    q2 = (int *) allocator_realloc(A0, q2, 2048*sizeof(int));
    BOOST_TEST(q2[2000]  == 1);

    /* Assert that reallocing does not move the base pointer.
     * Note this is true only for bottom allocations*/
    int * p2new = (int *) allocator_realloc(A0, p2, 3072*sizeof(int));
    BOOST_TEST(p2new == p2);

    BOOST_TEST(allocator_dealloc(A0, p1) == ALLOC_EMISMATCH);
    BOOST_TEST(allocator_dealloc(A0, q1) == ALLOC_EMISMATCH);

    BOOST_TEST(allocator_dealloc(A0, p2) == 0);
    BOOST_TEST(allocator_dealloc(A0, q2) == 0);

    allocator_free(p1);
    allocator_free(q1);

    allocator_print(A0);

    allocator_destroy(A0);
}

BOOST_AUTO_TEST_CASE(test_allocator_total)
{
 Allocator A0;
 allocator_init(&A0, "Default", 4096 * 1024, 1);
 test_allocator(&A0);
}

BOOST_AUTO_TEST_CASE(test_allocator_malloc)
{
    Allocator A0;
    allocator_malloc_init(&A0, "libc based", 4096 * 1024, 1);
    test_allocator(&A0);
}
