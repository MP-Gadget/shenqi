#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdarg.h>
#include <string.h>
#include <math.h>

#include <omp.h>
#include "mymalloc.h"
#include "string.h"
#include "memory.h"
#include "system.h"
#include "endrun.h"

/* The main allocator is used to store large objects, e.g. tree, toptree */
Allocator A_MAIN[1];

/* The temp allocator is used to store objects that lives on the stack;
 * replacing alloca and similar cases to avoid stack induced memory fragmentation
 * */
Allocator A_TEMP[1];

void
tamalloc_init(void)
{
    int Nt = omp_get_max_threads();
    int NTask;
    MPI_Comm_size(MPI_COMM_WORLD, &NTask);

    /* Reserve 4MB, 512 bytes per thread, 128 bytes per task and 128 bytes per thread per task (for export) for TEMP storage.*/
    size_t n = 4096 * 1024 + 128 * NTask + 128 * Nt * NTask + 512 * Nt;

    message(0, "Reserving %td bytes per rank for TEMP memory allocator. \n", n);

    if (MPIU_Any(ALLOC_ENOMEMORY == allocator_init(A_TEMP, "TEMP", n, 1), MPI_COMM_WORLD)) {
        endrun(0, "Insufficient memory for the TEMP allocator on at least one nodes."
                  "Requestion %td bytes. Try reducing MaxMemSizePerNode. Also check the node health status.\n", n);

    }
}

void
mymalloc_init(int UseGPU)
{
    if (MPIU_Any(ALLOC_ENOMEMORY == allocator_malloc_init(A_MAIN, "MAIN", UseGPU), MPI_COMM_WORLD)) {
        endrun(0, "Insufficient memory for the MAIN allocator on at least one nodes.\n");
    }
}

static size_t highest_memory_usage = 0;

void report_detailed_memory_usage(const char *label, const char * fmt, ...)
{
    size_t memory_usage = allocator_get_used_size_malloc(A_MAIN);
    if(memory_usage < highest_memory_usage) {
        return;
    }

    MPI_Comm comm = MPI_COMM_WORLD;

    int ThisTask;
    MPI_Comm_rank(comm, &ThisTask);

    if (ThisTask != 0) {
        return;
    }

    highest_memory_usage = memory_usage;

    va_list va;
    va_start(va, fmt);
    char * buf = fastpm_strdup_vprintf(fmt, va);
    va_end(va);

    message(1, "Peak Memory usage induced by %s\n", buf);
    myfree(buf);
    allocator_print_malloc(A_MAIN);
}
