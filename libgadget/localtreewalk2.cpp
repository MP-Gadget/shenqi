#include "localtreewalk2.h"
#include <omp.h>
#include "utils/endrun.h"
#include "utils/mymalloc.h"

/*!< Memory factor to leave for (N imported particles) > (N exported particles). */
static double ImportBufferBoost;

#define FACT1 0.366025403785    /* FACT1 = 0.5 * (sqrt(3)-1) */

#ifdef DEBUG
/*
 * for debugging
 */
#define WATCH { \
        printf("WorkSet[0] = %d (%d) %s:%d\n", WorkSet ? WorkSet[0] : 0, WorkSetSize, __FILE__, __LINE__); \
    }
static TreeWalk * GDB_current_ev = NULL;
#endif


/*Initialise global treewalk parameters*/
void set_treewalk_params(ParameterSet * ps)
{
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    if(ThisTask == 0)
        ImportBufferBoost = param_get_double(ps, "ImportBufferBoost");
    MPI_Bcast(&ImportBufferBoost, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

 /* 7/9/24: The code segfaults if the send/recv buffer is larger than 4GB in size.
  * Likely a 32-bit variable is overflowing but it is hard to debug. Easier to enforce a maximum buffer size.*/
size_t compute_bunchsize(const size_t query_type_elsize, const size_t result_type_elsize, const size_t MaxExportBufferBytes)
{
    /*The amount of memory eventually allocated per tree buffer*/
    size_t bytesperbuffer = sizeof(struct data_index) + query_type_elsize + result_type_elsize;
    /*This memory scales like the number of imports. In principle this could be much larger than Nexport
     * if the tree is very imbalanced and many processors all need to export to this one. In practice I have
     * not seen this happen, but provide a parameter to boost the memory for Nimport just in case.*/
    bytesperbuffer += ceil(ImportBufferBoost * (query_type_elsize + result_type_elsize));
    /*Use all free bytes for the tree buffer, as in exchange. Leave some free memory for array overhead.*/
    size_t freebytes = (size_t) mymalloc_freebytes();
    freebytes -= 4096 * 10 * bytesperbuffer;

    size_t BunchSize = (size_t) floor(((double)freebytes)/ bytesperbuffer);
    if(BunchSize * query_type_elsize > MaxExportBufferBytes)
        BunchSize = MaxExportBufferBytes / query_type_elsize;
    /* Per thread*/
    BunchSize /= omp_get_max_threads();

    if(freebytes <= 4096 * bytesperbuffer || BunchSize < 100) {
        endrun(1231245, "Not enough free memory to export particles: needed %ld bytes have %ld. Can export %ld \n", bytesperbuffer, freebytes, BunchSize);
    }
    return BunchSize;
}

/**
 * Cull a node.
 *
 * Returns 1 if the node shall be opened;
 * Returns 0 if the node has no business with this query.
 */
static int
cull_node(const double * const Pos, const double BoxSize, const double Hsml, const NgbTreeFindSymmetric symmetric, const struct NODE * const current)
{
    double dist;
    if(symmetric == NGB_TREEFIND_SYMMETRIC) {
        dist = DMAX(current->mom.hmax, Hsml) + 0.5 * current->len;
    } else {
        dist = Hsml + 0.5 * current->len;
    }

    double r2 = 0;
    double dx = 0;
    /* do each direction */
    int d;
    for(d = 0; d < 3; d ++) {
        dx = NEAREST(current->center[d] - Pos[d], BoxSize);
        if(dx > dist) return 0;
        if(dx < -dist) return 0;
        r2 += dx * dx;
    }
    /* now test against the minimal sphere enclosing everything */
    dist += FACT1 * current->len;

    if(r2 > dist * dist) {
        return 0;
    }
    return 1;
}
