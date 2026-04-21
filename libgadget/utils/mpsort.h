#ifndef _UTILS_MPSORT_H
#define _UTILS_MPSORT_H

#include <stdio.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#include <algorithm>
#include <numeric>
#include <vector>
#include <execution>
#include <bit>

#include <mpi.h>

#include "system.h"
#include "mymalloc.h"
#include "endrun.h"

struct crstruct {
    size_t size;
    void * arg;
    void (*radix)(const void * ptr, void * radix, void * arg);
};

template <typename T>
int _compar_radix (const T& u1, const T& u2) {
    if constexpr(sizeof(T) <= 8) {
        return (signed) (u1 > u2) - (signed) (u1 < u2);
    }
    else {
        /* from most significant */
        const uint64_t * pu1 = (const uint64_t *) &u1;
        const uint64_t * pu2 = (const uint64_t *) &u2;
        int dir = 1;
        if constexpr(std::endian::native == std::endian::little) {
            dir = -1;
            /* Last byte first*/
            pu1 = (const uint64_t *) ((&u1)+1);
            pu1--;
            pu2 = (const uint64_t *) ((&u2)+1);
            pu2--;
        }
        for(size_t i = 0; i < sizeof(T); i += 8) {
            if(*pu1 < *pu2) return -1;
            if(*pu1 > *pu2) return 1;
            pu1 += dir;
            pu2 += dir;
        }
        return 0;
    }
}

template <typename T>
void _bisect_radix (T * u, const T * u1, const T * u2) {
    if constexpr(sizeof(T) <= 8) {
        *u = *u1 + ((*u2 - *u1) >> 1);
    } else {
        constexpr size_t rsize = sizeof(T);
        const unsigned char * cu1 = (const unsigned char *) u1;
        const unsigned char * cu2 = (const unsigned char *) u2;
        unsigned char * cu = (unsigned char *) u;
        unsigned int carry = 0;
        int dir;
        if constexpr(std::endian::native == std::endian::little) {
            dir = -1;
        } else {
            dir = 1;
            cu1 += rsize - 1;
            cu2 += rsize - 1;
            cu  += rsize - 1;
        }
        /* from most significant */
        for(size_t i = 0; i < rsize; i ++) {
            unsigned int tmp = (unsigned int) *cu2 + *cu1 + carry;
            if(tmp >= 256) carry = 1;
            else carry = 0;
            *cu = tmp % (UINT8_MAX+1);
            cu -= dir;
            cu1 -= dir;
            cu2 -= dir;
        }
        cu += dir;
        for(size_t i = 0; i < rsize; i ++) {
            unsigned int tmp = *cu + carry * 256;
            carry = tmp & 1;
            *cu = (tmp >> 1) ;
            cu += dir;
        }
    }
}

template <size_t rsize>
void _setup_radix_sort(
        struct crstruct *d,
        size_t size,
        void (*radix)(const void * ptr, void * radix, void * arg),
        void * arg) {
    d->arg = arg;
    d->radix = radix;
    d->size = size;
}

/****
 * sort by radix using std::sort.
 *
 * The sort is performed by pre-computing the radix keys for every element,
 * sorting an index array according to those keys, and then rearranging
 * the underlying data.
 **** */

typedef void (*radix_fn)(const void * ptr, void * radix, void * arg);

template <typename T>
void radix_sort(void * base, size_t nmemb, size_t size,
        radix_fn radix, void * arg) {

    if (nmemb < 2) return;

    /* We enforce that rsize is a multiple of 8, so we can do
     * comparisons 64-bits at a time, which is much faster.
     * It is easy enough to pad large integers.*/
    static_assert(sizeof(T) < 8 || sizeof(T) % 8  == 0);

    char * cbase = static_cast<char *>(base);

    /* Pre-compute all radix keys, building a std::pair. */
    std::vector<std::pair<T, size_t> > keys(nmemb);
    #pragma omp parallel for
    for (size_t i = 0; i < nmemb; i++) {
        radix(cbase + i * size, &keys[i].first, arg);
        keys[i].second = i;
    }

    /* Sort by key (first part of the pair). For small types we can use operator < directly,
     * for large types we need the comparison function*/
    if constexpr(sizeof(T) <= 8) {
        std::sort(std::execution::par_unseq, keys.begin(), keys.end());
    }
    else {
        std::sort(keys.begin(), keys.end(), [](const auto& a, const auto& b) {
            return _compar_radix<T>(a.first, b.first) < 0;
        });
    }

    /* Rearrange the data into a temporary buffer according to indices,
     * then memcpy back to base. */
    char * tmp = (char *) mymalloc("tmp", nmemb * size);
    #pragma omp parallel for
    for (size_t i = 0; i < nmemb; i++) {
        memcpy(tmp + i * size, cbase + keys[i].second * size, size);
    }
    memcpy(cbase, tmp, nmemb * size);
    myfree(tmp);
}

/*
 * returns index of the last item satisfying
 * [item] < P,
 *
 * returns -1 if [all] < P
 * */

template <typename T>
ptrdiff_t _bsearch_last_lt(T& P,
    void * base, size_t nmemb,
    struct crstruct * d) {

    if (nmemb == 0) return -1;

    T tmpradix;
    ptrdiff_t left = 0;
    ptrdiff_t right = nmemb - 1;

    d->radix((char*) base, &tmpradix, d->arg);
    if(_compar_radix<T>(tmpradix, P) >= 0) {
        return - 1;
    }
    d->radix((char*) base + right * d->size, &tmpradix, d->arg);
    if(_compar_radix<T>(tmpradix, P) < 0) {
        return nmemb - 1;
    }

    /* left <= i <= right*/
    /* [left] < P <= [right] */
    while(right > left + 1) {
        ptrdiff_t mid = ((right - left + 1) >> 1) + left;
        d->radix((char*) base + mid * d->size, &tmpradix, d->arg);
        /* if [mid] < P , move left to mid */
        /* if [mid] >= P , move right to mid */
        int c1 = _compar_radix<T>(tmpradix, P);
        if(c1 < 0) {
            left = mid;
        } else {
            right = mid;
        }
    }
    return left;
}

/*
 * returns index of the last item satisfying
 * [item] <= P,
 *
 * */
template <typename T>
ptrdiff_t _bsearch_last_le(T& P,
    void * base, size_t nmemb,
    struct crstruct * d) {

    if (nmemb == 0) return -1;

    T tmpradix;
    ptrdiff_t left = 0;
    ptrdiff_t right = nmemb - 1;

    d->radix((char*) base, &tmpradix, d->arg);
    if(_compar_radix<T>(tmpradix, P) > 0) {
        return -1;
    }
    d->radix((char*) base + right * d->size, &tmpradix, d->arg);
    if(_compar_radix<T>(tmpradix, P) <= 0) {
        return nmemb - 1;
    }

    /* left <= i <= right*/
    /* [left] <= P < [right] */
    while(right > left + 1) {
        ptrdiff_t mid = ((right - left + 1) >> 1) + left;
        d->radix((char*) base + mid * d->size, &tmpradix, d->arg);
        /* if [mid] <= P , move left to mid */
        /* if [mid] > P , move right to mid*/
        int c1 = _compar_radix<T>(tmpradix, P);
        if(c1 <= 0) {
            left = mid;
        } else {
            right = mid;
        }
    }
    return left;
}

/*
 * do a histogram of mybase, based on bins defined in P.
 * P is an array of radix of length Plength,
 * myCLT, myCLE are of length Plength + 2
 *
 * myCLT[i + 1] is the count of items less than P[i]
 * myCLE[i + 1] is the count of items less than or equal to P[i]
 *
 * myCLT[0] is always 0
 * myCLT[Plength + 1] is always mynmemb
 *
 * */
template <typename T>
void _histogram(T* P, int Plength, void * mybase, size_t mynmemb,
        ptrdiff_t * myCLT, ptrdiff_t * myCLE,
        struct crstruct * d) {
    int it;

    if(myCLT) {
        myCLT[0] = 0;
        for(it = 0; it < Plength; it ++) {
            /* No need to start from the beginging of mybase, since myubase and P are both sorted */
            ptrdiff_t offset = myCLT[it];
            myCLT[it + 1] = _bsearch_last_lt<T>(P[it],
                            ((char*) mybase) + offset * d->size,
                            mynmemb - offset, d)
                            + 1 + offset;
        }
        myCLT[it + 1] = mynmemb;
    }
    if(myCLE) {
        myCLE[0] = 0;
        for(it = 0; it < Plength; it ++) {
            /* No need to start from the beginging of mybase, since myubase and P are both sorted */
            ptrdiff_t offset = myCLE[it];
            myCLE[it + 1] = _bsearch_last_le<T>(P[it],
                            ((char*) mybase) + offset * d->size,
                            mynmemb - offset, d)
                            + 1 + offset;
        }
        myCLE[it + 1] = mynmemb;
    }
}

template <typename T>
struct piter {
    int * stable;
    int * narrow;
    int Plength;
    T * Pleft;
    T * Pright;
    struct crstruct * d;

    piter(T& Pmin, T& Pmax, int Plength_i, struct crstruct * d_i): Plength(Plength_i), d(d_i)
    {
        stable = ta_malloc("stable", int, Plength);
        memset(stable, 0, Plength * sizeof(int));
        narrow = ta_malloc("narrow", int, Plength);
        memset(narrow, 0, Plength * sizeof(int));
        Pleft = (T* ) mymalloc("left", Plength * sizeof(T));
        Pright = (T*) mymalloc("right", Plength * sizeof(T));
        for(int i = 0; i < Plength; i ++) {
            Pleft[i] = Pmin;
            Pright[i] = Pmax;
        }
    }

    ~piter()
    {
        myfree(Pright);
        myfree(Pleft);
        myfree(narrow);
        myfree(stable);
    }

    /*
    * this will bisect the left / right in piter.
    * note that piter goes [left, right], thus we need
    * to maintain an internal status to make sure we go over
    * the additional 'right]'. (usual bisect range is
    * '[left, right)' )
    * */
    void bisect(T* P)
    {
        for(int i = 0; i < Plength; i ++) {
            if(stable[i])
                continue;
            if(narrow[i]) {
                /* The last iteration, test Pright directly */
                P[i] = Pright[i];
                stable[i] = 1;
            } else {
                /* ordinary iteration */
                _bisect_radix<T>(&P[i],
                        &Pleft[i],
                        &Pright[i]);
                /* in case the bisect can't move P beyond left,
                * the range is too small, so we set flag narrow,
                * and next iteration we will directly test Pright */
                if(_compar_radix<T>(P[i], Pleft[i]) <= 0) {
                    narrow[i] = 1;
                }
            }
        #if 0
            printf("bisect %d %u %u %u\n", i, *(int*) &P[i * d->rsize],
                    *(int*) &pi->Pleft[i * d->rsize],
                    *(int*) &pi->Pright[i * d->rsize]);
        #endif
        }
    }

    int all_done()
    {
        int done = 1;
        #if 0
        #pragma omp single
        for(i = 0; i < pi->Plength; i ++) {
            printf("P %d stable %d narrow %d\n",
                i, pi->stable[i], pi->narrow[i]);
        }
        #endif
        for(int i = 0; i < Plength; i ++) {
            if(!stable[i]) {
                done = 0;
                break;
            }
        }
        return done;
    }

/*
 * bisection acceptance test.
 *
 * test if the counts satisfies CLT < C <= CLE.
 * move Pleft / Pright accordingly.
 * */
    void accept(T* P, ptrdiff_t * C, ptrdiff_t * CLT, ptrdiff_t * CLE) {
        #if 0
        for(i = 0; i < pi->Plength + 1; i ++) {
            printf("counts %d LT %ld C %ld LE %ld\n",
                    i, CLT[i], C[i], CLE[i]);
        }
        #endif
        for(int i = 0; i < Plength; i ++) {
            if( CLT[i + 1] < C[i + 1] && C[i + 1] <= CLE[i + 1]) {
                stable[i] = 1;
                continue;
            }
            if(CLT[i + 1] >= C[i + 1]) {
                /* P[i] is too big */
                Pright[i] = P[i];
            } else {
                /* P[i] is too small */
                Pleft[i] = P[i];
            }
        }
    }
};

/* mpi version of radix sort;
 *
 * each caller provides the distributed array and number of items.
 * the sorted array is returned to the original array pointed to by
 * mybase. (AKA no rebalancing is done.)
 *
 * NOTE: may need an api to return a balanced array!
 *
 * uses the same amount of temporary storage space for communication
 * and local sort. (this will be allocated via malloc)
 *
 *
 * */

static MPI_Datatype MPI_TYPE_PTRDIFF = 0;

struct crmpistruct {
    MPI_Datatype MPI_TYPE_RADIX;
    MPI_Datatype MPI_TYPE_DATA;
    MPI_Comm comm;
    void * mybase;
    void * myoutbase;
    size_t mynmemb;
    size_t nmemb;
    size_t myoutnmemb;
    size_t outnmemb;
    int NTask;
    int ThisTask;
};

template <typename T>
void
_setup_mpsort_mpi(struct crmpistruct * o,
                  struct crstruct * d,
                  void * base,
                  size_t nmemb,
                  void * myoutbase, size_t myoutnmemb,
                  MPI_Comm comm)
{

    o->comm = comm;

    MPI_Comm_size(comm, &o->NTask);
    MPI_Comm_rank(comm, &o->ThisTask);

    o->mybase = base;
    o->mynmemb = nmemb;
    o->myoutbase = myoutbase;
    o->myoutnmemb = myoutnmemb;

    MPI_Allreduce(&o->mynmemb, &o->nmemb, 1, MPI_TYPE_PTRDIFF, MPI_SUM, comm);
    MPI_Allreduce(&o->myoutnmemb, &o->outnmemb, 1, MPI_TYPE_PTRDIFF, MPI_SUM, comm);

    if(o->outnmemb != o->nmemb) {
        endrun(4, "total number of items in the item does not match the input %ld != %ld\n", o->outnmemb, o->nmemb);
    }

    MPI_Type_contiguous(sizeof(T), MPI_BYTE, &o->MPI_TYPE_RADIX);
    MPI_Type_commit(&o->MPI_TYPE_RADIX);

    MPI_Type_contiguous(d->size, MPI_BYTE, &o->MPI_TYPE_DATA);
    MPI_Type_commit(&o->MPI_TYPE_DATA);

}
static void _destroy_mpsort_mpi(struct crmpistruct * o) {
    MPI_Type_free(&o->MPI_TYPE_RADIX);
    MPI_Type_free(&o->MPI_TYPE_DATA);
}

static int _solve_for_layout_mpi (
        int NTask,
        ptrdiff_t * C,
        ptrdiff_t * myT_CLT,
        ptrdiff_t * myT_CLE,
        ptrdiff_t * myT_C,
        MPI_Comm comm);

static int
_assign_colors(size_t glocalsize, size_t * sizes, int * ncolor, MPI_Comm comm)
{
    int NTask;
    int ThisTask;
    MPI_Comm_rank(comm, &ThisTask);
    MPI_Comm_size(comm, &NTask);

    int i;
    int mycolor = -1;
    size_t current_size = 0;
    size_t current_outsize = 0;
    int current_color = 0;
    int lastcolor = 0;
    for(i = 0; i < NTask; i ++) {
        current_size += sizes[i];

        lastcolor = current_color;

        if(i == ThisTask) {
            mycolor = lastcolor;
        }

        if(current_size > glocalsize || current_outsize > glocalsize) {
            current_size = 0;
            current_outsize = 0;
            current_color ++;
        }
    }
    /* no data for color of -1; exclude them later with special cases */
    if(sizes[ThisTask] == 0) {
        mycolor = -1;
    }

    *ncolor = lastcolor + 1;
    return mycolor;
}

struct SegmentGroupDescr {
    /* data model: rank <- segment <- group */
    int Ngroup;
    int Nsegments;
    int GroupID; /* ID of the group of this rank */
    int ThisSegment; /* SegmentID of the local data chunk on this rank*/

    size_t totalsize;
    int segment_start; /* segments responsible in this group */
    int segment_end;

    int is_group_leader;
    int group_leader_rank;
    int segment_leader_rank;
    MPI_Comm Group;  /* communicator for all ranks in the group */
    MPI_Comm Leaders; /* communicator for all ranks by leaders vs nonleaders */
    MPI_Comm Segment; /* communicator for all ranks in this segment */
};

/* Find the rank that has the value of MPI_MIN, or MPI_MAX.
 * If there is degeneracy, return the lower rank.
 * Avoids MPI_MINLOC and MPI_MAXLOC.
 * */
static int
MPIU_GetLoc(const void * base, MPI_Datatype type, MPI_Op op, MPI_Comm comm)
{
    ptrdiff_t lb;
    ptrdiff_t elsize;
    MPI_Type_get_extent(type, &lb, &elsize);

    void * tmp = malloc(elsize);
    /* find the result of the reduction. */
    MPI_Allreduce(base, tmp, 1, type, op, comm);

    int ThisTask;
    int NTask;
    MPI_Comm_size(comm, &NTask);
    MPI_Comm_rank(comm, &ThisTask);
    int rank = NTask;
    int ret = -1;
    if (memcmp(base, tmp, elsize) == 0) {
        rank = ThisTask;
    }
    /* find the rank that is the same as the reduction result */
    /* avoid MPI_IN_PLACE, since if we are using this code, we have assumed we are using
     * a crazy MPI impl...
     * */
    MPI_Allreduce(&rank, &ret, 1, MPI_INT, MPI_MIN, comm);
    free(tmp);
    return ret;
}

static void
_create_segment_group(struct SegmentGroupDescr * descr, size_t * sizes, size_t avgsegsize, int Ngroup, MPI_Comm comm)
{
    int ThisTask, NTask;

    MPI_Comm_size(comm, &NTask);
    MPI_Comm_rank(comm, &ThisTask);

    descr->ThisSegment = _assign_colors(avgsegsize, sizes, &descr->Nsegments, comm);

    if(descr->ThisSegment >= 0) {
        /* assign segments to groups.
         * if Nsegments < Ngroup, some groups will have no segments, and thus no ranks belong to them. */
        descr->GroupID = ((size_t) descr->ThisSegment) * Ngroup / descr->Nsegments;
    } else {
        descr->GroupID = Ngroup + 1;
        descr->ThisSegment = NTask + 1;
    }

    descr->Ngroup = Ngroup;

    MPI_Comm_split(comm, descr->GroupID, ThisTask, &descr->Group);

    MPI_Allreduce(&descr->ThisSegment, &descr->segment_start, 1, MPI_INT, MPI_MIN, descr->Group);
    MPI_Allreduce(&descr->ThisSegment, &descr->segment_end, 1, MPI_INT, MPI_MAX, descr->Group);

    descr->segment_end ++;

    int rank;

    MPI_Comm_rank(descr->Group, &rank);

    /* rank with most data in a group is the leader of the group. */
    descr->group_leader_rank = MPIU_GetLoc(&sizes[ThisTask], MPI_LONG, MPI_MAX, descr->Group);

    descr->is_group_leader = rank == descr->group_leader_rank;

    MPI_Comm_split(comm, (rank == descr->group_leader_rank)? 0 : 1, ThisTask, &descr->Leaders);

    MPI_Comm_split(descr->Group, descr->ThisSegment, ThisTask, &descr->Segment);

    /* rank with least data in a segment is the leader of the segment. */
    descr->segment_leader_rank = MPIU_GetLoc(&sizes[ThisTask], MPI_LONG, MPI_MIN, descr->Segment);
}

static void
_destroy_segment_group(struct SegmentGroupDescr * descr)
{
    MPI_Comm_free(&descr->Segment);
    MPI_Comm_free(&descr->Group);
    MPI_Comm_free(&descr->Leaders);
}

static void
MPIU_Scatter (MPI_Comm comm, int root, const void * sendbuffer, void * recvbuffer, int nrecv, size_t elsize, int * totalnsend);
static void
MPIU_Gather (MPI_Comm comm, int root, const void * sendbuffer, void * recvbuffer, int nsend, size_t elsize, int * totalnrecv);

static uint64_t
checksum(void * base, size_t nbytes, MPI_Comm comm)
{
    uint64_t sum = 0;
    char * ptr = (char *) base;
    size_t i;
    for(i = 0; i < nbytes; i ++) {
        sum += ptr[i];
    }
    MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_LONG, MPI_SUM, comm);
    return sum;
}

static void
MPIU_Gather (MPI_Comm comm, int root, const void * sendbuffer, void * recvbuffer, int nsend, size_t elsize, int * totalnrecv)
{
    int NTask;
    int ThisTask;

    MPI_Comm_size(comm, &NTask);
    MPI_Comm_rank(comm, &ThisTask);

    MPI_Datatype dtype;
    MPI_Type_contiguous(elsize, MPI_BYTE, &dtype);
    MPI_Type_commit(&dtype);

    int * recvcount = ta_malloc("recvcount", int, NTask);
    int * rdispls = ta_malloc("rdispls", int, NTask+1);
    int i;
    MPI_Gather(&nsend, 1, MPI_INT, recvcount, 1, MPI_INT, root, comm);

    rdispls[0] = 0;
    for(i = 1; i <= NTask; i ++) {
        rdispls[i] = rdispls[i - 1] + recvcount[i - 1];
    }

    if(ThisTask == root) {
        if(totalnrecv)
            *totalnrecv = rdispls[NTask];
    } else {
        if(totalnrecv)
            *totalnrecv = 0;
    }

    MPI_Gatherv(sendbuffer, nsend, dtype, recvbuffer, recvcount, rdispls, dtype, root, comm);

    ta_free(rdispls);
    ta_free(recvcount);
    MPI_Type_free(&dtype);
}

static void
MPIU_Scatter (MPI_Comm comm, int root, const void * sendbuffer, void * recvbuffer, int nrecv, size_t elsize, int * totalnsend)
{
    int NTask;
    int ThisTask;
    MPI_Comm_size(comm, &NTask);
    MPI_Comm_rank(comm, &ThisTask);

    MPI_Datatype dtype;
    MPI_Type_contiguous(elsize, MPI_BYTE, &dtype);
    MPI_Type_commit(&dtype);

    int * sendcount = ta_malloc("sendcount", int, NTask);
    int * sdispls = ta_malloc("sdispls", int, NTask+1);
    int i;

    MPI_Gather(&nrecv, 1, MPI_INT, sendcount, 1, MPI_INT, root, comm);

    sdispls[0] = 0;
    for(i = 1; i <= NTask; i ++) {
        sdispls[i] = sdispls[i - 1] + sendcount[i - 1];
    }

    if(ThisTask == root) {
        if(totalnsend)
            *totalnsend = sdispls[NTask];
    } else {
        if(totalnsend)
            *totalnsend = 0;
    }
    MPI_Scatterv(sendbuffer, sendcount, sdispls, dtype, recvbuffer, nrecv, dtype, root, comm);

    ta_free(sdispls);
    ta_free(sendcount);
    MPI_Type_free(&dtype);
}

template <typename T>
void _find_Pmax_Pmin_C(void * mybase, size_t mynmemb,
        size_t myoutnmemb,
        T& Pmax, T& Pmin,
        ptrdiff_t * C,
        struct crstruct * d,
        struct crmpistruct * o) {

    T myPmax{0};
    T myPmin{0};

    size_t * eachnmemb = ta_malloc("eachnmemb", size_t, o->NTask);
    size_t * eachoutnmemb = ta_malloc("eachoutnmemb", size_t, o->NTask);
    T * eachPmax = (T *) mymalloc("eachPmax", o->NTask * sizeof(T));
    T * eachPmin = (T *) mymalloc("eachPmin", o->NTask * sizeof(T));

    if(mynmemb > 0) {
        d->radix((char*) mybase + (mynmemb - 1) * d->size, &myPmax, d->arg);
        d->radix(mybase, &myPmin, d->arg);
    }

    MPI_Allgather(&mynmemb, 1, MPI_TYPE_PTRDIFF,
            eachnmemb, 1, MPI_TYPE_PTRDIFF, o->comm);
    MPI_Allgather(&myoutnmemb, 1, MPI_TYPE_PTRDIFF,
            eachoutnmemb, 1, MPI_TYPE_PTRDIFF, o->comm);
    MPI_Allgather(&myPmax, 1, o->MPI_TYPE_RADIX,
            eachPmax, 1, o->MPI_TYPE_RADIX, o->comm);
    MPI_Allgather(&myPmin, 1, o->MPI_TYPE_RADIX,
            eachPmin, 1, o->MPI_TYPE_RADIX, o->comm);


    C[0] = 0;
    for(int i = 0; i < o->NTask; i ++) {
        C[i + 1] = C[i] + eachoutnmemb[i];
        if(eachnmemb[i] == 0) continue;

        if(_compar_radix<T>(eachPmax[i], Pmax) > 0) {
            memcpy(&Pmax, &eachPmax[i], sizeof(T));
        }
        if(_compar_radix<T>(eachPmin[i], Pmin) < 0) {
            memcpy(&Pmin, &eachPmin[i], sizeof(T));
        }
    }

    myfree(eachPmin);
    myfree(eachPmax);
    myfree(eachoutnmemb);
    myfree(eachnmemb);
}

static int
_solve_for_layout_mpi (
        int NTask,
        ptrdiff_t * C,
        ptrdiff_t * myT_CLT,
        ptrdiff_t * myT_CLE,
        ptrdiff_t * myT_C,
        MPI_Comm comm) {
    int i, j;
    int ThisTask;
    MPI_Comm_rank(comm, &ThisTask);

    /* first assume we just send according to myT_CLT */
    for(i = 0; i < NTask; i ++) {
        myT_C[i] = myT_CLT[i];
    }

    /* Solve for each receiving task i
     *
     * this solves for GL_C[..., i + 1], which depends on GL_C[..., i]
     *
     * and we have GL_C[..., 0] == 0 by definition.
     *
     * this cannot be done in parallel wrt i because of the dependency.
     *
     *  a solution is guaranteed because GL_CLE and GL_CLT
     *  brackes the total counts C (we've found it with the
     *  iterative counting.
     *
     * */

    ptrdiff_t sure = 0;

    /* how many will I surely receive? */
    for(j = 0; j < NTask; j ++) {
        ptrdiff_t recvcount = myT_C[j];
        sure += recvcount;
    }
    /* let's see if we have enough */
    ptrdiff_t deficit = C[ThisTask + 1] - sure;

    for(j = 0; j < NTask; j ++) {
        /* deficit solved */
        if(deficit == 0) break;
        if(deficit < 0) {
            endrun(10, "More items than there should be at j=%d: deficit=%ld\n (C: %ld sure %ld)", j, deficit, C[ThisTask+1], sure);
        }
        /* how much task j can supply ? */
        ptrdiff_t supply = myT_CLE[j] - myT_C[j];
        if(supply < 0) {
            endrun(10, "Less items than there should be at j=%d: supply=%ld (myTCLE %ld myTC %ld)\n", j, supply, myT_CLE[j], myT_C[j]);
        }
        if(supply <= deficit) {
            myT_C[j] += supply;
            deficit -= supply;
        } else {
            myT_C[j] += deficit;
            deficit = 0;
        }
    }

    return 0;
}

template <typename T>
int
mpsort_mpi_histogram_sort(struct crstruct d, struct crmpistruct o)
{
    ptrdiff_t * myC = (ptrdiff_t *) mymalloc("myhistC", (o.NTask + 1) * sizeof(ptrdiff_t));

    /* Desired counts*/
    ptrdiff_t * C = (ptrdiff_t *) mymalloc("histC", (o.NTask + 1) * sizeof(ptrdiff_t));
    /* counts of less than P */
    ptrdiff_t * myCLT = (ptrdiff_t *) mymalloc("myhistC", (o.NTask + 1) * sizeof(ptrdiff_t));
    ptrdiff_t * CLT = (ptrdiff_t *) mymalloc("histCLT", (o.NTask + 1) * sizeof(ptrdiff_t));
    /* counts of less than or equal to P */
    ptrdiff_t * myCLE = (ptrdiff_t *) mymalloc("myhistCLE", (o.NTask + 1) * sizeof(ptrdiff_t));
    ptrdiff_t * CLE = (ptrdiff_t *) mymalloc("CLE", (o.NTask + 1) * sizeof(ptrdiff_t));

    int iter = 0;
    int done = 0;
    char * buffer;
    int i;

    /* and sort the local array */
    radix_sort<T>(o.mybase, o.mynmemb, d.size, d.radix, d.arg);

    MPI_Barrier(o.comm);

    T * P = (T *) mymalloc("PP", (o.NTask - 1) * sizeof(T));
    memset(P, 0, sizeof(T) * (o.NTask -1));

    T Pmax{0};
    T Pmin;
    memset(&Pmin, -1, sizeof(T));
    _find_Pmax_Pmin_C<T>(o.mybase, o.mynmemb, o.myoutnmemb, Pmax, Pmin, C, &d, &o);

    piter<T> pi(Pmin, Pmax, o.NTask - 1, &d);

    while(!done) {
        iter ++;
        pi.bisect(P);

        _histogram<T>(P, o.NTask - 1, o.mybase, o.mynmemb, myCLT, myCLE, &d);

        MPI_Allreduce(myCLT, CLT, o.NTask + 1,
                MPI_TYPE_PTRDIFF, MPI_SUM, o.comm);
        MPI_Allreduce(myCLE, CLE, o.NTask + 1,
                MPI_TYPE_PTRDIFF, MPI_SUM, o.comm);

        pi.accept(P, C, CLT, CLE);
#if 0
        {
            int k;
            for(k = 0; k < o.NTask; k ++) {
                MPI_Barrier(o.comm);
                int i;
                if(o.ThisTask != k) continue;

                printf("P (%d): PMin %d PMax %d P ",
                        o.ThisTask,
                        *(int*) Pmin,
                        *(int*) Pmax
                        );
                for(i = 0; i < o.NTask - 1; i ++) {
                    printf(" %d ", ((int*) P) [i]);
                }
                printf("\n");

                printf("C (%d): ", o.ThisTask);
                for(i = 0; i < o.NTask + 1; i ++) {
                    printf("%ld ", C[i]);
                }
                printf("\n");
                printf("CLT (%d): ", o.ThisTask);
                for(i = 0; i < o.NTask + 1; i ++) {
                    printf("%ld ", CLT[i]);
                }
                printf("\n");
                printf("CLE (%d): ", o.ThisTask);
                for(i = 0; i < o.NTask + 1; i ++) {
                    printf("%ld ", CLE[i]);
                }
                printf("\n");

            }
        }
#endif
        done = pi.all_done();
    }

    _histogram<T>(P, o.NTask - 1, o.mybase, o.mynmemb, myCLT, myCLE, &d);

    ta_free(P);

    ptrdiff_t * myT_C = (ptrdiff_t *) mymalloc("myhistT_C", (o.NTask) * sizeof(ptrdiff_t));
    ptrdiff_t * myT_CLT = (ptrdiff_t *) mymalloc("myhistCLT", (o.NTask) * sizeof(ptrdiff_t));
    ptrdiff_t * myT_CLE = (ptrdiff_t *) mymalloc("myhistCLE", (o.NTask) * sizeof(ptrdiff_t));

    /* transpose the matrix, could have been done with a new datatype */
    /*
    MPI_Alltoall(myCLT, 1, MPI_TYPE_PTRDIFF,
            myT_CLT, 1, MPI_TYPE_PTRDIFF, o.comm);
    */
    MPI_Alltoall(myCLT + 1, 1, MPI_TYPE_PTRDIFF,
            myT_CLT, 1, MPI_TYPE_PTRDIFF, o.comm);

    /*MPI_Alltoall(myCLE, 1, MPI_TYPE_PTRDIFF,
            myT_CLE, 1, MPI_TYPE_PTRDIFF, o.comm); */
    MPI_Alltoall(myCLE + 1, 1, MPI_TYPE_PTRDIFF,
            myT_CLE, 1, MPI_TYPE_PTRDIFF, o.comm);

    _solve_for_layout_mpi(o.NTask, C, myT_CLT, myT_CLE, myT_C, o.comm);

    myfree(myT_CLE);
    myfree(myT_CLT);

    myC[0] = 0;
    MPI_Alltoall(myT_C, 1, MPI_TYPE_PTRDIFF,
            myC + 1, 1, MPI_TYPE_PTRDIFF, o.comm);

    myfree(myT_C);

#if 0
    for(i = 0;i < o.NTask; i ++) {
        int j;
        MPI_Barrier(o.comm);
        if(o.ThisTask != i) continue;
        for(j = 0; j < o.NTask + 1; j ++) {
            printf("%d %d %d, ",
                    myCLT[j],
                    myC[j],
                    myCLE[j]);
        }
        printf("\n");

    }
#endif


    /* Desired counts*/
    myfree(CLE);
    myfree(myCLE);
    myfree(CLT);
    myfree(myCLT);
    myfree(C);

    int * SendCount = ta_malloc("SendCount", int, o.NTask);
    int * SendDispl = ta_malloc("SendDispl", int, o.NTask);
    int * RecvCount = ta_malloc("RecvCount", int, o.NTask);
    int * RecvDispl = ta_malloc("RecvDispl", int, o.NTask);

    for(i = 0; i < o.NTask; i ++) {
        SendCount[i] = myC[i + 1] - myC[i];
    }

    MPI_Alltoall(SendCount, 1, MPI_INT,
            RecvCount, 1, MPI_INT, o.comm);

    SendDispl[0] = 0;
    RecvDispl[0] = 0;
    size_t totrecv = RecvCount[0];
    for(i = 1; i < o.NTask; i ++) {
        SendDispl[i] = SendDispl[i - 1] + SendCount[i - 1];
        RecvDispl[i] = RecvDispl[i - 1] + RecvCount[i - 1];
        if(SendDispl[i] != myC[i]) {
            endrun(7, "SendDispl error\n");
        }
        totrecv += RecvCount[i];
    }
    if(totrecv != o.myoutnmemb) {
        endrun(8, "totrecv = %td, mismatch with %td\n", totrecv, o.myoutnmemb);
    }
#if 0
    {
        int k;
        for(k = 0; k < o.NTask; k ++) {
            MPI_Barrier(o.comm);

            if(o.ThisTask != k) continue;

            printf("P (%d): ", o.ThisTask);
            for(i = 0; i < o.NTask - 1; i ++) {
                printf("%d ", ((int*) P) [i]);
            }
            printf("\n");

            printf("C (%d): ", o.ThisTask);
            for(i = 0; i < o.NTask + 1; i ++) {
                printf("%d ", C[i]);
            }
            printf("\n");
            printf("CLT (%d): ", o.ThisTask);
            for(i = 0; i < o.NTask + 1; i ++) {
                printf("%d ", CLT[i]);
            }
            printf("\n");
            printf("CLE (%d): ", o.ThisTask);
            for(i = 0; i < o.NTask + 1; i ++) {
                printf("%d ", CLE[i]);
            }
            printf("\n");

            printf("MyC (%d): ", o.ThisTask);
            for(i = 0; i < o.NTask + 1; i ++) {
                printf("%d ", myC[i]);
            }
            printf("\n");
            printf("MyCLT (%d): ", o.ThisTask);
            for(i = 0; i < o.NTask + 1; i ++) {
                printf("%d ", myCLT[i]);
            }
            printf("\n");

            printf("MyCLE (%d): ", o.ThisTask);
            for(i = 0; i < o.NTask + 1; i ++) {
                printf("%d ", myCLE[i]);
            }
            printf("\n");

            printf("Send Count(%d): ", o.ThisTask);
            for(i = 0; i < o.NTask; i ++) {
                printf("%d ", SendCount[i]);
            }
            printf("\n");
            printf("My data(%d): ", o.ThisTask);
            for(i = 0; i < mynmemb; i ++) {
                printf("%d ", ((int*) mybase)[i]);
            }
            printf("\n");
        }
    }
#endif
    if(o.myoutbase == o.mybase)
        buffer = (char *) mymalloc("mpsortbuffer", d.size * o.myoutnmemb);
    else
        buffer = (char *) o.myoutbase;

    MPI_Alltoallv_smart(
            o.mybase, SendCount, SendDispl, o.MPI_TYPE_DATA,
            buffer, RecvCount, RecvDispl, o.MPI_TYPE_DATA,
            o.comm);

    if(o.myoutbase == o.mybase) {
        memcpy(o.myoutbase, buffer, o.myoutnmemb * d.size);
        myfree(buffer);
    }

    myfree(RecvDispl);
    myfree(RecvCount);
    myfree(SendDispl);
    myfree(SendCount);

    myfree(myC);
    MPI_Barrier(o.comm);

    radix_sort<T>(o.myoutbase, o.myoutnmemb, d.size, d.radix, d.arg);

    MPI_Barrier(o.comm);

    return 0;
}

template <typename T, size_t elsize>
void
mpsort_mpi_newarray_impl_type (void * mybase, size_t mynmemb,
        void * myoutbase, size_t myoutnmemb,
        void (*radix)(const void * ptr, void * radix, void * arg),
        void * arg,
        MPI_Comm comm,
        const int line,
        const char * file)
{
    if(MPI_TYPE_PTRDIFF == 0) {
        if(MPI_SUCCESS != MPI_Type_match_size(MPI_TYPECLASS_INTEGER, sizeof(ptrdiff_t), &MPI_TYPE_PTRDIFF))
            endrun(3, "Ptrdiff size %ld not recognised\n", sizeof(ptrdiff_t));
    }

    struct SegmentGroupDescr seggrp[1];

    uint64_t sum1 = checksum(mybase, elsize * mynmemb, comm);

    int NTask;
    int ThisTask;
    MPI_Comm_size(comm, &NTask);
    MPI_Comm_rank(comm, &ThisTask);

    //element size is large but not aligned to 8 bytes.This is known to frequently trigger MPI bugs.
    static_assert(elsize <= 8 || elsize % 8 == 0);
    /* MPSort: radix size is large (%ld) but not aligned to 8 bytes.
       This is known to frequently trigger MPI bugs.*/
    static_assert(sizeof(T) % 8 == 0 || sizeof(T) < 8);

    size_t * sizes = ta_malloc("sizes", size_t, NTask);
    sizes[ThisTask] = mynmemb;
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, sizes, 1, MPI_TYPE_PTRDIFF, comm);

    size_t avgsegsize = NTask; /* combine very small ranks to segments */
    if (avgsegsize * elsize > 4 * 1024 * 1024) {
        /* do not use more than 4MB in a segment */
        avgsegsize = 4 * 1024 * 1024 / elsize;
    }

    /* use as many groups as possible (some will be empty) but at most 1 segment per group */
    _create_segment_group(seggrp, sizes, avgsegsize, NTask, comm);

    myfree(sizes);
    /* group comm == seg comm */

    void * mysegmentbase = NULL;
    void * myoutsegmentbase = NULL;
    size_t mysegmentnmemb;
    size_t myoutsegmentnmemb;

    int groupsize;
    int grouprank;
    MPI_Comm_size(seggrp->Group, &groupsize);
    MPI_Comm_rank(seggrp->Group, &grouprank);

    MPI_Allreduce(&mynmemb, &mysegmentnmemb, 1, MPI_TYPE_PTRDIFF, MPI_SUM, seggrp->Group);
    MPI_Allreduce(&myoutnmemb, &myoutsegmentnmemb, 1, MPI_TYPE_PTRDIFF, MPI_SUM, seggrp->Group);

    if (groupsize > 1) {
        if(grouprank == seggrp->group_leader_rank) {
            mysegmentbase = mymalloc("segmentbase", mysegmentnmemb * elsize);
            myoutsegmentbase = mymalloc("outsegment", myoutsegmentnmemb * elsize);
        }
        MPIU_Gather(seggrp->Group, seggrp->group_leader_rank, mybase, mysegmentbase, mynmemb, elsize, NULL);
    } else {
        mysegmentbase = mybase;
        myoutsegmentbase = myoutbase;
    }

    /* only do sorting on the group leaders for each segment */
    if(seggrp->is_group_leader) {

        struct crstruct d;
        struct crmpistruct o;

        _setup_radix_sort<sizeof(T)>(&d, elsize, radix, arg);

        _setup_mpsort_mpi<T>(&o, &d, mysegmentbase, mysegmentnmemb, myoutsegmentbase, myoutsegmentnmemb, seggrp->Leaders);

        mpsort_mpi_histogram_sort<T>(d, o);

        _destroy_mpsort_mpi(&o);
    }

    if(groupsize > 1) {
        MPIU_Scatter(seggrp->Group, seggrp->group_leader_rank, myoutsegmentbase, myoutbase, myoutnmemb, elsize, NULL);
    }

    if(grouprank == seggrp->group_leader_rank) {
        if(myoutsegmentbase != myoutbase)
            myfree(myoutsegmentbase);
        if(mysegmentbase != mybase)
            myfree(mysegmentbase);
    }

    _destroy_segment_group(seggrp);

    uint64_t sum2 = checksum(myoutbase, elsize * myoutnmemb, comm);
    if (sum1 != sum2) {
        endrun(5, "Data changed after sorting; checksum mismatch.\n");
    }
}

/* Convert the size into a type*/
template <size_t rsize, size_t elsize>
void
mpsort_mpi_newarray_impl (void * mybase, size_t mynmemb,
        void * myoutbase, size_t myoutnmemb,
        void (*radix)(const void * ptr, void * radix, void * arg),
        void * arg,
        MPI_Comm comm,
        const int line,
        const char * file)
{
    if constexpr(rsize == 2) {
        mpsort_mpi_newarray_impl_type<uint16_t, elsize>(mybase, mynmemb, myoutbase, myoutnmemb, radix, arg, comm, line, file);
    } else if constexpr(rsize == 4) {
        mpsort_mpi_newarray_impl_type<uint32_t, elsize>(mybase, mynmemb, myoutbase, myoutnmemb, radix, arg, comm, line, file);
    } else if constexpr(rsize == 8) {
        mpsort_mpi_newarray_impl_type<uint64_t, elsize>(mybase, mynmemb, myoutbase, myoutnmemb, radix, arg, comm, line, file);
    } else {
        static_assert(sizeof(std::array<char, rsize>) == rsize);
        mpsort_mpi_newarray_impl_type<std::array<char, rsize>, elsize>(mybase, mynmemb, myoutbase, myoutnmemb, radix, arg, comm, line, file);
    }
}

#define mpsort_mpi(base, nmemb, elsize, radix, rsize, arg, comm) \
    mpsort_mpi_newarray_impl<rsize, elsize>(base, nmemb, base, nmemb, radix, arg, comm, __LINE__, __FILE__)

#define mpsort_mpi_newarray(base, nmemb, out, outnmemb, elsize, \
    radix, rsize, arg, comm) \
    mpsort_mpi_newarray_impl<rsize, elsize>(base, nmemb, out, outnmemb, radix, arg, comm, __LINE__, __FILE__)

#endif
