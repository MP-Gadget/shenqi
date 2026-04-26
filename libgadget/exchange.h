#ifndef __EXCHANGE_H
#define __EXCHANGE_H

#include "partmanager.h"
#include "slotsmanager.h"

typedef struct PreExchangeList{
    /*List of particles to exchange*/
    int * ExchangeList;
    /*Total number of exchanged particles*/
    size_t nexchange;
    /*Number of garbage particles*/
    int64_t ngarbage;
} PreExchangeList;

template <typename ExchangePlan>
int domain_exchange(PreExchangeList * preexch, struct part_manager_type * pman, struct slots_manager_type * sman, int maxiter, MPI_Comm Comm);

void domain_test_id_uniqueness(struct part_manager_type * pman);

#endif
