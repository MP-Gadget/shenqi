#ifndef BHDYNFRIC_H
#define BHDYNFRIC_H

#include "utils/paramset.h"
#include "forcetree.h"
#include "density2.h"

/* Do the dynamic friction treewalk if BH_DynFrictionMethod > 0.
 * Builds a private tree with the types needed for dynamic friction (mostly stars and DM).*/
void blackhole_dynfric(int * ActiveBlackHoles, int64_t NumActiveBlackHoles, DomainDecomp * ddecomp, KickFactorData& kf, inttime_t Ti_Current, MPI_Comm comm);
/* Compute the DF acceleration for all active black holes*/
void blackhole_dfaccel(int * ActiveBlackHoles, size_t NumActiveBlackHoles, const double atime, const double GravInternal);
void set_blackhole_dynfric_params(ParameterSet * ps);
/* Get the particle types used in dynfric*/
int blackhole_dynfric_treemask(void);

/* Stand-alone function to find the black hole local potential minimum, when using the repositioning model. Uses its own treebuild.
 * The local potential minimum is also found by the dynamic friction treewalk.*/
void blackhole_findminpot(int * ActiveBlackHoles, const int64_t NumActiveBlackHoles, DomainDecomp * ddecomp);

/* Decide whether black hole repositioning is enabled. */
int BHGetRepositionEnabled(void);

#endif
