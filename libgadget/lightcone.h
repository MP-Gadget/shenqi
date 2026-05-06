#ifndef LIGHTCONE_H
#define LIGHTCONE_H
#include "cosmology.h"
#include "types.h"
#include "utils/system.h"

/* Initialise the lightcone code module. */
void lightcone_init(Cosmology * CP, double timeBegin, const double UnitLength_in_cm, const char * OutputDir, int ThisTask);
void lightcone_compute(const double a, const struct part_manager_type * const PartManager, const double ddrift, const RandTable * const rnd);
#endif
