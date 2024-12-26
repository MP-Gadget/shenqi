#ifndef LIGHTCONE_H
#define LIGHTCONE_H

/* Initialise the lightcone code module. */
void lightcone_init(Cosmology * CP, double timeBegin, const double UnitLength_in_cm, const char * OutputDir);
void lightcone_compute(const double a, const struct part_manager_type * const PartManager, Cosmology * CP, const inttime_t ti_curr, const inttime_t ti_next, const RandTable * const rnd);
#endif
