#ifndef CHECKPOINT_H
#define CHECKPOINT_H
#include "cosmology.h"
#include <string>

void write_checkpoint(int snapnum, int WriteGroupID, int MetalReturnOn, double Time, const Cosmology * CP, const std::string OutputDir, const int OutputDebugFields);
void dump_snapshot(const std::string dump, const double Time, const Cosmology * CP, const std::string OutputDir);
int find_last_snapnum(const std::string OutputDir);

#endif
