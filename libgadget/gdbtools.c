#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "utils.h"
#include "partmanager.h"


/* these are for debuging in GDB */
int GDB_particle_by_id(MyIDType id, int from) {
    int i;
    for(i = from; i < PartManager->NumPart; i++) {
        if(PartManager->Base[i].ID == id) return i;
    }
    return -1;
}

int GDB_particle_by_type(int type, int from) {
    int i;
    for(i = from; i < PartManager->NumPart; i++) {
        if(PartManager->Base[i].Type == type) return i;
    }
    return -1;
}

int GDB_particle_by_generation(int gen, int from) {
    int i;
    for(i = from; i < PartManager->NumPart; i++) {
        if(PartManager->Base[i].Generation == gen) return i;
    }
    return -1;
}

char * GDB_particle_by_timebin(int bin) {
    int i;
    static char buf[1024];
    char tmp[20] = {'\0'};
    strcpy(buf, "");
    for(i = 0; i < PartManager->NumPart; i++) {
        if(PartManager->Base[i].TimeBinHydro == bin) {
            snprintf(tmp, 15, " %d", i);
            strncat(buf, tmp, 1024-strlen(tmp)-1);
        }
    }
    return buf;
}

int GDB_find_garbage(int from) {
    int i;
    for(i = from; i < PartManager->NumPart; i++) {
        if(PartManager->Base[i].IsGarbage) return i;
    }
    return -1;
}

char * GDB_format_particle(int i) {
    static char buf[1024];
    char * p = buf;
    int n = 1024;

#define add(fmt, ...) \
        snprintf(p, n - 1, fmt, __VA_ARGS__ ); \
        p = buf + strlen(buf); \
        n = 4096 - strlen(buf)

    add("PartManager->Base[%d]: ", i);
    add("ID : %lu ", PartManager->Base[i].ID);
    add("Generation: %d ", (int) PartManager->Base[i].Generation);
    add("Mass : %g ", PartManager->Base[i].Mass);
    add("Pos: %g %g %g ", PartManager->Base[i].Pos[0], PartManager->Base[i].Pos[1], PartManager->Base[i].Pos[2]);
    add("Vel: %g %g %g ", PartManager->Base[i].Vel[0], PartManager->Base[i].Vel[1], PartManager->Base[i].Vel[2]);
    add("FullTreeGravAccel: %g %g %g ", PartManager->Base[i].FullTreeGravAccel[0], PartManager->Base[i].FullTreeGravAccel[1], PartManager->Base[i].FullTreeGravAccel[2]);
    add("GravPM: %g %g %g ", PartManager->Base[i].GravPM[0], PartManager->Base[i].GravPM[1], PartManager->Base[i].GravPM[2]);
    return buf;
}

