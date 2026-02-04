#include <mpi.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
#include <alloca.h>

#include "treewalk2.h"

/* find the closest index from radius and numNgb list, update left and right bound, return new hsml */
double
ngb_narrow_down(double *right, double *left, const double *radius, const double *numNgb, int maxcmpt, int desnumngb, int *closeidx, double BoxSize)
{
    int j;
    int close = 0;
    double ngbdist = fabs(numNgb[0] - desnumngb);
    for(j = 1; j < maxcmpt; j++){
        double newdist = fabs(numNgb[j] - desnumngb);
        if(newdist < ngbdist){
            ngbdist = newdist;
            close = j;
        }
    }
    if(closeidx)
        *closeidx = close;

    for(j = 0; j < maxcmpt; j++){
        if(numNgb[j] < desnumngb)
            *left = radius[j];
        if(numNgb[j] > desnumngb){
            *right = radius[j];
            break;
        }
    }

    double hsml = radius[close];

    if(*right > 0.99 * BoxSize){
        double dngbdv = 0;
        if(maxcmpt > 1 && (radius[maxcmpt-1]>radius[maxcmpt-2]))
            dngbdv = (numNgb[maxcmpt-1]-numNgb[maxcmpt-2])/(pow(radius[maxcmpt-1],3) - pow(radius[maxcmpt-2],3));
        /* Increase hsml by a maximum factor to avoid madness. We can be fairly aggressive about this factor.*/
        double newhsml = 4 * hsml;
        if(dngbdv > 0) {
            double dngb = (desnumngb - numNgb[maxcmpt-1]);
            double newvolume = pow(hsml,3) + dngb / dngbdv;
            if(pow(newvolume, 1./3) < newhsml)
                newhsml = pow(newvolume, 1./3);
        }
        hsml = newhsml;
    }
    if(hsml > *right)
        hsml = *right;

    if(*left == 0) {
        /* Extrapolate using volume, ie locally constant density*/
        double dngbdv = 0;
        if(radius[1] > radius[0])
            dngbdv = (numNgb[1] - numNgb[0]) / (pow(radius[1],3) - pow(radius[0],3));
        /* Derivative is not defined for minimum, so use 0.*/
        if(maxcmpt == 1 && radius[0] > 0)
            dngbdv = numNgb[0] / pow(radius[0],3);

        if(dngbdv > 0) {
            double dngb = desnumngb - numNgb[0];
            double newvolume = pow(hsml,3) + dngb / dngbdv;
            hsml = pow(newvolume, 1./3);
        }
    }
    if(hsml < *left)
        hsml = *left;

    return hsml;
}
