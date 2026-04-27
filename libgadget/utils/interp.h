#ifndef INTERP_H
#define INTERP_H

#include <stdint.h>

template <int Ndim>
class InterpNLinear {
public:
    int dims[Ndim];
    ptrdiff_t strides[Ndim];
    double Min[Ndim];
    double Step[Ndim];
    double Max[Ndim];

    InterpNLinear() { }

    InterpNLinear(int64_t * idims, double * iMin, double * iMax) {
        /* fillin strides */
        ptrdiff_t N = 1;
        for(int d = Ndim - 1 ; d >= 0; d --) {
            dims[d] = idims[d];
            /* column first, C ordering */
            strides[d] = N;
            N *= dims[d];
        }
        for(int d = 0; d < Ndim; d++) {
            Min[d] = iMin[d];
            Max[d] = iMax[d];
            Step[d] = (iMax[d] - iMin[d]) / (dims[d] - 1);
        }
    }

    /* interpolate the table at point x */
    double eval(double * x, double * ydata) {
        int xi[Ndim];
        double f[Ndim];

        for(int d = 0; d < Ndim; d++) {
            double xd = (x[d] - Min[d]) / Step[d];
            if (x[d] < Min[d]) {
                xi[d] = 0;
                f[d] = 0;
            } else
            if (x[d] > Max[d]) {
                xi[d] = dims[d] - 1;
                f[d] = 0;
            } else {
                xi[d] = floor(xd);
                f[d] = xd - xi[d];
            }
        }

        double ret = 0;
        /* the origin, "this point" */
        ptrdiff_t l0 = linearindex(xi);

        /* for each point covered by the filter */
        for(int i = 0; i < (1 << Ndim); i ++) {
            double filter = 1.0;
            ptrdiff_t l = l0;
            int skip = 0;
            for(int d = 0; d < Ndim; d++ ) {
                int foffset = (i & (1 << d))?1:0;
                if(f[d] == 0 && foffset == 1) {
                    /* on this dimension the second data point
                    * is not needed */
                    skip = 1;
                    break;
                }

                /*
                * are we on this point or next point?
                *
                * weight on next point is f[d]
                * weight on this point is 1 - f[d]
                * */
                filter *= foffset?f[d] : (1 - f[d]);
                l += foffset * strides[d];
            }
            if(!skip) {
                ret += ydata[l] * filter;
            }
        }
        return ret;
    }

    /* interpolation assuming periodic boundary */
    double eval_periodic(double * x, double * ydata) {
        int xi[Ndim];
        int xi1[Ndim];
        double f[Ndim];

        for(int d = 0; d < Ndim; d++) {
            double xd = (x[d] - Min[d]) / Step[d];
            xi[d] = floor(xd);
            f[d] = xd - xi[d];
        }

        double ret = 0;
        /* the origin, "this point" */

        /* for each point covered by the filter */
        for(int i = 0; i < (1 << Ndim); i ++) {
            double filter = 1.0;
            for(int d = 0; d < Ndim; d++ ) {
                int foffset = (i & (1 << d))?1:0;
                xi1[d] = xi[d] + foffset;
                while(xi1[d] >= dims[d])
                    xi1[d] -= dims[d];
                while(xi1[d] < 0 )
                    xi1[d] += dims[d];
                /*
                * are we on this point or next point?
                *
                * weight on next point is f[d]
                * weight on this point is 1 - f[d]
                * */
                filter *= foffset?f[d] : (1 - f[d]);
            }
            ptrdiff_t l = linearindex(xi1);
            ret += ydata[l] * filter;
        }
        return ret;
    }
private:
    ptrdiff_t linearindex(int * xi) {
        ptrdiff_t rt = 0;
        for(int d = 0; d < Ndim; d++)
            rt += strides[d] * xi[d];
        return rt;
    }

};

#endif
