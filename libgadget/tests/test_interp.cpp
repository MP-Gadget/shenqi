#define BOOST_TEST_MODULE interp

#include "booststub.h"

#include <libgadget/utils/interp.hpp>

#define DMAX(x,y) ((x) > (y) ? (x) : (y))
#define DMIN(x,y) ((x) < (y) ? (x) : (y))

/*Modified modulus function which deals with negative numbers better*/
double modulus(double x, double mod) {
    if (x >= 0)
        return fmod(x,mod);
    else
        return fmod(x+mod,mod);
}

#define DSIZE 3

BOOST_AUTO_TEST_CASE(test_interp)
{
    int64_t dims[] = {DSIZE, DSIZE};
    double ydata[DSIZE*DSIZE], ydata_sum[DSIZE*DSIZE];
    /*Initialise the data*/
    {
        int i, j;
        for(i = 0; i < DSIZE; i++) {
            for(j = 0; j < DSIZE; j++) {
                ydata[i*DSIZE+j] = fabs((1. - j) * (1. - i));
                ydata_sum[i*DSIZE+j] = i+j;
            }
        }
    }

    double min[] = {0, 0};
    double max[] = {DSIZE-1, DSIZE-1};
    InterpNLinear<2> ip(dims, min, max);

    for(double i = -0.4; i <= DSIZE; i += 0.4) {
        for(double j = -0.4; j <= DSIZE; j += 0.4) {
            double x[2] = {i, j};
            double y = ip.eval(x, ydata);
            double yp = ip.eval_periodic (x, ydata);
            /*Note boundaries: without periodic use the maximum, with no remainers.*/
            double y_truth = fabs((1.-DMAX(DMIN(i,DSIZE-1),0))*(1.-DMAX(DMIN(j,DSIZE-1),0)));
            /* With a periodic boundary we normally use the modulus. However for this specific case
             * the boundaries happen to be identical so variation in the bin between them
             * is ignored by the interpolator and y == yp. This is not true if you increase the range on i.*/
/*             printf("(%g %g ) %3.2f/%3.2f/%3.2f \n", i,j, y, yp, yp_truth); */
            BOOST_TEST(y == y_truth, tt::tolerance(1e-5*y));
            BOOST_TEST(yp == y_truth, tt::tolerance(1e-5*yp));
        }
    }
    for(double i = -0.4; i <= 3.0; i += 0.4) {
        for(double j = -0.4; j <= 3.0; j += 0.4) {
            double x[2] = {i, j};
            double y = ip.eval(x, ydata_sum);
            double yp = ip.eval_periodic (x, ydata_sum);
            double y_truth = DMAX(DMIN(i,DSIZE-1),0)+DMAX(DMIN(j,DSIZE-1),0);
            double yp_truth = modulus(i,DSIZE)+ modulus(j, DSIZE);
/*             printf("(%g %g ) %3.2f/%3.2f/%3.2f \n", i,j, y, yp, yp_truth); */
            /*Linear interpolation is very inaccurate outside the boundaries in the periodic case!*/
            if(i >= 0 && j >= 0 && i <= DSIZE-1 && j <= DSIZE-1)
                BOOST_TEST(yp == yp_truth, tt::tolerance(1e-5*yp));
            BOOST_TEST(y == y_truth, tt::tolerance(1e-5*y));
        }
    }
}
