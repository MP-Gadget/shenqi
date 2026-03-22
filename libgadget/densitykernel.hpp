#ifndef _DENSITY_KERNEL_HPP
#define _DENSITY_KERNEL_HPP

#include <cmath>

#if !defined(TWODIMS) && !defined(ONEDIM)
#define  NUMDIMS 3		/*!< For 3D-normalized kernel */
constexpr static double normcoeff = (4.0/3 * M_PI);     /*!< Coefficient for kernel normalization. Note:  4.0/3 * PI = 4.188790204786 */
#else
#ifdef  TWODIMS
#define  NUMDIMS 2		/*!< For 2D-normalized kernel */
constexpr static double normcoeff = M_PI; 	/*!< Coefficient for kernel normalization. */
#else
#define  NUMDIMS 1             /*!< For 1D-normalized kernel */
constexpr static double normcoeff = 2.0;
#endif
#endif

#include "types.h"

/*enum DensityKernelType {
    DENSITY_KERNEL_CUBIC_SPLINE = 1,
    DENSITY_KERNEL_QUINTIC_SPLINE = 2,
    DENSITY_KERNEL_QUARTIC_SPLINE = 4,
};*/

/* support is H / h, see Price 2011: arxiv 1012.1885*/
template <typename DerivedKernel, double support, double sigma> class DensityKrnl
{
public:
    double H;/* convert from r to u with 1 / H*/

    DensityKrnl(const double i_H): H(i_H), Wknorm(sigma * pow(support / H, NUMDIMS))
    { }

    MYCUDAFN static double desnumngb(const double eta)
    {
        /* this returns the expected number of ngb in for given sph resolution
        * deseta */
        /* See Price: arxiv 1012.1885. eq 12 */
        return normcoeff * pow(support * eta, NUMDIMS);
    };

    MYCUDAFN double volume(void) const
    {
        return normcoeff * pow(H, NUMDIMS);
    };

    MYCUDAFN double dwk(const double u)
    {
        return Wknorm * support / H * static_cast<DerivedKernel * const>(this)->dwk_int(u * support);
    };

    MYCUDAFN double wk(const double u)
    {
        return Wknorm * static_cast<DerivedKernel * const>(this)->wk_int(u * support);
    };

    MYCUDAFN double density_kernel_dW(const double u) const
    {
        return - (NUMDIMS * wk(u) / H + u * dwk(u));
    };

protected:
    double Wknorm;
};

/**
 *
 * We use Price 1012.1885 kernels
 * sml in Gadget is the support big H in Price,
 *
 * u = r / H
 * q = r / h
 *
 * luckily, wk = 1 / H ** 3 W_volker(u)
 *             = 1 / h ** 3 W_price(q)
 * and     dwk = 1 / H ** 4 dw_volker/du
 *             = 1 / h ** 4 dw_price/dq
 *
 * wk_xx is Price eq 6 , 7, 8, without sigma
 *
 * the function density_kernel_wk and _dwk takes u to maintain compatibility
 * with volker's gadget.
 */
constexpr double cbsigma[3] = {2 / 3., 10 / (7 * M_PI), 1 / M_PI};

class CubicDensityKernel: public DensityKrnl<CubicDensityKernel, 2., cbsigma[NUMDIMS-1]> {
public:
    CubicDensityKernel(const double H): DensityKrnl(H) { }

    MYCUDAFN double wk_int(double q) const {
        if(q < 1.0) {
            return 0.25 * pow(2 - q, 3) - pow(1 - q, 3);
        }
        if(q < 2.0) {
            return 0.25 * pow(2 - q, 3);
        }
        return 0.0;
    };

    MYCUDAFN double dwk_int(double q) const {
        if(q < 1.0) {
            return - 0.25 * 3 * pow(2 - q, 2) + 3 * pow(1 - q, 2);
        }
        if(q < 2.0) {
            return -0.25 * 3 * pow(2 - q, 2);
        }
        return 0.0;
    };
};

constexpr double quarsigma[3] = {1 / 24., 96 / (1199 * M_PI), 1 / (20 * M_PI)};

class QuarticDensityKernel: public DensityKrnl <QuarticDensityKernel, 2.5, quarsigma[NUMDIMS-1]> {
public:
    QuarticDensityKernel(const double H): DensityKrnl(H) { }

    MYCUDAFN double wk_int(const double q) const {
        if(q < 0.5) {
            return pow(2.5 - q, 4) - 5 * pow(1.5 - q, 4) + 10 * pow(0.5 - q, 4);
        }
        if(q < 1.5) {
            return pow(2.5 - q, 4) - 5 * pow(1.5 - q, 4);
        }
        if(q < 2.5) {
            return pow(2.5 - q, 4);
        }
        return 0.0;
    }

    MYCUDAFN double dwk_int(const double q) const {
        if(q < 0.5) {
            return -4 * pow(2.5 - q, 3) + 20 * pow(1.5 - q, 3) - 40 * pow(0.5 - q, 3);
        }
        if(q < 1.5) {
            return -4 * pow(2.5 - q, 3) + 20 * pow(1.5 - q, 3);
        }
        if(q < 2.5) {
            return -4 * pow(2.5 - q, 3);
        }
        return 0.0;
    }
};

constexpr double quinsigma[3] = {1 / 120., 7 / (478 * M_PI), 1 / (120 * M_PI)};

class QuinticDensityKernel: public DensityKrnl<QuinticDensityKernel, 3., quinsigma[NUMDIMS-1]> {
public:
    QuinticDensityKernel(const double H): DensityKrnl(H) { }

    MYCUDAFN double wk_int(const double q) const {
        if(q < 1.0) {
            return pow(3 - q, 5) - 6 * pow(2 - q, 5) + 15 * pow(1 - q, 5);
        }
        if(q < 2.0) {
            return pow(3 - q, 5)- 6 * pow(2 - q, 5);
        }
        if(q < 3.0) {
            return pow(3 - q, 5);
        }
        return 0.0;
    }

    MYCUDAFN double dwk_int(double q) {
        if(q < 1.0) {
            return -5 * pow(3 - q, 4) + 30 * pow(2 - q, 4)
                - 75 * pow (1 - q, 4);
        }
        if(q < 2.0) {
            return -5 * pow(3 - q, 4) + 30 * pow(2 - q, 4);
        }
        if(q < 3.0) {
            return -5 * pow(3 - q, 4);
        }
        return 0.0;
    }
};

static inline void cross_product(const double v1[3], const double v2[3], double out[3])
{
    out[0] = v1[1] * v2[2] - v1[2] * v2[1];
    out[1] = v1[2] * v2[0] - v1[0] * v2[2];
    out[2] = v1[0] * v2[1] - v1[1] * v2[0];
}

/*auto
density_kernel_init(double H, enum DensityKernelType type)
{
    if(type == DENSITY_KERNEL_CUBIC_SPLINE) {
        return CubicDensityKernel(H);
    } else if(type == DENSITY_KERNEL_QUINTIC_SPLINE) {
        return QuarticDensityKernel(H);
    }
    return QuinticDensityKernel(H);
}*/

#endif
