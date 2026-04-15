#ifndef LONGRANGE_H
#define LONGRANGE_H

#include "forcetree.h"
#include "petapm.h"
#include "powerspectrum.h"

enum ShortRangeForceWindowType {
    SHORTRANGE_FORCE_WINDOW_TYPE_EXACT = 0,
    SHORTRANGE_FORCE_WINDOW_TYPE_ERFC = 1,
};

struct gravshort_tree_params
{
    double ErrTolForceAcc;      /*!< parameter for relative opening criterion in tree walk.
                                 * Desired accuracy of the tree force in units of the old acceleration.*/
    double BHOpeningAngle;      /*!< Barnes-Hut parameter for opening criterion in tree walk */
    double MaxBHOpeningAngle;    /* When using the relative acceleration criterion, we also enforce a maximum BH opening criterion to avoid pathological cases.*/
    int TreeUseBH;              /*!< If true, use the BH opening angle. Otherwise use acceleration. If > 0, use the Barnes-Hut opening angle.*
                                 *  If < 0, use the acceleration condition. */
    /*! RCUT gives the maximum distance (in units of the scale used for the force split) out to which short-range
     * forces are evaluated in the short-range tree walk.*/
    double Rcut;
    /* Softening as a fraction of DM mean separation. */
    double FractionalGravitySoftening;
    /* Maximum size of the export buffer. */
    size_t MaxExportBufferBytes;
    /* Type of the short range window function: exact from table or erfc */
    enum ShortRangeForceWindowType ShortRangeForceWindowType;
};

class GravShortTable
{
    private:
    #define NGRAVTAB 512
    /*! variables for short-range lookup table */
    float shortrange_table[NGRAVTAB];
    float shortrange_table_potential[NGRAVTAB];
    // shortrange_table_tidal[NTAB];
    double dx;

    public:
    /* Initialise the tables from pre-computed data */
    GravShortTable(const enum ShortRangeForceWindowType ShortRangeForceWindowType, const double Asmth);

    /* Compute force factor (*fac) and multiply potential (*pot) by the shortrange force window function.
     * If the distance is outside the range of the table, 1 is returned and the caller should assume zero acceleration.
     * If the distance is inside the range of the table, 0 is returned and the force factor and potential values
     * should be applied to the particle query. */
    MYCUDAFN int apply_short_range_window(const double r, double * fac, double * pot, const double cellsize) const
    {
        const double i = (r / cellsize / dx);
        size_t tabindex = floor(i);
        if(tabindex >= NGRAVTAB - 1)
            return 1;
        /* use a linear interpolation; */
        *fac *= (tabindex + 1 - i) * shortrange_table[tabindex] + (i - tabindex) * shortrange_table[tabindex + 1];
        *pot *= (tabindex + 1 - i) * shortrange_table_potential[tabindex] + (i - tabindex) * shortrange_table_potential[tabindex];
        return 0;
    }
};

/*! Sets the (comoving) softening length, converting from units of the mean DM separation to comoving internal units. */
void gravshort_set_softenings(double MeanDMSeparation);

/* gravitational softening length
 * (given in terms of an `equivalent' Plummer softening length) */
double FORCE_SOFTENING(void);

/*Defined in gravpm.c*/
void gravpm_init_periodic(PetaPM * pm, double BoxSize, double Asmth, int Nmesh, double G);

/* Apply the short-range window function, which includes the smoothing kernel.*/
int grav_apply_short_range_window(double r, double * fac, double * pot, const double cellsize);

/* Set up the module*/
void set_gravshort_tree_params(ParameterSet * ps);
/* Helpers for the tests*/
void set_gravshort_treepar(struct gravshort_tree_params tree_params);
struct gravshort_tree_params get_gravshort_treepar(void);

/* Computes the gravitational force on the PM grid
 * and saves the total matter power spectrum.
 * Parameters: Cosmology, Time, UnitLength_in_cm and PowerOutputDir are used by the power spectrum output code.
 * TimeIC is used by the massive neutrino code. A tree is built and freed during this function*/
void gravpm_force(PetaPM * pm, DomainDecomp * ddecomp, Cosmology * CP, double Time, double UnitLength_in_cm, const char * PowerOutputDir, double TimeIC);

/* Compute the short range gravitational tree force. */
void grav_short_tree(const ActiveParticles * act, PetaPM * pm, ForceTree * tree, MyFloat (* AccelStore)[3], double rho0, inttime_t Ti_Current, bool UseGPU=false);

/*Read the power spectrum, without changing the input value.*/
void measure_power_spectrum(PetaPM * pm, int64_t k2, int kpos[3], pfft_complex *value);

/* Compute the power spectrum of the Fourier transformed grid in value.*/
void powerspectrum_add_mode(Power * PowerSpectrum, const int64_t k2, const int kpos[3], pfft_complex * const value, const double invwindow, double Nmesh);

#endif
