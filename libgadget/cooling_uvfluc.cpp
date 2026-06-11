/**************
 * This file (C) Yu Feng 2017 implements support for a fluctuating UVB, using an external table.
 * This table is generated following the model explained in Battaglia & Trac 2010.
 ************** */

#include <mpi.h>
#include <string.h>
#include <math.h>
#include "cooling_rates.h"
#include "physconst.h"
#include "bigfile.h"
#include "bigfile-mpi.h"
#include "utils/mymalloc.h"
#include "utils/interp.hpp"
#include "utils/endrun.h"
#include "utils/paramset.h"

static struct {
    int enabled;
    InterpNLinear<3> interp;
    double * Table;
    ptrdiff_t Nside;
} UVF;

static struct UVFparams{
    /*settings for excursion set*/
    int ExcursionSetReionOn;
    double AlphaUV;
    double ExcursionSetZStop;
} uvf_params;

//set the parameters we need for the excursion set option
void set_uvf_params(ParameterSet * ps){
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    if(ThisTask==0)
    {
        uvf_params.ExcursionSetReionOn = param_get_int(ps,"ExcursionSetReionOn");
        uvf_params.ExcursionSetZStop = param_get_double(ps,"ExcursionSetZStop");
        uvf_params.AlphaUV = param_get_double(ps,"AlphaUV");
    }

    MPI_Bcast(&uvf_params, sizeof(struct UVFparams), MPI_BYTE, 0, MPI_COMM_WORLD);
    return;
}

/* The UV fluctation file is a bigfile with these tables:
 * ReionizedFraction: values of the reionized fraction as function of
 * redshift.
 * Redshift_Bins: uniform redshifts of the reionized fraction values
 *
 * XYZ_Bins: the uniform XYZ points where Z_reion is tabulated. (length of Nside)
 *
 * Zreion_Table: a Nside (X) x Nside (Y)x Nside (z) C ordering double array,
 * the reionization redshift as function of space, on a grid give by
 * XYZ_Bins.
 *
 * Notice that this table is broadcast to all MPI ranks, thus it can't be
 * too big. (400x400x400 is around 400 MBytes)
 *
 * */
void
init_uvf_table(const std::string& UVFluctuationFile, const double BoxSize, const double UnitLength_in_cm)
{
    if(UVFluctuationFile.size() == 0) {
        UVF.enabled = 0;
        return;
    }

    /* Open and validate the UV fluctuation file*/
    BigFile bf;
    BigBlock bh;
    if(0 != big_file_mpi_open(&bf, UVFluctuationFile.c_str(), MPI_COMM_WORLD)) {
        endrun(0, "Failed to open snapshot at %s:%s\n", UVFluctuationFile.c_str(),
                    big_file_get_error_message());
    }

    if(0 != big_file_mpi_open_block(&bf, &bh, "Zreion_Table", MPI_COMM_WORLD)) {
        endrun(0, "Failed to create block at %s:%s\n", "Header",
                    big_file_get_error_message());
    }
    double TableBoxSize;
    double ReionRedshift;
    if ((0 != big_block_get_attr(&bh, "Nmesh", &UVF.Nside, "u8", 1)) ||
        (0 != big_block_get_attr(&bh, "BoxSize", &TableBoxSize, "f8", 1)) ||
        (0 != big_block_get_attr(&bh, "Redshift", &ReionRedshift, "f8", 1))) {
        endrun(0, "Failed to read %s/Zreion_Table attributes: %s\n",
                    UVFluctuationFile.c_str(), big_file_get_error_message());
    }

    double BoxMpc = BoxSize * UnitLength_in_cm / CM_PER_MPC;
    if(fabs(TableBoxSize - BoxMpc) > BoxMpc * 1e-5)
        endrun(0, "Wrong UV fluctuation file! %s is for box size %g Mpc/h, but current box is %g Mpc/h\n", UVFluctuationFile.c_str(), TableBoxSize, BoxMpc);

    if(UVF.Nside * UVF.Nside * UVF.Nside != bh.size)
        endrun(0, "Corrupt UV Fluctuation table: Nside = %ld, but table is %lu != %ld^3\n", UVF.Nside, bh.size, UVF.Nside);

    message(0, "Using NON-UNIFORM UV BG fluctuations from %s. Median reionization redshift is %g\n", UVFluctuationFile.c_str(), ReionRedshift);
    UVF.enabled = 1;

    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);

    if(ThisTask == 0) {
        if(dtype_itemsize(bh.dtype) != sizeof(double))
            endrun(1, "UVfluctuationFile %s should contain double-precision data, contains %s\n", UVFluctuationFile.c_str(), bh.dtype);

        UVF.Table = (double *) mymalloc("UVFluctuationTable", bh.size * sizeof(double) * bh.nmemb);
        size_t dims[2] = {bh.size, static_cast<size_t>(bh.nmemb)};

        BigArray array[1];
        BigBlockPtr ptr = {0};
        big_array_init(array, UVF.Table, bh.dtype, 2, dims, NULL);
        if(0 != big_block_seek(&bh, &ptr, 0))
            endrun(1, "Failed to seek block %s %s: %s\n", UVFluctuationFile.c_str(), "Zreion_Table", big_file_get_error_message());
        if(0 != big_block_read(&bh, &ptr, array))
            endrun(1, "Failed to read %s %s: %s", UVFluctuationFile.c_str(), "Zreion_Table", big_file_get_error_message());
    }
    if(ThisTask != 0)
        UVF.Table = (double *) mymalloc("UVFluctuationTable", bh.size * bh.nmemb * sizeof(double));

    MPI_Bcast(UVF.Table, bh.size * bh.nmemb, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    big_block_mpi_close(&bh, MPI_COMM_WORLD);
    big_file_mpi_close(&bf, MPI_COMM_WORLD);

    int64_t dims[] = {UVF.Nside, UVF.Nside, UVF.Nside};
    double uvfmin[] = {0, 0, 0};
    double uvfmax[] = {BoxSize, BoxSize, BoxSize};
    InterpNLinear<3> interp(dims, uvfmin, uvfmax);
    UVF.interp = interp;
    if(UVF.Table[0] < 0.01 || UVF.Table[0] > 100.0) {
        endrun(0, "UV Fluctuation out of range: %g\n", UVF.Table[0]);
    }
}

/*
 * returns the spatial dependent UVBG if UV fluctuation is enabled.
 * Otherwise returns the global UVBG passed in.
 *
 * */
static struct UVBG get_local_UVBG_from_global(double redshift, const struct UVBG * const GlobalUVBG, const double * const Pos, const double * const PosOffset)
{
    if(!UVF.enabled) {
        /* directly use the TREECOOL table if UVF is disabled */
        return *GlobalUVBG;
    }

    struct UVBG uvbg = {0};

    uvbg.self_shield_dens = GlobalUVBG->self_shield_dens;

    double corrpos[3];
    int i;
    for(i = 0; i < 3; i++)
        corrpos[i] = Pos[i] - PosOffset[i];
    double zreion = UVF.interp.eval_periodic(corrpos, UVF.Table);
    if(zreion < redshift) {
        uvbg.zreion = zreion;
        return uvbg;
    }
    memcpy(&uvbg, GlobalUVBG, sizeof(struct UVBG));
    uvbg.zreion = zreion;
    return uvbg;
}

static struct UVBG get_local_UVBG_from_J21(double redshift, double J21, double zreion) {
    struct UVBG uvbg = {0};

    // N.B. J21 must be in units of 1e-21 erg s-1 Hz-1 (proper cm)-2 sr-1
    uvbg.J_UV = J21;
    uvbg.zreion = zreion;

    //interpolators in cooling_rates.c should now be rate coeffs
    //it seems a bit wasteful to calculate this for every particle
    //but the global uv does an interpolation every time and this allows
    //for future inhomogeneous alpha
    //if this becomes a bottleneck we can set the coeffs globally
    struct J21_coeffs J21toUV = get_J21_coeffs(uvf_params.AlphaUV);

    uvbg.gJH0   = J21toUV.gJH0 * J21; // s-1
    uvbg.epsH0  = J21toUV.epsH0 * J21 * 1.60218e-12;  // erg s-1
    uvbg.gJHe0  = J21toUV.gJHe0 * J21; // s-1
    uvbg.epsHe0 = J21toUV.epsHe0 * J21 * 1.60218e-12;  // erg s-1

    /*Since the excursion set only finds HII (& HeII) bubbles, and HeII -> HeIII
     * heating is taken care of by the qso_lightup model, there is never a case where we need these rates */
    /* NOTE: this means that the excursion set will be switched off before helium reionisation
     * and global rates must be used, otherwise helium will not ionise or heat */
    /* the excursion set (so far) only includes stellar ionising radiation, so
     * this is equivalent to the assumption that stars do not doubly ionise helium
     * which will need to change if we decide to add QSO radiation to the excursion set (i.e Qin et al. 2017, DRAGONS X)*/
    uvbg.gJHep = 0.;
    uvbg.epsHep = 0.;

    uvbg.self_shield_dens = get_self_shield_dens(redshift, &uvbg);

    return uvbg;
}

//switch function that decides whether to use excursion set or global UV background
/*TODO: Better continuity, if the z_reion tables provided finish after ExcursionSetZStop, particles could rapidly recombine.
 * also if helium reion starts before the excursion set finishes, flash reionisations occur as we switch to global*/
struct UVBG get_local_UVBG(double redshift, const struct UVBG * const GlobalUVBG, const double * const Pos, const double * const PosOffset, double J21, double zreion)
{
    if(uvf_params.ExcursionSetReionOn && (redshift > uvf_params.ExcursionSetZStop))
    {
        return get_local_UVBG_from_J21(redshift,J21,zreion);
    }
    else
    {
        return get_local_UVBG_from_global(redshift,GlobalUVBG,Pos,PosOffset);
    }
}

/*Here comes the Metal Cooling code*/
struct {
    int CoolingNoMetal;
    double * Lmet_table; /* metal cooling @ one solar metalicity*/

    InterpNLinear<3> interp;
} MetalCool;

/* Read a big array from filename/dataset into an array, allocating memory in buffer.
 * which is returned. Nread argument is set equal to number of elements read.*/
static double *
read_big_array(BigFile * bf, const char * dataset, int * Nread)
{
    size_t N = 0, nmemb = 0;
    double * buffer=NULL;
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);

    if(ThisTask == 0) {
        BigBlockPtr ptr = {0};
        BigBlock bb[1];
        BigArray array[1];
        if(0 != big_file_open_block(bf, bb, dataset)) {
            endrun(1, "Cannot open block %s: %s\n", dataset, big_file_get_error_message());
        }
        N = bb->size;
        nmemb = bb->nmemb;

        if(dtype_itemsize(bb->dtype) != sizeof(double))
            endrun(1, "Block %s should contain double-precision data, contains %s\n", dataset, bb->dtype);

        buffer = (double *) mymalloc(dataset, N  * bb->nmemb * sizeof(double));

        size_t dims[2] = {N, nmemb};
        big_array_init(array, buffer, bb->dtype, 2, dims, NULL);
        if(0 != big_block_seek(bb, &ptr, 0))
            endrun(1, "Failed to seek block %s: %s\n", dataset, big_file_get_error_message());

        if(0 != big_block_read(bb, &ptr, array))
            endrun(1, "Failed to read %s: %s", dataset, big_file_get_error_message());
        /* steal the buffer */
        big_block_close(bb);
    }

    N *= nmemb;
    MPI_Bcast(&N, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    if(ThisTask != 0)
        buffer = (double *) mymalloc(dataset, N* sizeof(double));

    MPI_Bcast(buffer, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    *Nread = N;
    return buffer;
}

void
InitMetalCooling(const std::string& MetalCoolFile)
{
    /* now initialize the metal cooling table from cloudy; we got this file
     * from vogelsberger's Arepo simulations; it is supposed to be
     * cloudy + UVB - H and He; look so.
     * the table contains only 1 Z_sun values. Need to be scaled to the
     * metallicity.
     * */
    /* let's see if the Metal Cool File is magic NoMetal */
    if(MetalCoolFile.size() == 0) {
        MetalCool.CoolingNoMetal = 1;
        return;
    } else {
        MetalCool.CoolingNoMetal = 0;
    }

    BigFile bf[1];
    if(0 != big_file_mpi_open(bf, MetalCoolFile.c_str(), MPI_COMM_WORLD))
        endrun(1, "Cannot open %s: %s\n", MetalCoolFile.c_str(), big_file_get_error_message());

    int size;
    //This is never used if MetalCoolFile == ""
    double * tabbedmet = read_big_array(bf, "MetallicityInSolar_bins", &size);

    if(size != 1 || tabbedmet[0] != 0.0) {
        endrun(123, "MetalCool file %s is wrongly tabulated\n", MetalCoolFile.c_str());
    }
    myfree(tabbedmet);

    int NRedshift_bins, NHydrogenNumberDensity_bins, NTemperature_bins;
    double * Redshift_bins = read_big_array(bf, "Redshift_bins", &NRedshift_bins);
    double * HydrogenNumberDensity_bins = read_big_array(bf, "HydrogenNumberDensity_bins", &NHydrogenNumberDensity_bins);
    double * Temperature_bins = read_big_array(bf, "Temperature_bins", &NTemperature_bins);
    MetalCool.Lmet_table = read_big_array(bf, "NetCoolingRate", &size);

    big_file_mpi_close(bf, MPI_COMM_WORLD);

    int64_t dims[] = {NRedshift_bins, NHydrogenNumberDensity_bins, NTemperature_bins};
    double metalmin[] = {Redshift_bins[0], HydrogenNumberDensity_bins[0],  Temperature_bins[0]};
    double metalmax[] = {Redshift_bins[NRedshift_bins - 1], HydrogenNumberDensity_bins[NHydrogenNumberDensity_bins - 1], Temperature_bins[NTemperature_bins - 1]};

    myfree(Temperature_bins);
    myfree(HydrogenNumberDensity_bins);
    myfree(Redshift_bins);

    InterpNLinear<3> interp(dims, metalmin, metalmax);
    MetalCool.interp = interp;
}

double
TableMetalCoolingRate(double redshift, double temp, double nHcgs)
{
    if(MetalCool.CoolingNoMetal)
        return 0;

    double lognH = log10(nHcgs);
    double logT = log10(temp);

    double x[] = {redshift, lognH, logT};
    double rate = MetalCool.interp.eval(x, MetalCool.Lmet_table);
    /* XXX: in case of very hot / very dense we just use whatever the table says at
     * the limit. should be OK. */
    return rate;
}
