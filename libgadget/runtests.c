#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <ctype.h>

#include "utils.h"

#include "allvars.h"
#include "init.h"
#include "partmanager.h"
#include "forcetree.h"
#include "cooling.h"
#include "gravity.h"
#include "petaio.h"
#include "domain.h"
#include "timestep.h"
#include "fof.h"
#include "sfr_eff.h"
#include "density.h"
#include "hydra.h"
#include "run.h"

char * GDB_format_particle(int i);

SIMPLE_GETTER(GTGravAccel, GravAccel[0], float, 3, struct particle_data)
SIMPLE_GETTER(GTGravPM, GravPM[0], float, 3, struct particle_data)

void register_extra_blocks(struct IOTable * IOTable)
{
    int ptype;
    for(ptype = 0; ptype < 6; ptype++) {
        IO_REG_WRONLY(GravAccel,       "f4", 3, ptype, IOTable);
        IO_REG_WRONLY(GravPM,       "f4", 3, ptype, IOTable);
    }
}

double copy_and_mean_accn(double (* PairAccn)[3])
{
    int i;
    double meanacc = 0;
    #pragma omp parallel for reduction(+: meanacc)
    for(i = 0; i < PartManager->NumPart; i++)
    {
        int k;
        for(k=0; k<3; k++) {
            PairAccn[i][k] = P[i].GravPM[k] + P[i].GravAccel[k];
            meanacc += fabs(PairAccn[i][k]);
        }
    }
    int64_t tot_npart;
    MPI_Allreduce(&PartManager->NumPart, &tot_npart, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &meanacc, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    meanacc/= (tot_npart*3.);
    return meanacc;
}

void check_accns(double * meanerr_tot, double * maxerr_tot, double (*PairAccn)[3], double meanacc)
{
    double meanerr=0, maxerr=-1;
    int i;
    /* This checks that the short-range force accuracy is being correctly estimated.*/
    #pragma omp parallel for reduction(+: meanerr) reduction(max:maxerr)
    for(i = 0; i < PartManager->NumPart; i++)
    {
        int k;
        for(k=0; k<3; k++) {
            double err = 0;
            if(PairAccn[i][k] != 0)
                err = fabs((PairAccn[i][k] - (P[i].GravPM[k] + P[i].GravAccel[k]))/meanacc);
            meanerr += err;
            if(maxerr < err)
                maxerr = err;
        }
    }
    MPI_Allreduce(&meanerr, meanerr_tot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&maxerr, maxerr_tot, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    int64_t tot_npart;
    MPI_Allreduce(&PartManager->NumPart, &tot_npart, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);
    *meanerr_tot/= (tot_npart*3.);
}

/* Run various checks on the gravity code. Check that the short-range/long-range force split is working.*/
void runtests(const int RestartSnapNum, const inttime_t Ti_Current, const struct header_data * header)
{
    PetaPM pm = {0};
    gravpm_init_periodic(&pm, PartManager->BoxSize, All.Asmth, All.Nmesh, All.CP.GravInternal);
    DomainDecomp ddecomp[1] = {0};

    domain_decompose_full(ddecomp);	/* do initial domain decomposition (gives equal numbers of particles) */

    struct IOTable IOTable = {0};
    /* NO metals written*/
    register_io_blocks(&IOTable, 0, 0);
    register_extra_blocks(&IOTable);

    int NTask;
    MPI_Comm_size(MPI_COMM_WORLD, &NTask);
    ActiveParticles Act = {0};
    DriftKickTimes times = init_driftkicktime(Ti_Current);
    rebuild_activelist(&Act, &times, 0, header->TimeSnapshot);

    ForceTree Tree = {0};
    force_tree_rebuild(&Tree, ddecomp, 1, 1, All.OutputDir);
    gravpm_force(&pm, &Tree, &All.CP, header->TimeSnapshot, header->UnitLength_in_cm, All.OutputDir, header->TimeIC, All.FastParticleType, All.BlackHoleOn);
    force_tree_rebuild(&Tree, ddecomp, 1, 1, All.OutputDir);

    struct gravshort_tree_params origtreeacc = get_gravshort_treepar();
    struct gravshort_tree_params treeacc = origtreeacc;
    const double rho0 = All.CP.Omega0 * 3 * All.CP.Hubble * All.CP.Hubble / (8 * M_PI * All.CP.GravInternal);
    grav_short_pair(&Act, &pm, &Tree, treeacc.Rcut, rho0, 0, All.FastParticleType);

    double (* PairAccn)[3] = (double (*) [3]) mymalloc2("PairAccns", 3*sizeof(double) * PartManager->NumPart);

    double meanacc = copy_and_mean_accn(PairAccn);
    message(0, "GravShort Pairs %s\n", GDB_format_particle(0));
    char * fname = fastpm_strdup_printf("%s/PART-pairs-%03d", All.OutputDir, RestartSnapNum);

    petaio_save_snapshot(fname, &IOTable, 0, header->TimeSnapshot, &All.CP);

    treeacc.ErrTolForceAcc = 0;
    set_gravshort_treepar(treeacc);
    grav_short_tree(&Act, &pm, &Tree, rho0, 0, All.FastParticleType);

    /* This checks fully opened tree force against pair force*/
    double meanerr, maxerr;
    check_accns(&meanerr,&maxerr,PairAccn, meanacc);
    message(0, "Force error, open tree vs pairwise. max : %g mean: %g forcetol: %g\n", maxerr, meanerr, treeacc.ErrTolForceAcc);

    if(maxerr > 0.1)
        endrun(2, "Fully open tree force does not agree with pairwise calculation! maxerr %g > 0.1!\n", maxerr);

    message(0, "GravShort Tree %s\n", GDB_format_particle(0));
    fname = fastpm_strdup_printf("%s/PART-tree-open-%03d", All.OutputDir, RestartSnapNum);
    petaio_save_snapshot(fname, &IOTable, 0, header->TimeSnapshot, &All.CP);

    /* This checks tree force against tree force with zero error (which always opens).*/
    copy_and_mean_accn(PairAccn);

    treeacc = origtreeacc;
    set_gravshort_treepar(treeacc);
    /* Code automatically sets the UseTreeBH parameter.*/
    grav_short_tree(&Act, &pm, &Tree, rho0, 0, All.FastParticleType);
    grav_short_tree(&Act, &pm, &Tree, rho0, 0, All.FastParticleType);

    fname = fastpm_strdup_printf("%s/PART-tree-%03d", All.OutputDir, RestartSnapNum);
    petaio_save_snapshot(fname, &IOTable, 0, header->TimeSnapshot, &All.CP);

    check_accns(&meanerr,&maxerr,PairAccn, meanacc);
    message(0, "Force error, open tree vs tree.: %g mean: %g forcetol: %g\n", maxerr, meanerr, treeacc.ErrTolForceAcc);

    if(meanerr > treeacc.ErrTolForceAcc* 1.2)
        endrun(2, "Average force error is underestimated: %g > 1.2 * %g!\n", meanerr, treeacc.ErrTolForceAcc);

    copy_and_mean_accn(PairAccn);
    /* This checks the tree against a larger Rcut.*/
    treeacc.Rcut = 9.5;
    set_gravshort_treepar(treeacc);
    grav_short_tree(&Act, &pm, &Tree, rho0, 0, All.FastParticleType);
    grav_short_tree(&Act, &pm, &Tree, rho0, 0, All.FastParticleType);
    fname = fastpm_strdup_printf("%s/PART-tree-rcut-%03d", All.OutputDir, RestartSnapNum);
    petaio_save_snapshot(fname, &IOTable, 0, header->TimeSnapshot, &All.CP);

    check_accns(&meanerr,&maxerr,PairAccn, meanacc);
    message(0, "Force error, tree vs rcut.: %g mean: %g Rcut = %g\n", maxerr, meanerr, treeacc.Rcut);

    if(maxerr > 0.2)
        endrun(2, "Rcut decreased below desired value, error too large %g\n", maxerr);

    /* This checks the tree against a box with a smaller Nmesh.*/
    treeacc = origtreeacc;
    force_tree_free(&Tree);
    gravpm_init_periodic(&pm, PartManager->BoxSize, All.Asmth, All.Nmesh/2., All.CP.GravInternal);
    force_tree_rebuild(&Tree, ddecomp, 1, 1, All.OutputDir);
    gravpm_force(&pm, &Tree, &All.CP, header->TimeSnapshot, header->UnitLength_in_cm, All.OutputDir, header->TimeIC, All.FastParticleType, All.BlackHoleOn);
    force_tree_rebuild(&Tree, ddecomp, 1, 1, All.OutputDir);
    set_gravshort_treepar(treeacc);
    grav_short_tree(&Act, &pm, &Tree, rho0, 0, All.FastParticleType);
    grav_short_tree(&Act, &pm, &Tree, rho0, 0, All.FastParticleType);
    fname = fastpm_strdup_printf("%s/PART-tree-nmesh2-%03d", All.OutputDir, RestartSnapNum);
    petaio_save_snapshot(fname, &IOTable, 0, header->TimeSnapshot, &All.CP);

    check_accns(&meanerr, &maxerr, PairAccn, meanacc);
    message(0, "Force error, nmesh %d vs %d: %g mean: %g \n", All.Nmesh, All.Nmesh/2, maxerr, meanerr);

    if(maxerr > 0.5 || meanerr > 0.05)
        endrun(2, "Nmesh sensitivity worse, something may be wrong\n");

    myfree(PairAccn);
    force_tree_free(&Tree);
    destroy_io_blocks(&IOTable);
    petapm_destroy(&pm);
}
