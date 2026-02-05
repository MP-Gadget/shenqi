#include <mpi.h>
#include <math.h>

#include "partmanager.h"
#include "forcetree.h"
#include "gravity.h"
#include "petaio.h"
#include "domain.h"
#include "run.h"
#include "treewalk.h"
#include "utils/endrun.h"
#include "utils/system.h"
#include "utils/mymalloc.h"
#include "utils/string.h"

char * GDB_format_particle(int i);

SIMPLE_GETTER(GTGravAccel, FullTreeGravAccel[0], float, 3, struct particle_data)
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
            PairAccn[i][k] = Part[i].GravPM[k] + Part[i].FullTreeGravAccel[k];
            meanacc += fabs(PairAccn[i][k]);
        }
    }
    int64_t tot_npart;
    MPI_Allreduce(&PartManager->NumPart, &tot_npart, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &meanacc, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    meanacc/= (tot_npart*3.);
    return meanacc;
}

void check_accns(double * meanerr_tot, double * maxerr_tot, double * meanangle_tot, double * maxangle_tot, double (*PairAccn)[3])
{
    double meanerr=0, maxerr=-1, meanangle = 0, maxangle = 0;
    int i;
    /* This checks that the short-range force accuracy is being correctly estimated.*/
    #pragma omp parallel for reduction(+: meanerr) reduction(max:maxerr)
    for(i = 0; i < PartManager->NumPart; i++)
    {
        if(Part[i].IsGarbage || Part[i].Swallowed)
            continue;
        int k;
        double pairmag = 0, checkmag = 0, dotprod = 0;
        for(k=0; k<3; k++) {
            pairmag += PairAccn[i][k]*PairAccn[i][k];
            checkmag += (Part[i].GravPM[k] + Part[i].FullTreeGravAccel[k])*(Part[i].GravPM[k] + Part[i].FullTreeGravAccel[k]);
            dotprod += PairAccn[i][k] * (Part[i].GravPM[k] + Part[i].FullTreeGravAccel[k]);
        }
        checkmag = sqrt(checkmag);
        pairmag = sqrt(pairmag);
        double err = fabs(checkmag/pairmag - 1);
        dotprod = dotprod / checkmag / pairmag;
        double angle = 0;
        if(dotprod <= 1 && dotprod >= -1) {
            angle = fabs(acos(dotprod));
            meanangle += angle;
        }
        if(maxangle < angle) {
            maxangle = angle;
            // message(0, "i %d type %d angle %g acc %g %g %g pair %g %g %g\n", i, Part[i].Type, angle, Part[i].GravPM[0] + Part[i].FullTreeGravAccel[0], Part[i].GravPM[1] + Part[i].FullTreeGravAccel[1], Part[i].GravPM[2] + Part[i].FullTreeGravAccel[2], PairAccn[i][0], PairAccn[i][1], PairAccn[i][2]);
        }
        meanerr += err;
        if(maxerr < err) {
            // message(0, "i %d type %d err %g acc %g %g %g pair %g %g %g\n", i, Part[i].Type, err, Part[i].GravPM[0] + Part[i].FullTreeGravAccel[0], Part[i].GravPM[1] + Part[i].FullTreeGravAccel[1], Part[i].GravPM[2] + Part[i].FullTreeGravAccel[2], PairAccn[i][0], PairAccn[i][1], PairAccn[i][2]);
            maxerr = err;
        }
    }
    MPI_Allreduce(&meanerr, meanerr_tot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&maxerr, maxerr_tot, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&meanangle, meanangle_tot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&maxangle, maxangle_tot, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    int64_t tot_npart;
    MPI_Allreduce(&PartManager->NumPart, &tot_npart, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);
    *meanerr_tot /= (tot_npart);
    *meanangle_tot /= (tot_npart);
}

void
run_gravity_test(int RestartSnapNum, Cosmology * CP, const double Asmth, const int Nmesh, const inttime_t Ti_Current, const char * OutputDir, const struct header_data * header)
{
    DomainDecomp ddecomp[1] = {0};
    domain_decompose_full(ddecomp, MPI_COMM_WORLD);

    struct IOTable IOTable = {0};
    /* NO metals written*/
    register_io_blocks(&IOTable, 0, 0);
    register_extra_blocks(&IOTable);

    double (* PairAccn)[3] = (double (*) [3]) mymalloc2("PairAccns", 3*sizeof(double) * PartManager->NumPart);

    PetaPM pm[1] = {0};
    gravpm_init_periodic(pm, PartManager->BoxSize, Asmth, Nmesh, CP->GravInternal);

    int NTask;
    MPI_Comm_size(MPI_COMM_WORLD, &NTask);

    DriftKickTimes times = init_driftkicktime(Ti_Current);
    /* All particles are active*/
    ActiveParticles Act = init_empty_active_particles(PartManager);
    build_active_particles(&Act, &times, 0, header->TimeSnapshot, PartManager);

    gravpm_force(pm, ddecomp, CP, header->TimeSnapshot, header->UnitLength_in_cm, OutputDir, header->TimeIC);

    ForceTree Tree = {0};
    force_tree_full(&Tree, ddecomp, 0, OutputDir);

    struct gravshort_tree_params origtreeacc = get_gravshort_treepar();
    /* Reset to normal tree */
    if(origtreeacc.TreeUseBH > 1)
        origtreeacc.TreeUseBH = 0;
    struct gravshort_tree_params treeacc = origtreeacc;
    const double rho0 = CP->Omega0 * CP->RhoCrit;
    grav_short_pair(&Act, pm, &Tree, treeacc.Rcut, rho0);

    copy_and_mean_accn(PairAccn);
    message(0, "GravShort Pairs %s\n", GDB_format_particle(0));
    char * fname = fastpm_strdup_printf("%s/PART-pairs-%03d", OutputDir, RestartSnapNum);

    petaio_save_snapshot(fname, &IOTable, 0, header->TimeSnapshot, CP);

    treeacc.ErrTolForceAcc = 0;
    treeacc.BHOpeningAngle = 0;
    set_gravshort_treepar(treeacc);
    grav_short_tree(&Act, pm, &Tree, NULL, rho0, times.Ti_Current);

    /* This checks fully opened tree force against pair force*/
    double meanerr, maxerr, meanangle, maxangle;
    check_accns(&meanerr,&maxerr, &meanangle, &maxangle, PairAccn);
    message(0, "Force error, open tree vs pairwise. max : %g mean: %g angle %g max angle %g forcetol: %g\n", maxerr, meanerr, meanangle, maxangle, treeacc.ErrTolForceAcc);

    if(maxerr > 0.1)
        endrun(2, "Fully open tree force does not agree with pairwise calculation! maxerr %g > 0.1!\n", maxerr);

    message(0, "GravShort Tree %s\n", GDB_format_particle(0));
    fname = fastpm_strdup_printf("%s/PART-tree-open-%03d", OutputDir, RestartSnapNum);
    petaio_save_snapshot(fname, &IOTable, 0, header->TimeSnapshot, CP);

    /* This checks tree force against tree force with zero error (which always opens).*/
    copy_and_mean_accn(PairAccn);

    /* Check that we get the same answer if we fill up the exchange buffer*/
    const size_t maxbuf = 2*1024*1024L;
    grav_short_tree(&Act, pm, &Tree, NULL, rho0, times.Ti_Current, maxbuf);
    /* This checks fully opened tree force against pair force*/
    check_accns(&meanerr,&maxerr, &meanangle, &maxangle, PairAccn);
    message(0, "Force error, filling buffer vs not, open tree. max : %g mean: %g angle %g max angle %g forcetol: %g\n", maxerr, meanerr, meanangle, maxangle, treeacc.ErrTolForceAcc);
    if(maxerr > 1e-7)
        endrun(2, "Found force error when filling buffer!  %g  (mean %g). Buffer %ld\n", maxerr, meanerr, maxbuf);

    treeacc = origtreeacc;
    set_gravshort_treepar(treeacc);
    grav_short_tree(&Act, pm, &Tree, NULL, rho0, times.Ti_Current);
    grav_short_tree(&Act, pm, &Tree, NULL, rho0, times.Ti_Current);
    fname = fastpm_strdup_printf("%s/PART-tree-%03d", OutputDir, RestartSnapNum);
    petaio_save_snapshot(fname, &IOTable, 0, header->TimeSnapshot, CP);

    check_accns(&meanerr,&maxerr,&meanangle, &maxangle, PairAccn);
    message(0, "Force error, open tree vs tree. max : %g mean: %g angle %g max angle %g forcetol: %g\n", maxerr, meanerr, meanangle, maxangle, treeacc.ErrTolForceAcc);

    if(meanerr > treeacc.ErrTolForceAcc* 1.2)
        endrun(2, "Average force error is underestimated: %g > 1.2 * %g!\n", meanerr, treeacc.ErrTolForceAcc);

    const double defaultmeanerr = meanerr;
    const double defaultmaxerr = maxerr;
    /* This checks the tree against a larger Rcut.*/
    treeacc.Rcut = 9.5;
    set_gravshort_treepar(treeacc);
    grav_short_tree(&Act, pm, &Tree, NULL, rho0, times.Ti_Current);
    grav_short_tree(&Act, pm, &Tree, NULL, rho0, times.Ti_Current);
    fname = fastpm_strdup_printf("%s/PART-tree-rcut-%03d", OutputDir, RestartSnapNum);
    petaio_save_snapshot(fname, &IOTable, 0, header->TimeSnapshot, CP);

    check_accns(&meanerr,&maxerr,&meanangle, &maxangle, PairAccn);
    message(0, "Force error, Rcut=%g. max : %g mean: %g angle %g max angle %g\n", treeacc.Rcut, maxerr, meanerr, meanangle, maxangle);

    if(meanerr > treeacc.ErrTolForceAcc)
        endrun(2, "Rcut decreased but error increased %g > %g or %g > %g\n", maxerr, defaultmaxerr, meanerr, defaultmeanerr);

    /* This checks the tree against a box with a smaller Nmesh.*/
    treeacc = origtreeacc;
    force_tree_free(&Tree);

    petapm_destroy(pm);

    gravpm_init_periodic(pm, PartManager->BoxSize, Asmth, Nmesh/2., CP->GravInternal);
    gravpm_force(pm, ddecomp, CP, header->TimeSnapshot, header->UnitLength_in_cm, OutputDir, header->TimeIC);
    force_tree_full(&Tree, ddecomp, 0, OutputDir);
    set_gravshort_treepar(treeacc);
    grav_short_tree(&Act, pm, &Tree, NULL, rho0, times.Ti_Current);
    grav_short_tree(&Act, pm, &Tree, NULL, rho0, times.Ti_Current);
    fname = fastpm_strdup_printf("%s/PART-tree-nmesh2-%03d", OutputDir, RestartSnapNum);
    petaio_save_snapshot(fname, &IOTable, 0, header->TimeSnapshot, CP);

    check_accns(&meanerr,&maxerr,&meanangle, &maxangle, PairAccn);
    message(0, "Force error, nmesh %d vs %d: max : %g mean: %g angle %g max angle %g\n", Nmesh, Nmesh/2, maxerr, meanerr, meanangle, maxangle);

    if(maxerr < defaultmaxerr || meanerr < defaultmeanerr)
        endrun(2, "Nmesh decreased but force accuracy better %g < %g or %g < %g\n", maxerr, defaultmaxerr, meanerr, defaultmeanerr);

    force_tree_free(&Tree);
    petapm_destroy(pm);

    myfree(PairAccn);

    destroy_io_blocks(&IOTable);
    domain_free(ddecomp);
}

/* Compute accelerations using two different routines, check they give the same answer. */
void
run_consistency_test(int RestartSnapNum, Cosmology * CP, const double Asmth, const int Nmesh, const inttime_t Ti_Current, const char * OutputDir, const struct header_data * header)
{
    DomainDecomp ddecomp[1] = {0};
    domain_decompose_full(ddecomp, MPI_COMM_WORLD);

    struct IOTable IOTable = {0};
    /* NO metals written*/
    register_io_blocks(&IOTable, 0, 0);
    register_extra_blocks(&IOTable);

    double (* PairAccn)[3] = (double (*) [3]) mymalloc2("PairAccns", 3*sizeof(double) * PartManager->NumPart);

    PetaPM pm[1] = {0};
    gravpm_init_periodic(pm, PartManager->BoxSize, Asmth, Nmesh, CP->GravInternal);

    int NTask;
    MPI_Comm_size(MPI_COMM_WORLD, &NTask);

    DriftKickTimes times = init_driftkicktime(Ti_Current);
    /* All particles are active*/
    ActiveParticles Act = init_empty_active_particles(PartManager);
    build_active_particles(&Act, &times, 0, header->TimeSnapshot, PartManager);

    gravpm_force(pm, ddecomp, CP, header->TimeSnapshot, header->UnitLength_in_cm, OutputDir, header->TimeIC);

    ForceTree Tree = {0};
    force_tree_full(&Tree, ddecomp, 0, OutputDir);

    struct gravshort_tree_params treeacc = get_gravshort_treepar();
    /* Reset to normal tree */
    if(treeacc.TreeUseBH > 1)
        treeacc.TreeUseBH = 0;
    /* Compare the new and old gravity tree. */
    const double rho0 = CP->Omega0 * CP->RhoCrit;
    grav_short_tree(&Act, pm, &Tree, NULL, rho0, times.Ti_Current);
    copy_and_mean_accn(PairAccn);
    grav_short_tree_old(&Act, pm, &Tree, NULL, rho0, times.Ti_Current);

    /* This checks fully opened tree force against pair force*/
    double meanerr, maxerr, meanangle, maxangle;
    check_accns(&meanerr,&maxerr, &meanangle, &maxangle, PairAccn);
    message(0, "Force error, new grav tree vs old gravtree. max : %g mean: %g angle %g max angle %g forcetol: %g\n", maxerr, meanerr, meanangle, maxangle, treeacc.ErrTolForceAcc);

    if(maxerr > 0.1)
        endrun(2, "New and old tree forces do not agree! maxerr %g > 0.1!\n", maxerr);

    /* Check density code is the same */

    /* Check hydro code is the same */

    char * fname = fastpm_strdup_printf("%s/PART-consistent-%03d", OutputDir, RestartSnapNum);
    petaio_save_snapshot(fname, &IOTable, 0, header->TimeSnapshot, CP);

    force_tree_free(&Tree);
    petapm_destroy(pm);

    myfree(PairAccn);

    destroy_io_blocks(&IOTable);
    domain_free(ddecomp);
}
