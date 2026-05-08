#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <string>
#include <dirent.h>
#include <vector>
#include "lenstools.h"
#include "utils/mymalloc.h"
#include "cosmology.h"
#include "plane.h"
#include "physconst.h"
#include "partmanager.h"
#include "utils/endrun.h"
#include "utils/system.h"
#include "timebinmgr.h"

static struct plane_params
{
    int Resolution;
    double Thickness; // in kpc/h
    std::vector<double> CutPoints;
    std::vector<int> Normals;
} PlaneParams;

std::string
plane_get_output_fname(const int snapnum, const std::string OutputDir, const int cut, const int normal)
{
    // Format the filename to include '!' to overwrite existing files
    return "!" + OutputDir + "/snap" + std::to_string(snapnum) + "_potentialPlane" + std::to_string(cut) + "_normal" + std::to_string(normal) + ".fits";
}

/*Set the plane parameters*/
void
set_plane_params(ParameterSet * ps)
{
    // plane resolution
    PlaneParams.Resolution = param_get_int(ps, "PlaneResolution");

    // plane thickness in internal units (kpc/h)
    PlaneParams.Thickness = param_get_double(ps, "PlaneThickness");
    // Plane normals
    PlaneParams.Normals = BuildOutputList<int>(param_get_string(ps, "PlaneNormals"));
    int nmax = *std::max_element(PlaneParams.Normals.begin(), PlaneParams.Normals.end());
    int nmin = *std::min_element(PlaneParams.Normals.begin(), PlaneParams.Normals.end());
    if(nmax > 2 || nmin < 0)
        endrun(4, "Requesting a normal direction beyond 0, 1 and 2: max: %d, min %d\n", nmax, nmin);
    // Plane cut points
    PlaneParams.CutPoints = BuildOutputList<double>(param_get_string(ps, "PlaneCutPoints"));
}

void write_plane(int snapnum, const double atime, Cosmology * CP, const std::string OutputDir, const double UnitVelocity_in_cm_per_s, const double UnitLength_in_cm)
{
    double BoxSize = PartManager->BoxSize;

    /* NOTE: this is correct only for pure DM runs because this code is called on a PM step and we garbage collect after the exchange.
     * It is not generally the total number of particles*/
    int64_t num_particles_tot = 0; // number of dark matter particles
    // Use MPI_Allreduce to get the total number of particles on all ranks
    MPI_Allreduce(&PartManager->NumPart, &num_particles_tot, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);
    // printf("Total number of particles: %ld\n", num_particles_tot);

    // plane parameters
    int plane_resolution = PlaneParams.Resolution;
    if (PlaneParams.Thickness <= 0.0) {
        message(0, "No positive thickness provided, the side length of the box, %g, will be used.\n", BoxSize);
        PlaneParams.Thickness = BoxSize;
    }

    if (PlaneParams.CutPoints.size() == 0) {
        message(0, "No cut points provided, a set of default values will be set: (1/2 + i) * plane thickness (< box size, i = 0, 1, 2...)\n");
        PlaneParams.CutPoints.resize((BoxSize / PlaneParams.Thickness));
        for (size_t i = 0; i < PlaneParams.CutPoints.size(); i++) {
            PlaneParams.CutPoints[i] = (.5 + i) * PlaneParams.Thickness;
            message(0,"CutPoints[%lu] = %g\n", i, PlaneParams.CutPoints[i]);
        }
    }

    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);

    double redshift = 1./atime - 1.;
    message(0, "Computing and writing potential planes.\n");

    double *plane_result = allocate_2d_array_as_1d(plane_resolution, plane_resolution);
    double *summed_plane_result = allocate_2d_array_as_1d(plane_resolution, plane_resolution);

    double comoving_distance = compute_comoving_distance(CP, atime, 1., UnitVelocity_in_cm_per_s);

    // print comoving distance
    message(0, "Comoving distance: %g\n", comoving_distance);

    /* loop over cut points and normal directions to generate lensing potential planes */
    for (size_t i = 0; i < PlaneParams.CutPoints.size(); i++) {
        for (int j = 0; j < PlaneParams.Normals.size(); j++) {
            message(0, "Computing for cut point %g and normal %d\n", PlaneParams.CutPoints[i], PlaneParams.Normals[j]);

            double left_corner[3] = {0, 0, 0};
            int64_t num_particles_plane = 0, num_particles_plane_tot = 0;

            memset(plane_result, 0, plane_resolution * plane_resolution * sizeof(double));

            /*computing lensing potential planes*/
            num_particles_plane = cutPlaneGaussianGrid(num_particles_tot,  comoving_distance, BoxSize, CP, atime, PlaneParams.Normals[j], PlaneParams.CutPoints[i], PlaneParams.Thickness, left_corner, plane_resolution, plane_result);

            /*sum up planes from all tasks*/
            MPI_Reduce(plane_result, summed_plane_result, plane_resolution * plane_resolution, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(&num_particles_plane, &num_particles_plane_tot, 1, MPI_INT64, MPI_SUM, 0, MPI_COMM_WORLD);

            /*saving planes*/
            if (ThisTask == 0) {
#ifdef USE_CFITSIO
                auto file_path = plane_get_output_fname(snapnum, OutputDir, i, PlaneParams.Normals[j]);
                savePotentialPlane(summed_plane_result, plane_resolution, plane_resolution, file_path, BoxSize, CP, redshift, comoving_distance, num_particles_plane_tot, UnitLength_in_cm);
                message(0, "Plane saved for cut %d and normal %d to %s\n", i, PlaneParams.Normals[j], file_path.c_str() + 1); // skip the '!' in the filename
#endif
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }
    myfree(summed_plane_result);
    myfree(plane_result);


    if (ThisTask == 0) {
        double comoving_distance_Mpc  = comoving_distance * UnitLength_in_cm / CM_PER_MPC;
        std::string buf = OutputDir + "/info.txt";
        FILE * fd = fopen(buf.c_str(), "a");
        fprintf(fd, "s=%d,d=%lf Mpc/h,z=%lf\n", snapnum, comoving_distance_Mpc, redshift);
        fclose(fd);
    }
}
