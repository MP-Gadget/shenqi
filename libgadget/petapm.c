#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
/* do NOT use complex.h it breaks the code */

#include "types.h"
#include "petapm.h"
#include "pm_kernel.cuh"

#include "utils.h"
#include "walltime.h"


static void
layout_prepare(PetaPM * pm,
               struct Layout * L,
               double * meshbuf,
               PetaPMRegion * regions,
               const int Nregions,
               MPI_Comm comm);
static void layout_finish(struct Layout * L);
static void layout_build_and_exchange_cells_to_fft(PetaPM * pm, struct Layout * L, double * meshbuf, double * real);
static void layout_build_and_exchange_cells_to_local(PetaPM * pm, struct Layout * L, double * meshbuf, double * real);

/* cell_iterator needs to be thread safe !*/
typedef void (* cell_iterator)(double * cell_value, double * comm_buffer);
static void layout_iterate_cells(PetaPM * pm, struct Layout * L, cell_iterator iter, double * real);

struct Pencil { /* a pencil starting at offset, with lenght len */
    int offset[3];
    int len;
    int first;
    int meshbuf_first; /* first pixel in meshbuf */
    int task;
};
static int pencil_cmp_target(const void * v1, const void * v2);
static int pos_get_target(PetaPM * pm, const int pos[2]);

/* FIXME: move this to MPIU_. */
static int64_t reduce_int64(int64_t input, MPI_Comm comm);
#ifdef DEBUG
/* for debugging */
static void verify_density_field(PetaPM * pm, double * real, double * meshbuf, const size_t meshsize);
#endif

static MPI_Datatype MPI_PENCIL;

/*Used only in MP-GenIC*/
cufftComplex *
petapm_alloc_rhok(PetaPM * pm)
{
    cufftComplex * rho_k = (cufftComplex * ) mymalloc("PMrho_k", pm->priv->fftsize * sizeof(double));
    memset(rho_k, 0, pm->priv->fftsize * sizeof(double));
    return rho_k;
}

static void pm_init_regions(PetaPM * pm, PetaPMRegion * regions, const int Nregions);

static PetaPMParticleStruct * CPS; /* stored by petapm_force, how to access the P array */
#define POS(i) ((double*)  (&((char*)CPS->Parts)[CPS->elsize * (i) + CPS->offset_pos]))
#define MASS(i) ((float*) (&((char*)CPS->Parts)[CPS->elsize * (i) + CPS->offset_mass]))
#define INACTIVE(i) (CPS->active && !CPS->active(i))


PetaPMRegion * petapm_get_fourier_region(PetaPM * pm) {
    return &pm->fourier_space_region;
}
PetaPMRegion * petapm_get_real_region(PetaPM * pm) {
    return &pm->real_space_region;
}
int petapm_mesh_to_k(PetaPM * pm, int i) {
    /*Return the position of this point on the Fourier mesh*/
    return i<=pm->Nmesh/2 ? i : (i-pm->Nmesh);
}
int *petapm_get_thistask2d(PetaPM * pm) {
    return pm->ThisTask2d;
}
int *petapm_get_ntask2d(PetaPM * pm) {
    return pm->NTask2d;
}

void
petapm_module_init(int Nthreads)
{
    // CUDA Device Initialization if necessary (optional if only one GPU is used)
    int device_id = 0;
    cudaSetDevice(device_id);  // Set the active GPU device

    // Handle CPU threads manually, if needed (optional if not using multithreading on the CPU)
    // #ifdef _OPENMP
    // omp_set_num_threads(Nthreads); // Set number of threads for OpenMP parallelism
    // #endif
    // cuFFT itself is inherently multithreaded; no cuFFT-specific thread setting needed

    MPI_Type_contiguous(sizeof(struct Pencil), MPI_BYTE, &MPI_PENCIL);
    MPI_Type_commit(&MPI_PENCIL);
}

void
petapm_init(PetaPM * pm, double BoxSize, double Asmth, int Nmesh, double G, MPI_Comm comm)
{
    /* define the global long / short range force cut */
    pm->BoxSize = BoxSize;
    pm->Asmth = Asmth;
    pm->Nmesh = Nmesh;
    pm->G = G;
    pm->CellSize = BoxSize / Nmesh;
    pm->comm = comm;

    int ThisTask;
    int NTask;
    pm->Mesh2Task[0] = (int *) mymalloc2("Mesh2Task", 2*sizeof(int) * Nmesh);
    pm->Mesh2Task[1] = pm->Mesh2Task[0] + Nmesh;
    MPI_Comm_rank(comm, &ThisTask);
    MPI_Comm_size(comm, &NTask);
    
    int ndevices;
    cudaGetDeviceCount(&ndevices);
    cudaSetDevice(ThisTask % ndevices);

    message(0, "Cuda Devices %d \n", ndevices);

    /* try to find a square 2d decomposition */
    /* CUDA NOTE: CufftMp only supports square decomposition, 
    so Ntask has to be a perfect square*/
    int nranks1d;
    nranks1d = sqrt(NTask);
    if (nranks1d != NTask/nranks1d) {
        endrun(0, "Error: The number of MPI ranks has to be a perfect square for CufftMp\n");
    }

    message(0, "Using 2D Task mesh %d x %d \n", nranks1d, nranks1d);
    
    // Define custom data distribution
    int64 nx               = Nmesh;
    int64 ny               = Nmesh;
    int64 nz               = Nmesh;
    int64 nz_real          = nz;
    int64 nz_complex       = (nz/2+1);
    int64 nz_real_padded   = 2*nz_complex;

    // create 2D cartesian MPI comm without pfft
    int dims[2] = {nranks1d, nranks1d};
    int periods[2] = {0, 0};  // non-periodic in both dimensions
    // Allow the ranks to be reordered by MPI for efficiency
    // Actually don't allow reordering for now to be safe
    int reorder = 0;
    MPI_Cart_create(comm, 2, dims, periods, reorder, &pm->priv->comm_cart_2d);
    if (pm->priv->comm_cart_2d == MPI_COMM_NULL) {
        endrun(0, "Error: comm_cart_2d is MPI_COMM_NULL\n");
    }
    MPI_Cart_get(pm->priv->comm_cart_2d, 2, pm->NTask2d, periods, pm->ThisTask2d);
    message(1, "Task = %d ThisTask2d = (%d, %d) Ntask2d = (%d, %d) \n", 
        ThisTask, pm->ThisTask2d[0], pm->ThisTask2d[1], pm->NTask2d[0], pm->NTask2d[1]);


    // compute offset, size and strides
    auto displacement = [](int64 length, int rank, int size) {
        int ranks_cutoff = length % size;
        int chunk_size = length / size;
        return (rank < ranks_cutoff ? rank * (chunk_size + 1) : ranks_cutoff * (chunk_size + 1) + (rank - ranks_cutoff) * chunk_size);
    };
    
    // update region properties, also have a redundant box struct for now to use box_iterator, will merge it to region
    auto update_region_and_box = [](int64 lower[3], int64 upper[3], int64 strides[3], PetaPMRegion &region, Box3D &box) {
        region.totalsize = 1;
        for (int i = 0; i < 3; i++) {
            region.offset[i]  = lower[i];
            region.upper[i]   = upper[i];
            region.size[i]    = upper[i] - lower[i];
            region.strides[i] = strides[i];
            region.totalsize *= region.size[i];
            // init box3d
            box.lower[i] = lower[i];
            box.upper[i] = upper[i];
            box.strides[i] = strides[i];
        }
        region.buffer = NULL;
    };
    
    int i = ThisTask / nranks1d;
    int j = ThisTask % nranks1d;
    
    // real region setup
    // note the petapm->region has non-padded strides, while cufft takes in padded strides
    int64 lower_real[3]   = {displacement(nx, i, nranks1d), displacement(ny, j, nranks1d), 0};
    int64 upper_real[3]   = {displacement(nx, i+1, nranks1d), displacement(ny, j+1, nranks1d), nz_real};
    int64 strides_real[3] = {(upper_real[1] - lower_real[1]) * nz_real_padded, nz_real_padded, 1};
    int64 strides_real_nopad[3] = {(upper_real[1] - lower_real[1]) * nz_real, nz_real, 1};

    update_region_and_box(lower_real, upper_real, strides_real_nopad, pm->real_space_region, pm->box_real);
    
    // complex region setup
    int64 lower_fourier[3]   = {displacement(nx, i, nranks1d), 0, displacement(nz_complex, j, nranks1d)};
    int64 upper_fourier[3]   = {displacement(nx, i+1, nranks1d), ny, displacement(nz_complex, j+1, nranks1d)};
    int64 strides_fourier[3] = {(upper_fourier[1] - lower_fourier[1]) * (upper_fourier[2] - lower_fourier[2]), (upper_fourier[2] - lower_fourier[2]), 1};
    update_region_and_box(lower_fourier, upper_fourier, strides_fourier, pm->fourier_space_region, pm->box_complex);

    //===============================================================================================
    cudaStreamCreate(&pm->priv->stream);
    cufftCreate(&pm->priv->plan_forw);
    cufftCreate(&pm->priv->plan_back);
    // Attach the MPI communicator to the plans
    cufftMpAttachComm(pm->priv->plan_forw, CUFFT_COMM_MPI, &comm);
    cufftMpAttachComm(pm->priv->plan_back, CUFFT_COMM_MPI, &comm);

    // Describe the data distribution (only for custumized data decomposition, not needed for default slab decomposition)
    // R2C plans only support CUFFT_XT_FORMAT_DISTRIBUTED_INPUT and always perform a CUFFT_FORWARD transform
    // C2R plans only support CUFFT_XT_FORMAT_DISTRIBUTED_OUTPUT ans always perform a CUFFT_INVERSE transform
    // So, in both, the "input" box should be the real box and the "output" box should be the complex box
    cufftXtSetDistribution(pm->priv->plan_forw, 3, lower_real, upper_real, lower_fourier, upper_fourier, strides_real, strides_fourier);
    cufftXtSetDistribution(pm->priv->plan_back, 3, lower_real, upper_real, lower_fourier, upper_fourier, strides_real, strides_fourier);

    // Set the stream
    cufftSetStream(pm->priv->plan_forw, pm->priv->stream);
    cufftSetStream(pm->priv->plan_back, pm->priv->stream);

    // Make the plan
    size_t workspace;
    cufftMakePlan3d(pm->priv->plan_forw, Nmesh, Nmesh, Nmesh, CUFFT_R2C, &workspace);
    cufftMakePlan3d(pm->priv->plan_back, Nmesh, Nmesh, Nmesh, CUFFT_C2R, &workspace);

    // Allocate GPU memory, copy CPU data to GPU
    // Data is initially distributed according to CUFFT_XT_FORMAT_DISTRIBUTED_INPUT, i.e., box_real
    cudaLibXtDesc *desc;
    cufftXtMalloc(pm->priv->plan_forw, &desc, CUFFT_XT_FORMAT_DISTRIBUTED_INPUT);
    pm->priv->fftsize = desc->descriptor->size[0];
    //===============================================================================================

    message(1, "Task %d NGPUs=%d, pfftsize=%d \n", ThisTask, desc->descriptor->nGPUs, pm->priv->fftsize);
    /* now lets fill up the mesh2task arrays */
    #if 0
        message(1, "Real Box3d %d lower=(%td %td %td) upper=(%td %td %td) strides=(%td %td %td)\n", ThisTask,
                pm->box_real.lower[0],
                pm->box_real.lower[1],
                pm->box_real.lower[2],
                pm->box_real.upper[0],
                pm->box_real.upper[1],
                pm->box_real.upper[2],
                pm->box_real.strides[0],
                pm->box_real.strides[1],
                pm->box_real.strides[2]);
        message(1, "Complex Region %d lower=(%td %td %td) upper=(%td %td %td) strides=(%td %td %td)\n", ThisTask,
                pm->fourier_space_region.offset[0],
                pm->fourier_space_region.offset[1],
                pm->fourier_space_region.offset[2],
                pm->fourier_space_region.upper[0],
                pm->fourier_space_region.upper[1],
                pm->fourier_space_region.upper[2],
                pm->fourier_space_region.strides[0],
                pm->fourier_space_region.strides[1],
                pm->fourier_space_region.strides[2]);
    #endif
        int * tmp = (int *) mymalloc("tmp", sizeof(int) * Nmesh);
        int k;
        for(k = 0; k < 2; k ++) {
            for(i = 0; i < Nmesh; i ++) {
                tmp[i] = 0;
            }
            for(i = 0; i < pm->real_space_region.size[k]; i ++) {
                tmp[i + pm->real_space_region.offset[k]] = pm->ThisTask2d[k];
            }
            /* which column / row hosts this tile? */
            /* FIXME: this is very inefficient */
            MPI_Allreduce(tmp, pm->Mesh2Task[k], Nmesh, MPI_INT, MPI_MAX, comm);
            // for(i = 0; i < Nmesh; i ++) {
            //     message(0, "Mesh2Task[%d][%d] == %d\n", k, i, pm->Mesh2Task[k][i]);
            // }
        }
        myfree(tmp);
}

void
petapm_destroy(PetaPM * pm)
{
    cufftDestroy(pm->priv->plan_forw);
    cufftDestroy(pm->priv->plan_back);
    cudaStreamDestroy(pm->priv->stream);
    MPI_Comm_free(&pm->priv->comm_cart_2d);
    myfree(pm->Mesh2Task[0]);
}

/*
 * read out field to particle i, with value no need to be thread safe
 * (particle i is never done by same thread)
 * */
typedef void (* pm_iterator)(PetaPM * pm, int i, double * mesh, double weight);
static void pm_iterate(PetaPM * pm, pm_iterator iterator, PetaPMRegion * regions, const int Nregions);
/* apply transfer function to value, kpos array is in x, y, z order */
static void pm_apply_transfer_function(PetaPM * pm,
        cufftComplex * src,
        cufftComplex * dst, petapm_transfer_func H);

static void put_particle_to_mesh(PetaPM * pm, int i, double * mesh, double weight);
/*
 * 1. calls prepare to build the Regions covering particles
 * 2. CIC the particles
 * 3. Transform to rho_k
 * 4. apply global_transfer (if not NULL --
 *       this is the place to fill in gaussian seeds,
 *       the transfer is stacked onto all following transfers.
 * 5. for each transfer, readout in functions
 * 6.    apply transfer from global_transfer -> complex
 * 7.    transform to real
 * 8.    readout
 * 9. free regions
 * */

PetaPMRegion *
petapm_force_init(
        PetaPM * pm,
        petapm_prepare_func prepare,
        PetaPMParticleStruct * pstruct,
        int * Nregions,
        void * userdata) {
    CPS = pstruct;

    *Nregions = 0;
    PetaPMRegion * regions = prepare(pm, pstruct, userdata, Nregions);
    pm_init_regions(pm, regions, *Nregions);

    pm_iterate(pm, put_particle_to_mesh, regions, *Nregions);

    layout_prepare(pm, &pm->priv->layout, pm->priv->meshbuf, regions, *Nregions, pm->comm);

    walltime_measure("/PMgrav/init");
    return regions;
}



static void pm_apply_transfer_function(PetaPM * pm,
        cufftComplex * src,
        cufftComplex * dst, petapm_transfer_func H
        ){
    size_t ip = 0;
    

    PetaPMRegion * region = &pm->fourier_space_region;
    message(1, "**region size %d; pfftsize %d \n", region->totalsize, pm->priv->fftsize);

#pragma omp parallel for
    for(ip = 0; ip < region->totalsize; ip ++) {
        ptrdiff_t tmp = ip;
        int pos[3];
        int kpos[3];
        int64_t k2 = 0.0;
        int k;
        for(k = 0; k < 3; k ++) {
            pos[k] = tmp / region->strides[k];
            tmp -= pos[k] * region->strides[k];
            /* lets get the abs pos on the grid*/
            pos[k] += region->offset[k];
            /* check */
            if(pos[k] >= pm->Nmesh) {
                endrun(1, "position didn't make sense\n");
            }
            kpos[k] = petapm_mesh_to_k(pm, pos[k]);
            /* Watch out the cast */
            k2 += ((int64_t)kpos[k]) * kpos[k];
        }
        /* swap 0 and 1 because fourier space was transposed */
        /* kpos is y, z, x */
        pos[0] = kpos[2];
        pos[1] = kpos[0];
        pos[2] = kpos[1];
        message(1, "ip=%d\n", ip);

        dst[ip].x = src[ip].x;
        dst[ip].y = src[ip].y;

        message(1, "dst=%f \n", dst[ip].x);
        message(1, "src=%f \n", src[ip].x);
        
        if(H) {
            H(pm, k2, pos, &dst[ip]);
        }
    }

}

cufftComplex * petapm_force_r2c(PetaPM * pm,
        PetaPMGlobalFunctions * global_functions
        ) {
     // CUDA TODO: figureout how to properly get fftsize
    double * real = (double * ) mymalloc2("PMreal", pm->priv->fftsize * sizeof(double));
    memset(real, 0, sizeof(double) * pm->priv->fftsize);
    layout_build_and_exchange_cells_to_fft(pm, &pm->priv->layout, pm->priv->meshbuf, real);
    walltime_measure("/PMgrav/comm2");
#ifdef DEBUG
    verify_density_field(pm, real, pm->priv->meshbuf, pm->priv->meshbufsize);
    walltime_measure("/PMgrav/Verify");
#endif

    // CUDA TODO: figure out if this is needed
    // Allocate GPU memory, copy CPU data to GPU
    // Data is initially distributed according to CUFFT_XT_FORMAT_INPLACE
    cufftXtMalloc(pm->priv->plan_forw, &pm->priv->desc, CUFFT_XT_FORMAT_INPLACE);
    // copy real array to gpu
    cufftXtMemcpy(pm->priv->plan_forw, pm->priv->desc, real, CUFFT_COPY_HOST_TO_DEVICE);
    message(1, "Real array first element %f\n", real[0]);
    
    // execute the plan
    cufftXtExecDescriptor(pm->priv->plan_forw, pm->priv->desc, pm->priv->desc, CUFFT_FORWARD);
    myfree(real);

     // CUDA TODO: need to check if the output complex array is transpose
     // need to verify
    //=============================== End of R2C =============================================
    //========================== Begin Transfer Function =====================================
    int ThisTask;
    int NTask;
    MPI_Comm_rank(pm->comm, &ThisTask);
    MPI_Comm_size(pm->comm, &NTask);
    
    cufftComplex * rho_k = (cufftComplex * ) mymalloc2("PMrho_k", pm->priv->fftsize * sizeof(double));

    launch_potential_transfer(pm->box_complex, (cufftComplex *) pm->priv->desc->descriptor->data[0], ThisTask, NTask, pm, pm->priv->stream);
    message(1, "Simple kernel suceeded \n");
        
    /*Do any analysis that may be required before the transfer function is applied*/
    /* CUDA Note: global readout and analysis is NULL unless CP->MassiveNuLinRespOn*/
    /* CUDA TODO: add back the CP->MassiveNuLinRespOn function later*/
    
    // /*Apply the transfer function*/
    /* global transfer is potential transfer in gravpm*/
    // petapm_transfer_func global_transfer = global_functions->global_transfer;
    // pm_apply_transfer_function(pm, complex_data, rho_k, global_transfer);
    walltime_measure("/PMgrav/r2c");
    return rho_k;
}

void
petapm_force_c2r(PetaPM * pm,
        cufftComplex * rho_k,
        PetaPMRegion * regions,
        const int Nregions,
        PetaPMFunctions * functions)
{
    // For grav the functions are: potential, forcex, forcey, forcez, 
    // where the potential has no transfer function, only readout
    // as the potential transfer is applied in r2c
    PetaPMFunctions * f = functions;
    for (f = functions; f->name; f ++) {
        petapm_transfer_func transfer = f->transfer;
        petapm_readout_func readout = f->readout;

        /* apply the greens function turn rho_k into potential in fourier space */
        // pm_apply_transfer_function(pm, rho_k, (cufftComplex*) pm->priv->desc->descriptor->data[0], transfer);
        walltime_measure("/PMgrav/calc");
        // execute c2r
        cufftXtExecDescriptor(pm->priv->plan_back, pm->priv->desc, pm->priv->desc, CUFFT_INVERSE);
        cudaStreamSynchronize(pm->priv->stream);
        // copy data back to cpu
        double * real = (double * ) mymalloc2("PMreal", pm->priv->fftsize * sizeof(double));
        cufftXtMemcpy(pm->priv->plan_back, real, pm->priv->desc, CUFFT_COPY_DEVICE_TO_HOST);
        cufftXtFree(pm->priv->desc);

        walltime_measure("/PMgrav/c2r");
        if(f == functions) // Once
            report_memory_usage("PetaPM");
        message(1, "FREED DESC ***************** \n");
        /* read out the potential: this will copy and free real.*/
        layout_build_and_exchange_cells_to_local(pm, &pm->priv->layout, pm->priv->meshbuf, real);
        walltime_measure("/PMgrav/comm");

        pm_iterate(pm, readout, regions, Nregions);
        walltime_measure("/PMgrav/readout");
//    }
    
        message(1, "READ OUT DONE ***************** \n");
}

void petapm_force_finish(PetaPM * pm) {
    layout_finish(&pm->priv->layout);
    myfree(pm->priv->meshbuf);
}

void petapm_force(PetaPM * pm, petapm_prepare_func prepare,
        PetaPMGlobalFunctions * global_functions, //petapm_transfer_func global_transfer,
        PetaPMFunctions * functions,
        PetaPMParticleStruct * pstruct,
        void * userdata) {
    int Nregions;
    PetaPMRegion * regions = petapm_force_init(pm, prepare, pstruct, &Nregions, userdata);
    cufftComplex * rho_k = petapm_force_r2c(pm, global_functions);
    if(functions)
        petapm_force_c2r(pm, rho_k, regions, Nregions, functions);
    myfree(rho_k);
    if(CPS->RegionInd)
        myfree(CPS->RegionInd);
    myfree(regions);
    petapm_force_finish(pm);
}

/******************************************************************************************************************************************** */
/* build a communication layout */

static void layout_build_pencils(PetaPM * pm, struct Layout * L, double * meshbuf, PetaPMRegion * regions, const int Nregions);
static void layout_exchange_pencils(struct Layout * L);
static void
layout_prepare (PetaPM * pm,
                struct Layout * L,
                double * meshbuf,
                PetaPMRegion * regions,
                const int Nregions,
                MPI_Comm comm)
{
    int r;
    int i;
    int NTask;
    L->comm = comm;

    MPI_Comm_size(L->comm, &NTask);

    L->ibuffer = (int *) mymalloc("PMlayout", sizeof(int) * NTask * 8);

    memset(L->ibuffer, 0, sizeof(int) * NTask * 8);
    L->NpSend = &L->ibuffer[NTask * 0];
    L->NpRecv = &L->ibuffer[NTask * 1];
    L->NcSend = &L->ibuffer[NTask * 2];
    L->NcRecv = &L->ibuffer[NTask * 3];
    L->DcSend = &L->ibuffer[NTask * 4];
    L->DcRecv = &L->ibuffer[NTask * 5];
    L->DpSend = &L->ibuffer[NTask * 6];
    L->DpRecv = &L->ibuffer[NTask * 7];

    L->NpExport = 0;
    L->NcExport = 0;
    L->NpImport = 0;
    L->NcImport = 0;

    int NpAlloc = 0;
    /* count pencils until buffer would run out */
    for (r = 0; r < Nregions; r ++) {
        NpAlloc += regions[r].size[0] * regions[r].size[1];
    }

    L->PencilSend = (struct Pencil *) mymalloc("PencilSend", NpAlloc * sizeof(struct Pencil));

    layout_build_pencils(pm, L, meshbuf, regions, Nregions);

    /* sort the pencils by the target rank for ease of next step */
    qsort_openmp(L->PencilSend, NpAlloc, sizeof(struct Pencil), pencil_cmp_target);
    /* zero length pixels are moved to the tail */

    /* now shrink NpExport*/
    L->NpExport = NpAlloc;
    while(L->NpExport > 0 && L->PencilSend[L->NpExport - 1].len == 0) {
        L->NpExport --;
    }

    /* count total number of cells to be exported */
    int NcExport = 0;
    for(i = 0; i < L->NpExport; i++) {
        int task = L->PencilSend[i].task;
        L->NcSend[task] += L->PencilSend[i].len;
        NcExport += L->PencilSend[i].len;
        L->NpSend[task] ++;
    }
    L->NcExport = NcExport;

    MPI_Alltoall(L->NpSend, 1, MPI_INT, L->NpRecv, 1, MPI_INT, L->comm);
    MPI_Alltoall(L->NcSend, 1, MPI_INT, L->NcRecv, 1, MPI_INT, L->comm);

    /* build the displacement array; why doesn't MPI build these automatically? */
    L->DpSend[0] = 0; L->DpRecv[0] = 0;
    L->DcSend[0] = 0; L->DcRecv[0] = 0;
    for(i = 1; i < NTask; i ++) {
        L->DpSend[i] = L->NpSend[i - 1] + L->DpSend[i - 1];
        L->DpRecv[i] = L->NpRecv[i - 1] + L->DpRecv[i - 1];
        L->DcSend[i] = L->NcSend[i - 1] + L->DcSend[i - 1];
        L->DcRecv[i] = L->NcRecv[i - 1] + L->DcRecv[i - 1];
    }
    L->NpImport = L->DpRecv[NTask -1] + L->NpRecv[NTask -1];
    L->NcImport = L->DcRecv[NTask -1] + L->NcRecv[NTask -1];

    /* some checks */
    if(L->DpSend[NTask - 1] + L->NpSend[NTask -1] != L->NpExport) {
        endrun(1, "NpExport = %d NpSend=%d DpSend=%d\n", L->NpExport, L->NpSend[NTask -1], L->DpSend[NTask - 1]);
    }
    if(L->DcSend[NTask - 1] + L->NcSend[NTask -1] != L->NcExport) {
        endrun(1, "NcExport = %d NcSend=%d DcSend=%d\n", L->NcExport, L->NcSend[NTask -1], L->DcSend[NTask - 1]);
    }
    int64_t totNpAlloc = reduce_int64(NpAlloc, L->comm);
    int64_t totNpExport = reduce_int64(L->NpExport, L->comm);
    int64_t totNcExport = reduce_int64(L->NcExport, L->comm);
    int64_t totNpImport = reduce_int64(L->NpImport, L->comm);
    int64_t totNcImport = reduce_int64(L->NcImport, L->comm);

    if(totNpExport != totNpImport) {
        endrun(1, "totNpExport = %ld\n", totNpExport);
    }
    if(totNcExport != totNcImport) {
        endrun(1, "totNcExport = %ld\n", totNcExport);
    }

    /* exchange the pencils */
    message(0, "PetaPM:  %010ld/%010ld Pencils and %010ld Cells\n", totNpExport, totNpAlloc, totNcExport);
    L->PencilRecv = (struct Pencil *) mymalloc("PencilRecv", L->NpImport * sizeof(struct Pencil));
    memset(L->PencilRecv, 0xfc, L->NpImport * sizeof(struct Pencil));
    layout_exchange_pencils(L);
}

static void
layout_build_pencils(PetaPM * pm,
                     struct Layout * L,
                     double * meshbuf,
                     PetaPMRegion * regions,
                     const int Nregions)
{
    /* now build pencils to be exported */
    int p0 = 0;
    int r;
    for (r = 0; r < Nregions; r++) {
        int ix;
#pragma omp parallel for private(ix)
        for(ix = 0; ix < regions[r].size[0]; ix++) {
            int iy;
            for(iy = 0; iy < regions[r].size[1]; iy++) {
                int poffset = ix * regions[r].size[1] + iy;
                struct Pencil * p = &L->PencilSend[p0 + poffset];

                p->offset[0] = ix + regions[r].offset[0];
                p->offset[1] = iy + regions[r].offset[1];
                p->offset[2] = regions[r].offset[2];
                p->len = regions[r].size[2];
                p->meshbuf_first = (regions[r].buffer - meshbuf) +
                    regions[r].strides[0] * ix +
                    regions[r].strides[1] * iy;
                /* now lets compress the pencil */
                while((p->len > 0) && (meshbuf[p->meshbuf_first + p->len - 1] == 0.0)) {
                    p->len --;
                }
                while((p->len > 0) && (meshbuf[p->meshbuf_first] == 0.0)) {
                    p->len --;
                    p->meshbuf_first++;
                    p->offset[2] ++;
                }

                p->task = pos_get_target(pm, p->offset);
            }
        }
        p0 += regions[r].size[0] * regions[r].size[1];
    }

}

static void layout_exchange_pencils(struct Layout * L) {
    int i;
    int offset;
    int NTask;
    MPI_Comm_size(L->comm, &NTask);
    /* build the first pointers to refer to the correct relative buffer locations */
    /* note that the buffer hasn't bee assembled yet */
    offset = 0;
    for(i = 0; i < NTask; i ++) {
        int j;
        struct Pencil * p = &L->PencilSend[offset];
        if(L->NpSend[i] == 0) continue;
        p->first = 0;
        for(j = 1; j < L->NpSend[i]; j++) {
            p[j].first = p[j - 1].first + p[j - 1].len;
        }
        offset += L->NpSend[i];
    }

    MPI_Alltoallv(
            L->PencilSend, L->NpSend, L->DpSend, MPI_PENCIL,
            L->PencilRecv, L->NpRecv, L->DpRecv, MPI_PENCIL,
            L->comm);

    /* set first to point to absolute position in the full import cell buffer */
    offset = 0;
    for(i = 0; i < NTask; i ++) {
        struct Pencil * p = &L->PencilRecv[offset];
        int j;
        for(j = 0; j < L->NpRecv[i]; j++) {
            p[j].first += L->DcRecv[i];
        }
        offset += L->NpRecv[i];
    }

    /* set first to point to absolute position in the full export cell buffer */
    offset = 0;
    for(i = 0; i < NTask; i ++) {
        struct Pencil * p = &L->PencilSend[offset];
        int j;
        for(j = 0; j < L->NpSend[i]; j++) {
            p[j].first += L->DcSend[i];
        }
        offset += L->NpSend[i];
    }
}

static void layout_finish(struct Layout * L) {
    myfree(L->PencilRecv);
    myfree(L->PencilSend);
    myfree(L->ibuffer);
}

/* exchange cells to their fft host, then reduce the cells to the fft
 * array */
static void to_fft(double * cell, double * buf) {
#pragma omp atomic update
            cell[0] += buf[0];
}

static void
layout_build_and_exchange_cells_to_fft(
        PetaPM * pm,
        struct Layout * L,
        double * meshbuf,
        double * real)
{
    L->BufSend = (double *) mymalloc("PMBufSend", L->NcExport * sizeof(double));
    L->BufRecv = (double *) mymalloc("PMBufRecv", L->NcImport * sizeof(double));

    int i;
    int offset;

    /* collect all cells into the send buffer */
    offset = 0;
    for(i = 0; i < L->NpExport; i ++) {
        struct Pencil * p = &L->PencilSend[i];
        memcpy(L->BufSend + offset, &meshbuf[p->meshbuf_first],
                sizeof(double) * p->len);
        offset += p->len;
    }

    /* receive cells */
    MPI_Alltoallv(
            L->BufSend, L->NcSend, L->DcSend, MPI_DOUBLE,
            L->BufRecv, L->NcRecv, L->DcRecv, MPI_DOUBLE,
            L->comm);

#if 1
    double massExport = 0;
    for(i = 0; i < L->NcExport; i ++) {
        massExport += L->BufSend[i];
    }

    double massImport = 0;
    for(i = 0; i < L->NcImport; i ++) {
        massImport += L->BufRecv[i];
    }
    double totmassExport;
    double totmassImport;
    MPI_Allreduce(&massExport, &totmassExport, 1, MPI_DOUBLE, MPI_SUM, L->comm);
    MPI_Allreduce(&massImport, &totmassImport, 1, MPI_DOUBLE, MPI_SUM, L->comm);
    message(0, "totmassExport = %g totmassImport = %g\n", totmassExport, totmassImport);
#endif

    layout_iterate_cells(pm, L, to_fft, real);
    myfree(L->BufRecv);
    myfree(L->BufSend);
}

/* readout cells on their fft host, then exchange the cells to the domain
 * host */
static void to_region(double * cell, double * region) {
    *region = *cell;
}

static void
layout_build_and_exchange_cells_to_local(
        PetaPM * pm,
        struct Layout * L,
        double * meshbuf,
        double * real)
{
    L->BufRecv = (double *) mymalloc("PMBufRecv", L->NcImport * sizeof(double));
    int i;
    int offset;

    /*layout_iterate_cells transfers real to L->BufRecv*/
    layout_iterate_cells(pm, L, to_region, real);

    /*Real is done now: reuse the memory for BufSend*/
    myfree(real);
    /*Now allocate BufSend, which is confusingly used to receive data*/
    L->BufSend = (double *) mymalloc("PMBufSend", L->NcExport * sizeof(double));

    /* exchange cells */
    /* notice the order is reversed from to_fft */
    MPI_Alltoallv(
            L->BufRecv, L->NcRecv, L->DcRecv, MPI_DOUBLE,
            L->BufSend, L->NcSend, L->DcSend, MPI_DOUBLE,
            L->comm);

    /* distribute BufSend to meshbuf */
    offset = 0;
    for(i = 0; i < L->NpExport; i ++) {
        struct Pencil * p = &L->PencilSend[i];
        memcpy(&meshbuf[p->meshbuf_first],
                L->BufSend + offset,
                sizeof(double) * p->len);
        offset += p->len;
    }
    myfree(L->BufSend);
    myfree(L->BufRecv);
}

/* iterate over the pairs of real field cells and RecvBuf cells
 *
 * !!! iter has to be thread safe. !!!
 * */
static void
layout_iterate_cells(PetaPM * pm,
                     struct Layout * L,
                     cell_iterator iter,
                     double * real)
{
    int i;
    message(1, "******** NpImport %d \n", L->NpImport);
#pragma omp parallel for
    for(i = 0; i < L->NpImport; i ++) {
        struct Pencil * p = &L->PencilRecv[i];
        int k;
        ptrdiff_t linear0 = 0;
        for(k = 0; k < 2; k ++) {
            int ix = p->offset[k];
            while(ix < 0) ix += pm->Nmesh;
            while(ix >= pm->Nmesh) ix -= pm->Nmesh;
            ix -= pm->real_space_region.offset[k];
            if(ix >= pm->real_space_region.size[k]) {
                /* serious problem assumption about fft layout was wrong*/
                endrun(1, "bad fft: original k: %d ix: %d, cur ix: %d, region: off %ld size %ld\n", k, p->offset[k], ix, pm->real_space_region.offset[k], pm->real_space_region.size[k]);
            }
            linear0 += ix * pm->real_space_region.strides[k];
        }
        
        
        int j;
        for(j = 0; j < p->len; j ++) {
            int iz = p->offset[2] + j;
            while(iz < 0) iz += pm->Nmesh;
            while(iz >= pm->Nmesh) iz -= pm->Nmesh;
            if(iz >= pm->real_space_region.size[2]) {
                /* serious problem assmpution about fft layout was wrong*/
                endrun(1, "bad fft: original iz: %d, cur iz: %d, region: off %ld size %ld\n", p->offset[2], iz, pm->real_space_region.offset[2], pm->real_space_region.size[2]);
            }
            ptrdiff_t linear = iz * pm->real_space_region.strides[2] + linear0;
            /*
             * operate on the pencil, either modifying real or BufRecv
             * */
            iter(&real[linear], &L->BufRecv[p->first + j]);
        }
    }
}

static void
pm_init_regions(PetaPM * pm, PetaPMRegion * regions, const int Nregions)
{
    if(regions) {
        int i;
        size_t size = 0;
        for(i = 0 ; i < Nregions; i ++) {
            size += regions[i].totalsize;
        }
        pm->priv->meshbufsize = size;
        if ( size == 0 ) return;
        pm->priv->meshbuf = (double *) mymalloc("PMmesh", size * sizeof(double));
        /* this takes care of the padding */
        memset(pm->priv->meshbuf, 0, size * sizeof(double));
        size = 0;
        for(i = 0 ; i < Nregions; i ++) {
            regions[i].buffer = pm->priv->meshbuf + size;
            size += regions[i].totalsize;
        }
    }
}


static void
pm_iterate_one(PetaPM * pm,
               int i,
               pm_iterator iterator,
               PetaPMRegion * regions,
               const int Nregions)
{
    int k;
    int iCell[3];  /* integer coordinate on the regional mesh */
    double Res[3]; /* residual*/
    double * Pos = POS(i);
    const int RegionInd = CPS->RegionInd ? CPS->RegionInd[i] : 0;

    /* Asserts that the swallowed particles are not considered (region -2).*/
    if(RegionInd < 0)
        return;
    /* This should never happen: it is pure paranoia and to avoid icc being crazy*/
    if(RegionInd >= Nregions)
        endrun(1, "Particle %d has region %d out of bounds %d\n", i, RegionInd, Nregions);

    PetaPMRegion * region = &regions[RegionInd];
    for(k = 0; k < 3; k++) {
        double tmp = Pos[k] / pm->CellSize;
        iCell[k] = floor(tmp);
        Res[k] = tmp - iCell[k];
        iCell[k] -= region->offset[k];
        /* seriously?! particles are supposed to be contained in cells */
        if(iCell[k] >= region->size[k] - 1 || iCell[k] < 0) {
            endrun(1, "particle out of cell better stop %d (k=%d) %g %g %g region: %td %td\n", iCell[k],k,
                Pos[0], Pos[1], Pos[2],
                region->offset[k], region->size[k]);
        }
    }

    int connection;
    for(connection = 0; connection < 8; connection++) {
        double weight = 1.0;
        size_t linear = 0;
        for(k = 0; k < 3; k++) {
            int offset = (connection >> k) & 1;
            int tmp = iCell[k] + offset;
            linear += tmp * region->strides[k];
            weight *= offset?
                /* offset == 1*/ (Res[k])    :
                /* offset == 0*/ (1 - Res[k]);
        }
        if(linear >= region->totalsize) {
            endrun(1, "particle linear index out of cell better stop\n");
        }
        iterator(pm, i, &region->buffer[linear], weight);
    }
}

/*
 * iterate over all particle / mesh pairs, call iterator
 * function . iterator function shall be aware of thread safety.
 * no threads run on same particle same time but may
 * access one mesh points same time.
 * */
static void pm_iterate(PetaPM * pm, pm_iterator iterator, PetaPMRegion * regions, const int Nregions) {
    int i;
#pragma omp parallel for
    for(i = 0; i < CPS->NumPart; i ++) {
        pm_iterate_one(pm, i, iterator, regions, Nregions);
    }
}

void petapm_region_init_strides(PetaPMRegion * region) {
    int k;
    size_t rt = 1;
    for(k = 2; k >= 0; k --) {
        region->strides[k] = rt;
        rt = region->size[k] * rt;
    }
    region->totalsize = rt;
    region->buffer = NULL;
}


static int pos_get_target(PetaPM * pm, const int pos[2]) {
    int k;
    int task2d[2];
    int rank;
    for(k = 0; k < 2; k ++) {
        int ix = pos[k];
        while(ix < 0) ix += pm->Nmesh;
        while(ix >= pm->Nmesh) ix -= pm->Nmesh;
        task2d[k] = pm->Mesh2Task[k][ix];
    }
    MPI_Cart_rank(pm->priv->comm_cart_2d, task2d, &rank);
    return rank;
}
static int pencil_cmp_target(const void * v1, const void * v2) {
    const struct Pencil * p1 = (const struct Pencil *) v1;
    const struct Pencil * p2 = (const struct Pencil *) v2;
    /* move zero length pixels to the end */
    if(p2->len == 0) return -1;
    if(p1->len == 0) return 1;
    int t1 = p1->task;
    int t2 = p2->task;
    return ((t2 < t1) - (t1 < t2)) * 2 +
        ((p2->meshbuf_first < p1->meshbuf_first) - (p1->meshbuf_first < p2->meshbuf_first));
}




/********************************************************************************************/

#ifdef DEBUG
static void verify_density_field(PetaPM * pm, double * real, double * meshbuf, const size_t meshsize) {
    /* verify the density field */
    double mass_Part = 0;
    int j;
#pragma omp parallel for reduction(+: mass_Part)
    for(j = 0; j < CPS->NumPart; j ++) {
        double Mass = *MASS(j);
        mass_Part += Mass;
    }
    double totmass_Part = 0;
    MPI_Allreduce(&mass_Part, &totmass_Part, 1, MPI_DOUBLE, MPI_SUM, pm->comm);

    double mass_Region = 0;
    size_t i;

#pragma omp parallel for reduction(+: mass_Region)
    for(i = 0; i < meshsize; i ++) {
        mass_Region += meshbuf[i];
    }
    double totmass_Region = 0;
    MPI_Allreduce(&mass_Region, &totmass_Region, 1, MPI_DOUBLE, MPI_SUM, pm->comm);
    double mass_CIC = 0;
#pragma omp parallel for reduction(+: mass_CIC)
    for(i = 0; i < pm->real_space_region.totalsize; i ++) {
        mass_CIC += real[i];
    }
    double totmass_CIC = 0;
    MPI_Allreduce(&mass_CIC, &totmass_CIC, 1, MPI_DOUBLE, MPI_SUM, pm->comm);

    message(0, "total Region mass err = %g CIC mass err = %g Particle mass = %g\n", totmass_Region / totmass_Part - 1, totmass_CIC / totmass_Part - 1, totmass_Part);
}
#endif







/**************
 * functions iterating over particle / mesh pairs
 ***************/
 // can write to some other place and add up later
 // look for numpy reduce at/bin count
static void put_particle_to_mesh(PetaPM * pm, int i, double * mesh, double weight) {
    double Mass = *MASS(i);
    if(INACTIVE(i))
        return;
#pragma omp atomic update
    mesh[0] += weight * Mass;
}
static int64_t reduce_int64(int64_t input, MPI_Comm comm) {
    int64_t result = 0;
    MPI_Allreduce(&input, &result, 1, MPI_INT64, MPI_SUM, comm);
    return result;
}

/** Some FFT notes
 *
 *
 * CFT = dx * iDFT (thus CFT has no 2pi factors and iCFT has,
 *           same as wikipedia.)
 *
 * iCFT = dk * DFT
 * iCFT(CFG) = dx * dk * DFT(iDFT)
 *           = L / N * (2pi / L) * N
 *           = 2 pi
 * agreed with the usual def that
 * iCFT(CFT) = 2pi
 *
 * **************************8*/
