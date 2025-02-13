
#include <cuda_runtime.h>           // For CUDA runtime API functions.
#include <device_launch_parameters.h>  // To support device-related parameters.
// #include "treewalk.h"               // Include necessary header for TreeWalk structures and methods
#include "treewalk_kernel.h"
#include "gravshort.h"
// treewalk_kernel.cu
#include "shortrange-kernel_device.cu"
// #include "gravity.h"

#define NTAB_device (sizeof(shortrange_force_kernels) / sizeof(shortrange_force_kernels[0]))
/*! variables for short-range lookup table */
__device__ static float shortrange_table[NTAB_device], shortrange_table_potential[NTAB_device], shortrange_table_tidal[NTAB_device];

__device__ static double GravitySoftening_device = 0.0;

__global__ void gravshort_fill_ntab_device(const enum ShortRangeForceWindowType ShortRangeForceWindowType, const double Asmth) {
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= NTAB_device) return;  // Ensure we don't access out of bounds.

    double u = shortrange_force_kernels[i][0] * 0.5 / Asmth;

    switch (ShortRangeForceWindowType) {
        case SHORTRANGE_FORCE_WINDOW_TYPE_EXACT:
            if (Asmth != 1.5) {
                // printf("The short range force window is calibrated for Asmth = 1.5, but running with %g\n", Asmth);
            }
            shortrange_table[i] = shortrange_force_kernels[i][2];  // Use calibrated value.
            shortrange_table_potential[i] = shortrange_force_kernels[i][1];
            break;

        case SHORTRANGE_FORCE_WINDOW_TYPE_ERFC:
            shortrange_table[i] = erfc(u) + 2.0 * u / sqrt(M_PI) * exp(-u * u);
            shortrange_table_potential[i] = erfc(u);
            break;
    }
    shortrange_table_tidal[i] = 4.0 * u * u * u / sqrt(M_PI) * exp(-u * u);
}

void run_gravshort_fill_ntab(const enum ShortRangeForceWindowType ShortRangeForceWindowType, const double Asmth) {

    // Set up CUDA kernel launch parameters.
    int threadsPerBlock = 256;  // Number of threads per block.
    int blocks = (NTAB_device + threadsPerBlock - 1) / threadsPerBlock;  // Calculate the number of blocks needed.

    // Launch the kernel.
    gravshort_fill_ntab_device<<<blocks, threadsPerBlock>>>(ShortRangeForceWindowType, Asmth);

    // Synchronize to ensure kernel completion and check for errors.
    cudaDeviceSynchronize();
}

__device__ double FORCE_SOFTENING_device(void)
{
    // Return -1 to indicate an error if GravitySoftening_device is not set
    if (GravitySoftening_device == 0.0) {
        return -1.0; // error indicator
    }
    return 2.8 * GravitySoftening_device;
}

/* multiply force factor (*fac) and potential (*pot) by the shortrange force window function*/
__device__ int
grav_apply_short_range_window_device(double r, double * fac, double * pot, const double cellsize)
{
    const double dx = shortrange_force_kernels[1][0];
    double i = (r / cellsize / dx);
    size_t tabindex = floor(i);
    if(tabindex >= NTAB_device - 1)
        return 1;
    
    /* use a linear interpolation; */
    *fac *= (tabindex + 1 - i) * shortrange_table[tabindex] + (i - tabindex) * shortrange_table[tabindex + 1];
    *pot *= (tabindex + 1 - i) * shortrange_table_potential[tabindex] + (i - tabindex) * shortrange_table_potential[tabindex];
    return 0;
}

/* Add the acceleration from a node or particle to the output structure,
 * computing the short-range kernel and softening.*/
__device__ static void
apply_accn_to_output_device(void * output, const double dx[3], const double r2, const double mass, const double cellsize, int children = 0)
{
    const double r = sqrt(r2);

    const double h = FORCE_SOFTENING_device();
    double fac = mass / (r2 * r);
    double facpot = -mass / r;

    if(r2 < h*h)
    {
        double wp;
        const double h3_inv = 1.0 / h / h / h;
        const double u = r / h;
        if(u < 0.5) {
            fac = mass * h3_inv * (10.666666666667 + u * u * (32.0 * u - 38.4));
            wp = -2.8 + u * u * (5.333333333333 + u * u * (6.4 * u - 9.6));
        }
        else {
            fac =
                mass * h3_inv * (21.333333333333 - 48.0 * u +
                        38.4 * u * u - 10.666666666667 * u * u * u - 0.066666666667 / (u * u * u));
            wp =
                -3.2 + 0.066666666667 / u + u * u * (10.666666666667 +
                        u * (-16.0 + u * (9.6 - 2.133333333333 * u)));
        }
        facpot = mass / h * wp;
    }
    if(0 != grav_apply_short_range_window_device(r, &fac, &facpot, cellsize))
        return;

    int i;
    if (children) {
        TreeWalkResultChildren * result = (TreeWalkResultChildren *) output;
        for(i = 0; i < 3; i++)
            result->Acc[i] += dx[i] * fac;
        result->Potential += facpot;
    } else {
        TreeWalkResultGravShort * result = (TreeWalkResultGravShort *) output;
        for(i = 0; i < 3; i++)
            result->Acc[i] += dx[i] * fac;
        result->Potential += facpot;
    }
}

__device__ static int
shall_we_discard_node_device(const double len, const double r2, const double center[3], const double inpos[3], const double BoxSize, const double rcut, const double rcut2)
{
    /* This checks the distance from the node center of mass
     * is greater than the cutoff. */
    if(r2 > rcut2)
    {
        /* check whether we can stop walking along this branch */
        const double eff_dist = rcut + 0.5 * len;
        int i;
        /*This checks whether we are also outside this region of the oct-tree*/
        /* As long as one dimension is outside, we are fine*/
        for(i=0; i < 3; i++)
            if(fabs(NEAREST(center[i] - inpos[i], BoxSize)) > eff_dist)
                return 1;
    }
    return 0;
}

__device__ static int
shall_we_open_node_device(const double len, const double mass, const double r2, const double center[3], const double inpos[3], const double BoxSize, const double aold, const int TreeUseBH, const double BHOpeningAngle2)
{
    /* Check the relative acceleration opening condition*/
    if((TreeUseBH == 0) && (mass * len * len > r2 * r2 * aold))
         return 1;

    double bhangle = len * len  / r2;
     /*Check Barnes-Hut opening angle*/
    if(bhangle > BHOpeningAngle2)
         return 1;

    const double inside = 0.6 * len;
    /* Open the cell if we are inside it, even if the opening criterion is not satisfied.*/
    if(fabs(NEAREST(center[0] - inpos[0], BoxSize)) < inside &&
        fabs(NEAREST(center[1] - inpos[1], BoxSize)) < inside &&
        fabs(NEAREST(center[2] - inpos[2], BoxSize)) < inside)
        return 1;

    /* ok, node can be used */
    return 0;
}

__device__ void
treewalk_add_counters_device(LocalTreeWalk * lv, const int64_t ninteractions)
{
    if(lv->maxNinteractions < ninteractions)
        lv->maxNinteractions = ninteractions;
    if(lv->minNinteractions > ninteractions)
        lv->minNinteractions = ninteractions;
    lv->Ninteractions += ninteractions;
}

__device__ int treewalk_export_particle_device(LocalTreeWalk * lv, int no)
{
    // if(lv->mode != TREEWALK_TOPTREE || no < lv->tw->tree->lastnode) {
    //     endrun(1, "Called export not from a toptree.\n");
    // }
    // if(!lv->DataIndexTable)
    //     endrun(1, "DataIndexTable not allocated, treewalk_export_particle called in the wrong way\n");
    // if(no - lv->tw->tree->lastnode > lv->tw->tree->NTopLeaves)
    //     endrun(1, "Bad export leaf: no = %d lastnode %d ntop %d target %d\n", no, lv->tw->tree->lastnode, lv->tw->tree->NTopLeaves, lv->target);
    const int target = lv->target;
    TreeWalk * tw = lv->tw;
    const int task = tw->tree->TopLeaves[no - tw->tree->lastnode].Task;
    /* This index is a unique entry in the global DataIndexTable.*/
    size_t nexp = lv->Nexport;
    /* If the last export was to this task, we can perhaps just add this export to the existing NodeList. We can
     * be sure that all exports of this particle are contiguous.*/
    if(lv->NThisParticleExport >= 1 && lv->DataIndexTable[nexp-1].Task == task) {
#ifdef DEBUG
        /* This is just to be safe: only happens if our indices are off.*/
        if(lv->DataIndexTable[nexp - 1].Index != target)
            endrun(1, "Previous of %ld exports is target %d not current %d\n", lv->NThisParticleExport, lv->DataIndexTable[nexp-1].Index, target);
#endif
        if(lv->nodelistindex < NODELISTLENGTH) {
#ifdef DEBUG
            if(lv->DataIndexTable[nexp-1].NodeList[lv->nodelistindex] != -1)
                endrun(1, "Current nodelist %ld entry (%d) not empty!\n", lv->nodelistindex, lv->DataIndexTable[nexp-1].NodeList[lv->nodelistindex]);
#endif
            lv->DataIndexTable[nexp-1].NodeList[lv->nodelistindex] = tw->tree->TopLeaves[no - tw->tree->lastnode].treenode;
            lv->nodelistindex++;
            return 0;
        }
    }
    /* out of buffer space. Need to interrupt. */
    if(lv->Nexport >= tw->BunchSize) {
        return -1;
    }
    lv->DataIndexTable[nexp].Task = task;
    lv->DataIndexTable[nexp].Index = target;
    lv->DataIndexTable[nexp].NodeList[0] = tw->tree->TopLeaves[no - tw->tree->lastnode].treenode;
    int i;
    for(i = 1; i < NODELISTLENGTH; i++)
        lv->DataIndexTable[nexp].NodeList[i] = -1;
    lv->Nexport++;
    lv->nodelistindex = 1;
    lv->NThisParticleExport++;
    return 0;
}

__device__ int force_treeev_shortrange_device(TreeWalkQueryGravShort * input,
        TreeWalkResultGravShort * output,
        LocalTreeWalk * lv, const struct gravshort_tree_params * TreeParams_ptr, const particle_data * particles)
{
    const ForceTree * tree = lv->tw->tree;
    const double BoxSize = tree->BoxSize;

    /*Tree-opening constants*/
    const double cellsize = GRAV_GET_PRIV(lv->tw)->cellsize;
    const double rcut = GRAV_GET_PRIV(lv->tw)->Rcut;
    const double rcut2 = rcut * rcut;
    const double aold = TreeParams_ptr->ErrTolForceAcc * input->OldAcc;
    const int TreeUseBH = TreeParams_ptr->TreeUseBH;
    double BHOpeningAngle2 = TreeParams_ptr->BHOpeningAngle * TreeParams_ptr->BHOpeningAngle;
    /* Enforce a maximum opening angle even for relative acceleration criterion, to avoid
     * pathological cases. Default value is 0.9, from Volker Springel.*/
    if(TreeUseBH == 0)
        BHOpeningAngle2 = TreeParams_ptr->MaxBHOpeningAngle * TreeParams_ptr->MaxBHOpeningAngle;

    /*Input particle data*/
    const double * inpos = input->base.Pos;

    TreeWalkResultChildren output_children[1] = {{0}};

    /*Start the tree walk*/
    int listindex, ninteractions=0;

    /* Primary treewalk only ever has one nodelist entry*/
    for(listindex = 0; listindex < NODELISTLENGTH; listindex++)
    {
        int numcand = 0;
        /* Use the next node in the node list if we are doing a secondary walk.
         * For a primary walk the node list only ever contains one node. */
        int no = input->base.NodeList[listindex];
        int startno = no;
        if(no < 0)
            break;

        while(no >= 0)
        {
            /* The tree always walks internal nodes*/
            struct NODE *nop = &tree->Nodes[no];

            if(lv->mode == TREEWALK_GHOSTS && nop->f.TopLevel && no != startno)  /* we reached a top-level node again, which means that we are done with the branch */
                break;

            int i;
            double dx[3];
            for(i = 0; i < 3; i++)
                dx[i] = NEAREST(nop->mom.cofm[i] - inpos[i], BoxSize);
            const double r2 = dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2];

            /* Discard this node, move to sibling*/
            if(shall_we_discard_node_device(nop->len, r2, nop->center, inpos, BoxSize, rcut, rcut2))
            {
                no = nop->sibling;
                /* Don't add this node*/
                continue;
            }

            /* This node accelerates the particle directly, and is not opened.*/
            int open_node = shall_we_open_node_device(nop->len, nop->mom.mass, r2, nop->center, inpos, BoxSize, aold, TreeUseBH, BHOpeningAngle2);
            
            if(!open_node)
            {
                /* ok, node can be used */
                no = nop->sibling;
                if(lv->mode != TREEWALK_TOPTREE) {
                    /* Compute the acceleration and apply it to the output structure*/
                    apply_accn_to_output_device(output, dx, r2, nop->mom.mass, cellsize);
                }
                continue;
            }
            // ev_primary does not do anything about export/import
            if(lv->mode == TREEWALK_TOPTREE) {  
                if(nop->f.ChildType == PSEUDO_NODE_TYPE) {
                    /* Export the pseudo particle*/
                    if(-1 == treewalk_export_particle_device(lv, nop->s.suns[0]))
                        return -1;
                    /* Move sideways*/
                    no = nop->sibling;
                    continue;
                }
                /* Only walk toptree nodes here*/
                if(nop->f.TopLevel && !nop->f.InternalTopLevel) {
                    no = nop->sibling;
                    continue;
                }
                no = nop->s.suns[0];
            }
            else {
                /* Now we have a cell that needs to be opened.
                * If it contains particles we can add them directly here */
                if(nop->f.ChildType == PARTICLE_NODE_TYPE)
                {
                    /* Loop over child particles*/
                    for(i = 0; i < nop->s.noccupied; i++) {
                        int pp = nop->s.suns[i];
                        // lv->ngblist[numcand++] = pp;
                        numcand++;
                        double dx[3];
                        int j;
                        for(j = 0; j < 3; j++)
                            dx[j] = NEAREST(particles[pp].Pos[j] - inpos[j], BoxSize);
                        const double r2 = dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2];
                        /* Compute the acceleration and apply it to the output structure*/
                        apply_accn_to_output_device(output_children, dx, r2, particles[pp].Mass, cellsize, 1);
                    }
                    no = nop->sibling;
                }
                else if (nop->f.ChildType == PSEUDO_NODE_TYPE)
                {
                    /* Move to the sibling (likely also a pseudo node)*/
                    no = nop->sibling;
                }
                else //NODE_NODE_TYPE
                    /* This node contains other nodes and we need to open it.*/
                    no = nop->s.suns[0];
            }
        }
        ninteractions = numcand;
    }
    for (int i = 0; i < 3; i++)
        output->Acc[i] += output_children->Acc[i];
    output->Potential += output_children->Potential;

    treewalk_add_counters_device(lv, ninteractions);
    return 1;
}

__device__ static MyFloat
grav_get_abs_accel_device(struct particle_data * PP, const double G)
{
    double aold=0;
    int j;
    for(j = 0; j < 3; j++) {
       double ax = PP->FullTreeGravAccel[j] + PP->GravPM[j];
       aold += ax*ax;
    }
    return sqrt(aold) / G;
}

__device__ static void
grav_short_copy_device(int place, TreeWalkQueryGravShort * input, TreeWalk * tw, struct particle_data *particles)
{
    input->OldAcc = grav_get_abs_accel_device(&particles[place], GRAV_GET_PRIV(tw)->G);
}

__device__ static void
treewalk_init_query_device(TreeWalk *tw, TreeWalkQueryGravShort *query_short, int i, const int *NodeList, struct particle_data *particles) {
    // Access particle data through particles argument
    for(int d = 0; d < 3; d++) {
        query_short->base.Pos[d] = particles[i].Pos[d];  // Use particles instead of P macro
    }

    if (NodeList) {
        memcpy(query_short->base.NodeList, NodeList, sizeof(query_short->base.NodeList[0]) * NODELISTLENGTH);
    } else {
        query_short->base.NodeList[0] = tw->tree->firstnode;  // root node
        query_short->base.NodeList[1] = -1;  // terminate immediately
    }

    grav_short_copy_device(i, query_short, tw, particles);
}

__device__ static void
treewalk_init_query_device_old(TreeWalk *tw, TreeWalkQueryBase *query, int i, const int *NodeList, struct particle_data *particles) {
    // Access particle data through particles argument
    for(int d = 0; d < 3; d++) {
        query->Pos[d] = particles[i].Pos[d];  // Use particles instead of P macro
    }

    if (NodeList) {
        memcpy(query->NodeList, NodeList, sizeof(query->NodeList[0]) * NODELISTLENGTH);
    } else {
        query->NodeList[0] = tw->tree->firstnode;  // root node
        query->NodeList[1] = -1;  // terminate immediately
    }
    TreeWalkQueryGravShort * query_short;
    // point query_short to the query
    query_short = (TreeWalkQueryGravShort *) query;
    // tw->fill(i, query, tw);
    grav_short_copy_device(i, query_short, tw, particles);
}
__device__ static void
treewalk_init_result_device(TreeWalk *tw, TreeWalkResultGravShort *result_short, TreeWalkQueryGravShort *query_short) {
    memset(result_short, 0, tw->result_type_elsize);  // Initialize the result structure
}

__device__ static void
grav_short_reduce_device(int place, TreeWalkResultGravShort * result, enum TreeWalkReduceMode mode, TreeWalk * tw, struct particle_data *particles)
{
    TREEWALK_REDUCE(GRAV_GET_PRIV(tw)->Accel[place][0], result->Acc[0]);
    TREEWALK_REDUCE(GRAV_GET_PRIV(tw)->Accel[place][1], result->Acc[1]);
    TREEWALK_REDUCE(GRAV_GET_PRIV(tw)->Accel[place][2], result->Acc[2]);
    if(tw->tree->full_particle_tree_flag)
        TREEWALK_REDUCE(particles[place].Potential, result->Potential);
}

__device__ void
treewalk_reduce_result_device(TreeWalk *tw, TreeWalkResultGravShort * result, int i, enum TreeWalkReduceMode mode, struct particle_data *particles) {
    // if (tw->reduce != NULL) {
    //     tw->reduce(i, result, mode, tw);  // Call the reduce function
    // }
    grav_short_reduce_device(i, (TreeWalkResultGravShort *) result, mode, tw, particles);
}

__device__ static void
ev_init_thread_device(TreeWalk * const tw, LocalTreeWalk * lv)
{
    // Use the CUDA thread index instead of omp_get_thread_num
    const size_t thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    const size_t Nthreads = tw->NThread;
    
    lv->tw = tw;
    lv->maxNinteractions = 0;
    lv->minNinteractions = 1L << 45;
    lv->Ninteractions = 0;
    lv->Nexport = 0;
    lv->NThisParticleExport = 0;
    lv->nodelistindex = 0;

    // Assign the correct DataIndexTable for each thread
    if (tw->ExportTable_thread && thread_id < Nthreads)
        lv->DataIndexTable = tw->ExportTable_thread[thread_id];
    else
        lv->DataIndexTable = NULL;

    // Assign ngblist specific to each thread, adapted to GPU thread indexing
    // let us process neighbers one by one with immediate accn calculation so without using a list (then we won't need much memory for ngblist)
    // if (tw->Ngblist)
    //     lv->ngblist = tw->Ngblist + thread_id * tw->tree->NumParticles;
}

__global__ void treewalk_kernel(TreeWalk *tw, struct particle_data *particles, const struct gravshort_tree_params * TreeParams_ptr, unsigned long long int *maxNinteractions, unsigned long long int *minNinteractions, unsigned long long int *Ninteractions, const double GravitySoftening) {
    GravitySoftening_device = GravitySoftening;

    // Use a direct instance rather than an array
    LocalTreeWalk lv;
    ev_init_thread_device(tw, &lv);
    lv.mode = TREEWALK_PRIMARY;

    // Avoid stack-heavy allocations, be mindful of per-thread memory usage
    TreeWalkQueryGravShort input;
    TreeWalkResultGravShort output;

    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < tw->WorkSetSize) {
        const int64_t i = tw->WorkSet ? (int64_t) tw->WorkSet[tid] : tid;

        // Initialize query and result using device functions
        treewalk_init_query_device(tw, &input, i, NULL, particles);
        treewalk_init_result_device(tw, &output, &input);

        // Perform treewalk for particle
        lv.target = i;
        force_treeev_shortrange_device(&input, &output, &lv, TreeParams_ptr, particles);

        // Reduce results for this particle
        treewalk_reduce_result_device(tw, &output, i, TREEWALK_PRIMARY, particles);

        // Update interactions count using atomic operations
        // in the gpu case here, lv.Ninteractions, lv.maxNinteractions, lv.minNinteractions should all be equal (each thread exactly corresponds to one particle)
        atomicAdd(Ninteractions, lv.Ninteractions);
        atomicMax(maxNinteractions, lv.maxNinteractions);
        atomicMin(minNinteractions, lv.minNinteractions);
    }
}

__global__ void test_kernel_1(TreeWalk *tw, struct particle_data *particles, struct gravshort_tree_params * TreeParams_ptr, unsigned long long int *maxNinteractions, unsigned long long int *minNinteractions, unsigned long long int *Ninteractions, const double GravitySoftening){
    // access shortrange_table test from 0 to NTAB_device
    for (int i = 0; i < NTAB_device; i++) {
        printf("shortrange_table[%d]: %f\n", i, shortrange_table[i]);
    }
}

// Function to launch kernel (wrapper)
void run_treewalk_kernel(TreeWalk *tw, struct particle_data *particles, const struct gravshort_tree_params * TreeParams_ptr, const double GravitySoftening, unsigned long long int *maxNinteractions, unsigned long long int *minNinteractions, unsigned long long int *Ninteractions) {
    // workset is NULL at a PM step
    int threadsPerBlock = 256;
    int blocks = (tw->WorkSetSize + threadsPerBlock - 1) / threadsPerBlock;
    // treewalk_kernel<<<blocks, threadsPerBlock>>>(tw, particles, TreeParams_ptr, maxNinteractions, minNinteractions, Ninteractions, GravitySoftening);
    treewalk_kernel<<<blocks, threadsPerBlock>>>(tw, particles, TreeParams_ptr, maxNinteractions, minNinteractions, Ninteractions, GravitySoftening);
    // kernel_test_fac<<<blocks, threadsPerBlock>>>();
    cudaDeviceSynchronize();
    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     message(0, "CUDA error: %s\n", cudaGetErrorString(err));
    // }
}

__global__ void treewalk_secondary_kernel(TreeWalk *tw, struct particle_data *particles, const struct gravshort_tree_params * TreeParams_ptr, char* databufstart, char* dataresultstart, const int64_t nimports_task) {

    // Use a direct instance rather than an array
    LocalTreeWalk lv;
    ev_init_thread_device(tw, &lv);
    lv.mode = TREEWALK_GHOSTS;

    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < nimports_task) {

        TreeWalkQueryGravShort * input = (TreeWalkQueryGravShort *) (databufstart + tid * tw->query_type_elsize);
        TreeWalkResultGravShort * output = (TreeWalkResultGravShort *) (dataresultstart + tid * tw->result_type_elsize);

        // Initialize query and result using device functions
        // treewalk_init_query_device(tw, &input, i, NULL, particles);
        treewalk_init_result_device(tw, output, input);

        // Perform treewalk for particle
        lv.target = -1;
        force_treeev_shortrange_device(input, output, &lv, TreeParams_ptr, particles);

    }
}

void run_treewalk_secondary_kernel(TreeWalk *tw, struct particle_data *particles, const struct gravshort_tree_params * TreeParams_ptr, char* databufstart, char* dataresultstart, const int64_t nimports_task) {
    // workset is NULL at a PM step
    int threadsPerBlock = 256;
    int blocks = (nimports_task + threadsPerBlock - 1) / threadsPerBlock;
    treewalk_secondary_kernel<<<blocks, threadsPerBlock>>>(tw, particles, TreeParams_ptr, databufstart, dataresultstart, nimports_task);
    cudaDeviceSynchronize();
}
