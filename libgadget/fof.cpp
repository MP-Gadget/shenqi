#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <inttypes.h>
#include <execution>
#include <omp.h>

#include "types.h"
#include "utils/endrun.h"
#include "utils/mpsort.h"

#include "walltime.h"
#include "blackhole.h"
#include "domain.h"
#include "winds.h"
#include "petaio.h"

#include "slotsmanager.h"
#include "partmanager.h"
#include "densitykernel.h"
#include "utils/mymalloc.h"

/*! \file fof.c
 *  \brief parallel FoF group finder
 */

#include "fof.h"
#include "fof2.h"


struct FOFParams
{
    int FOFSaveParticles ; /* saving particles in the fof group */
    double MinFoFMassForNewSeed;	/* Halo mass required before new seed is put in */
    double MinMStarForNewSeed; /* Minimum stellar mass required before new seed */
    double FOFHaloLinkingLength;
    double FOFHaloComovingLinkingLength; /* in code units */
    int FOFHaloMinLength;
    int FOFPrimaryLinkTypes;
    int FOFSecondaryLinkTypes;
    int ExcursionSetReionOn;
} fof_params;

/*Set the parameters of the BH module*/
void set_fof_params(ParameterSet * ps)
{
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    if(ThisTask == 0) {
        fof_params.FOFSaveParticles = param_get_int(ps, "FOFSaveParticles");
        fof_params.FOFHaloLinkingLength = param_get_double(ps, "FOFHaloLinkingLength");
        fof_params.FOFHaloMinLength = param_get_int(ps, "FOFHaloMinLength");
        fof_params.MinFoFMassForNewSeed = param_get_double(ps, "MinFoFMassForNewSeed");
        fof_params.MinMStarForNewSeed = param_get_double(ps, "MinMStarForNewSeed");
        fof_params.FOFPrimaryLinkTypes = param_get_int(ps, "FOFPrimaryLinkTypes");
        fof_params.FOFSecondaryLinkTypes = param_get_int(ps, "FOFSecondaryLinkTypes");
        fof_params.ExcursionSetReionOn = param_get_int(ps, "ExcursionSetReionOn");
    }
    MPI_Bcast(&fof_params, sizeof(struct FOFParams), MPI_BYTE, 0, MPI_COMM_WORLD);
}

/* Set parameters for the tests*/
void set_fof_testpar(int FOFSaveParticles, double FOFHaloLinkingLength, int FOFHaloMinLength)
{
    fof_params.FOFSaveParticles = FOFSaveParticles;
    fof_params.FOFPrimaryLinkTypes = 2;
    fof_params.FOFSecondaryLinkTypes = 1+16+32;
    fof_params.FOFHaloLinkingLength = FOFHaloLinkingLength;
    fof_params.FOFHaloMinLength = FOFHaloMinLength;
    /* For seeding (not yet tested)*/
    fof_params.MinFoFMassForNewSeed = 2;
    fof_params.MinMStarForNewSeed = 5e-4;

}

void fof_init(double DMMeanSeparation)
{
    fof_params.FOFHaloComovingLinkingLength = fof_params.FOFHaloLinkingLength * DMMeanSeparation;
}

static double fof_periodic_wrap(double x, double BoxSize)
{
    while(x >= BoxSize)
        x -= BoxSize;
    while(x < 0)
        x += BoxSize;
    return x;
}

template <typename GroupType> void fof_reduce_groups(GroupType * groups, int nmemb, MPI_Comm Comm);


static void fof_finish_group_properties(FOFGroups * fof, double BoxSize);

static int fof_compile_base(struct BaseGroup * base, int NgroupsExt, struct fof_particle_list * HaloLabel, MPI_Comm Comm);
static void fof_compile_catalogue(FOFGroups * fof, const int NgroupsExt, struct fof_particle_list * HaloLabel, MPI_Comm Comm);

static struct Group *
fof_alloc_group(const struct BaseGroup * base, const int NgroupsExt);

static void fof_assign_grnr(struct BaseGroup * base, const int NgroupsExt, MPI_Comm Comm);


static MPI_Datatype MPI_TYPE_GROUP;

/*
 * The FOF finder will produce Group[], which is allocated to the top side of the
 * main heap.
 *
 **/

FOFGroups
fof_fof(DomainDecomp * ddecomp, const int StoreGrNr, MPI_Comm Comm)
{
    int64_t i;

    message(0, "Begin to compute FoF group catalogues. (allocated: %g MB)\n",
            mymalloc_usedbytes() / (1024.0 * 1024.0));

    message(0, "Comoving linking length: %g\n", fof_params.FOFHaloComovingLinkingLength);

    struct fof_particle_list * HaloLabel = mymalloc("HaloLabel", fof_particle_list, PartManager->NumPart);

    /* HaloLabel stores the MinID and MinIDTask of particles, this pair serves as a halo label. */
    #pragma omp parallel for
    for(i = 0; i < PartManager->NumPart; i++) {
        HaloLabel[i].Pindex = i;
    }

    /* Find the halo label of each particle with the treewalks: first link the primary particles into groups,
     * then attach the secondary particles to the nearest primary particle.*/
    fof_label_particles(HaloLabel, ddecomp, fof_params.FOFHaloComovingLinkingLength, fof_params.FOFPrimaryLinkTypes, fof_params.FOFSecondaryLinkTypes, Comm);

    /* sort HaloLabel according to MinID, because we need that for compiling catalogues */
    std::sort(std::execution::par_unseq, HaloLabel, HaloLabel+PartManager->NumPart);

    int NgroupsExt = 1;

    for(i = 1; i < PartManager->NumPart; i ++) {
        if(HaloLabel[i].MinID != HaloLabel[i - 1].MinID)
            NgroupsExt ++;
    }

    /* The first round is to eliminate groups that are too short. */
    /* We create the smaller 'BaseGroup' data set for this. */
    struct BaseGroup * base = mymalloc("BaseGroup", BaseGroup, NgroupsExt);

    NgroupsExt = fof_compile_base(base, NgroupsExt, HaloLabel, Comm);

    message(0, "Compiled local group data and catalogue.\n");

    fof_assign_grnr(base, NgroupsExt, Comm);

    /*Store the group number in the particle struct*/
    if(StoreGrNr) {
        #pragma omp parallel for
        for(i = 0; i < PartManager->NumPart; i++)
            Part[i].GrNr = -1;	/* will mark particles that are not in any group */

        int64_t start = 0;
        for(i = 0; i < NgroupsExt; i++)
        {
            for(;start < PartManager->NumPart; start++) {
                if (HaloLabel[start].MinID >= base[i].MinID)
                    break;
            }

            for(;start < PartManager->NumPart; start++) {
                if (HaloLabel[start].MinID != base[i].MinID)
                    break;
                Part[HaloLabel[start].Pindex].GrNr = base[i].GrNr;
            }
        }
    }

    /*Initialise the Group object from the BaseGroup*/
    FOFGroups fof;
    MPI_Type_contiguous(sizeof(fof.Group[0]), MPI_BYTE, &MPI_TYPE_GROUP);
    MPI_Type_commit(&MPI_TYPE_GROUP);

    fof.Group = fof_alloc_group(base, NgroupsExt);

    myfree(base);

    fof_compile_catalogue(&fof, NgroupsExt, HaloLabel, Comm);

    MPI_Barrier(Comm);
    message(0, "Finished FoF. Group properties are now allocated.. (presently allocated=%g MB)\n",
            mymalloc_usedbytes() / (1024.0 * 1024.0));

    walltime_measure("/FOF/Compile");

    myfree(HaloLabel);

    return fof;
}

void
fof_finish(FOFGroups * fof)
{
    myfree(fof->Group);

    message(0, "Finished computing FoF groups.  (presently allocated=%g MB)\n",
            mymalloc_usedbytes() / (1024.0 * 1024.0));

    MPI_Type_free(&MPI_TYPE_GROUP);
}

static void add_particle_to_group(struct Group * gdst, int i, int ThisTask) {

    /* My local number of particles contributing to the full catalogue. */
    const int index = i;
    if(gdst->Length == 0) {
        struct BaseGroup base = gdst->base;
        memset(gdst, 0, sizeof(gdst[0]));
        gdst->base = base;
        gdst->seed_index = gdst->seed_task = -1;
    }

    gdst->Length ++;
    gdst->Mass += Part[index].Mass;
    gdst->LenType[Part[index].Type]++;
    gdst->MassType[Part[index].Type] += Part[index].Mass;

    if(Part[index].Type == 0) {
        gdst->MassHeIonized += Part[index].Mass * Part[index].HeIIIionized;
        gdst->Sfr += SPHP(index).Sfr;
        gdst->GasMetalMass += SPHP(index).Metallicity * Part[index].Mass;
        int j;
        for(j = 0; j < NMETALS; j++)
            gdst->GasMetalElemMass[j] += SPHP(index).Metals[j] * Part[index].Mass;
    }
    if(Part[index].Type == 4) {
        int j;
        gdst->StellarMetalMass += STARP(index).Metallicity * Part[index].Mass;
        for(j = 0; j < NMETALS; j++)
            gdst->StellarMetalElemMass[j] += STARP(index).Metals[j] * Part[index].Mass;
    }

    if(Part[index].Type == 5)
    {
        gdst->BH_Mdot += BHP(index).Mdot;
        gdst->BH_Mass += BHP(index).Mass;
    }
    /*This used to depend on black holes being enabled, but I do not see why.
     * I think because it is only useful for seeding*/
    /* Don't make bh in wind.*/
    if(Part[index].Type == 0 && !winds_is_particle_decoupled(index))
        if(SPHP(index).Density > gdst->MaxDens)
        {
            gdst->MaxDens = SPHP(index).Density;
            gdst->seed_index = index;
            gdst->seed_task = ThisTask;
        }

    int d1, d2;
    double xyz[3];
    double rel[3];
    double vel[3];
    double jmom[3];

    for(d1 = 0; d1 < 3; d1++)
    {
        double first = gdst->base.FirstPos[d1];
        rel[d1] = NEAREST(Part[index].Pos[d1] - first, PartManager->BoxSize) ;
        xyz[d1] = rel[d1] + first;
        vel[d1] = Part[index].Vel[d1];
    }

    crossproduct(rel, vel, jmom);

    for(d1 = 0; d1 < 3; d1++) {
        gdst->CM[d1] += Part[index].Mass * xyz[d1];
        gdst->Vel[d1] += Part[index].Mass * vel[d1];
        gdst->Jmom[d1] += Part[index].Mass * jmom[d1];

        for(d2 = 0; d2 < 3; d2++) {
            gdst->Imom[d1][d2] += Part[index].Mass * rel[d1] * rel[d2];
        }
    }
}

static void
fof_finish_group_properties(struct FOFGroups * fof, double BoxSize)
{
    int i;

    for(i = 0; i < fof->Ngroups; i++)
    {
        int d1, d2;
        double cm[3];
        double rel[3];
        double jcm[3];
        double vcm[3];

        struct Group * gdst = &fof->Group[i];
        for(d1 = 0; d1 < 3; d1++)
        {
            gdst->Vel[d1] /= gdst->Mass;
            vcm[d1] = gdst->Vel[d1];
            cm[d1] = gdst->CM[d1] / gdst->Mass;

            rel[d1] = NEAREST(cm[d1] - gdst->base.FirstPos[d1], BoxSize);

            cm[d1] = fof_periodic_wrap(cm[d1], BoxSize);
            gdst->CM[d1] = cm[d1];

        }
        crossproduct(rel, vcm, jcm);

        for(d1 = 0; d1 < 3; d1 ++) {
            gdst->Jmom[d1] -= jcm[d1] * gdst->Mass;
        }

        for(d1 = 0; d1 < 3; d1 ++) {
            for(d2 = 0; d2 < 3; d2++) {
                /* Parallel Axis theorem:
                 * https://en.wikipedia.org/wiki/Parallel_axis_theorem ;
                 * J was relative to FirstPos, I is relative to CM.
                 *
                 * Note that our definition of Imom follows the astronomy one,
                 *
                 * I_ij = sum x_i x_j (where x_i x_j is relative displacement)
                 * */

                double diff = rel[d1] * rel[d2];

                gdst->Imom[d1][d2] -= gdst->Mass * diff;
            }
        }
    }

}

static int
fof_compile_base(struct BaseGroup * base, int NgroupsExt, struct fof_particle_list * HaloLabel, MPI_Comm Comm)
{
    memset(base, 0, sizeof(base[0]) * NgroupsExt);

    int64_t i;
    int64_t start;

    start = 0;
    for(i = 0; i < PartManager->NumPart; i++)
    {
        if(i == 0 || HaloLabel[i].MinID != HaloLabel[i - 1].MinID) {
            base[start].MinID = HaloLabel[i].MinID;
            base[start].MinIDTask = HaloLabel[i].MinIDTask;
            int d;
            for(d = 0; d < 3; d ++) {
                base[start].FirstPos[d] = Part[HaloLabel[i].Pindex].Pos[d];
            }
            start ++;
        }
    }

    /* count local lengths */
    /* This works because base is sorted by MinID by construction. */
    start = 0;
    for(i = 0; i < NgroupsExt; i++)
    {
        /* find the first particle */
        for(;start < PartManager->NumPart; start++) {
            if(HaloLabel[start].MinID >= base[i].MinID) break;
        }
        /* count particles */
        for(;start < PartManager->NumPart; start++) {
            if(HaloLabel[start].MinID != base[i].MinID) {
                break;
            }
            base[i].Length ++;
        }
    }

    /* update global attributes */
    fof_reduce_groups<BaseGroup>(base, NgroupsExt, Comm);

    /* eliminate all groups that are too small */
    for(i = 0; i < NgroupsExt; i++)
    {
        if(base[i].Length < fof_params.FOFHaloMinLength)
        {
            base[i] = base[NgroupsExt - 1];
            NgroupsExt--;
            i--;
        }
    }
    return NgroupsExt;
}

/* Allocate memory for and initialise a Group object
 * from a BaseGroup object.*/
static struct Group *
fof_alloc_group(const struct BaseGroup * base, const int NgroupsExt)
{
    int i;
    struct Group * group = mymalloc2("Group", Group, NgroupsExt);
    memset(group, 0, sizeof(group[0]) * NgroupsExt);

    /* copy in the base properties */
    /* at this point base group shall be sorted by MinID */
    #pragma omp parallel for
    for(i = 0; i < NgroupsExt; i ++) {
        group[i].base = base[i];
    }
    return group;
}

/* TODO: It would be a good idea to generalise this to arbitrary fof/particle properties */
#ifdef EXCUR_REION
static void fof_set_escapefraction(struct FOFGroups * fof, const int NgroupsExt, struct fof_particle_list * HaloLabel)
{
    int64_t i = 0;
    #pragma omp parallel for
    for(i = 0; i < PartManager->NumPart; i++){
        if(Part[i].Type == 0){
            SPHP(i).EscapeFraction = 0.;
        }
        if(Part[i].Type == 4){
            STARP(i).EscapeFraction = 0.;	/* will mark particles that are not in any group */
        }
    }

    int64_t start = 0;
    for(i = 0; i < NgroupsExt; i++)
    {
        /* find the first particle */
        for(;start < PartManager->NumPart; start++) {
            if(HaloLabel[start].MinID >= fof->Group[i].base.MinID) break;
        }
        /* add particles */
        for(;start < PartManager->NumPart; start++) {
            if(HaloLabel[start].MinID != fof->Group[i].base.MinID) {
                break;
            }
            int pi = HaloLabel[start].Pindex;

            /* putting halo mass in escape fraction for now, converted before uvbg calculation */
            //TODO: switch this off for gas particles if we are smoothing the star formation rate
            if(Part[pi].Type == 0){
                SPHP(pi).EscapeFraction = fof->Group[i].Mass;
            }
            else if(Part[pi].Type == 4){
                STARP(pi).EscapeFraction = fof->Group[i].Mass;
            }
        }
    }
}
#endif

static void
fof_compile_catalogue(struct FOFGroups * fof, const int NgroupsExt, struct fof_particle_list * HaloLabel, MPI_Comm Comm)
{
    int ThisTask;
    int64_t i, start;

    MPI_Comm_rank(Comm, &ThisTask);

    start = 0;
    for(i = 0; i < NgroupsExt; i++)
    {
        /* find the first particle */
        for(;start < PartManager->NumPart; start++) {
            if(HaloLabel[start].MinID >= fof->Group[i].base.MinID) break;
        }
        /* add particles */
        for(;start < PartManager->NumPart; start++) {
            if(HaloLabel[start].MinID != fof->Group[i].base.MinID) {
                break;
            }
            add_particle_to_group(&fof->Group[i], HaloLabel[start].Pindex, ThisTask);
        }
    }

    /* collect global properties */
    fof_reduce_groups<Group>(fof->Group, NgroupsExt, Comm);

    /* count Groups and number of particles hosted by me */
    fof->Ngroups = 0;
    int64_t Nids = 0;
    for(i = 0; i < NgroupsExt; i ++) {
        if(fof->Group[i].base.MinIDTask != ThisTask) continue;

        fof->Ngroups++;
        Nids += fof->Group[i].base.Length;

        if(fof->Group[i].base.Length != fof->Group[i].Length) {
            /* These two shall be consistent */
            endrun(3333, "i=%ld Group base Length %d != Group Length %d\n", i, fof->Group[i].base.Length, fof->Group[i].Length);
        }
    }

    fof_finish_group_properties(fof, PartManager->BoxSize);
#ifdef EXCUR_REION
    /* feed group property back to each particle. */
    if(fof_params.ExcursionSetReionOn)
        fof_set_escapefraction(fof, NgroupsExt, HaloLabel);
#endif
    int64_t TotNids;
    MPI_Allreduce(&fof->Ngroups, &fof->TotNgroups, 1, MPI_INT64, MPI_SUM, Comm);
    MPI_Allreduce(&Nids, &TotNids, 1, MPI_INT64, MPI_SUM, Comm);

    /* report some statistics */
    int largestloc_tot = 0;
    double largestmass_tot= 0;
    if(fof->TotNgroups > 0)
    {
        double largestmass = 0;
        int largestlength = 0;

        for(i = 0; i < NgroupsExt; i++)
            if(fof->Group[i].Length > largestlength) {
                largestlength = fof->Group[i].Length;
                largestmass = fof->Group[i].Mass;
            }
        MPI_Allreduce(&largestlength, &largestloc_tot, 1, MPI_INT, MPI_MAX, Comm);
        MPI_Allreduce(&largestmass, &largestmass_tot, 1, MPI_DOUBLE, MPI_MAX, Comm);
    }

    message(0, "Total number of groups with at least %d particles: %ld\n", fof_params.FOFHaloMinLength, fof->TotNgroups);
    if(fof->TotNgroups > 0)
    {
        message(0, "Largest group has %d particles, mass %g.\n", largestloc_tot, largestmass_tot);
        message(0, "Total number of particles in groups: %012ld\n", TotNids);
    }
}

template <typename GroupType>
void fof_reduce_groups(GroupType * groups, int nmemb, MPI_Comm Comm)
{
    int NTask, ThisTask;
    MPI_Comm_size(Comm, &NTask);
    MPI_Comm_rank(Comm, &ThisTask);
    /* slangs:
     *   prime: groups hosted by ThisTask
     *   ghosts: groups that spans into ThisTask but not hosted by ThisTask;
     *           part of the local catalogue
     *   images: ghosts that are sent from another rank.
     *           images are reduced to prime, then the prime attributes
     *           are copied to images, and sent back to the ghosts.
     *
     *   in the beginning, prime and ghosts contains local group attributes.
     *   in the end, prime and ghosts all contain full group attributes.
     **/
    int * Send_count = ta_malloc("Send_count", int, NTask);
    int * Recv_count = ta_malloc("Recv_count", int, NTask);

    MPI_Datatype dtype;

    MPI_Type_contiguous(sizeof(GroupType), MPI_BYTE, &dtype);
    MPI_Type_commit(&dtype);

    /* local groups will be moved to the beginning, we skip them with offset */
    std::sort(std::execution::par_unseq, groups, groups + nmemb, [ThisTask](const GroupType& a, const GroupType& b) {
        int t1 = a.get_base()->MinIDTask;
        int t2 = b.get_base()->MinIDTask;
        if(t1 == ThisTask) t1 = -1;
        if(t2 == ThisTask) t2 = -1;
        return t1 < t2;
    });
    /* count how many we have of each task */
    memset(Send_count, 0, sizeof(int) * NTask);

    for(int i = 0; i < nmemb; i++) {
        const BaseGroup * base = groups[i].get_base();
        Send_count[base->MinIDTask]++;
    }

    /* Skip local groups */
    int Nmine = Send_count[ThisTask];
    Send_count[ThisTask] = 0;

    MPI_Alltoall(Send_count, 1, MPI_INT, Recv_count, 1, MPI_INT, Comm);

    int nimport = 0;
    for(int i = 0; i < NTask; i ++) {
        nimport += Recv_count[i];
    }

    GroupType * images = mymalloc("images", GroupType, nimport);

    /* Fill out local copies of the groups from other ranks*/
    MPI_Alltoallv_smart(groups + Nmine, Send_count, NULL, dtype,
                        images, Recv_count, NULL, dtype, Comm);

    for(int i = 0; i < nimport; i++) {
        reinterpret_cast<BaseGroup *>(&images[i])->OriginalIndex = i;
    }

    /* sort the images according to MinID so equal_range can search them */
    std::sort(std::execution::par_unseq, groups, groups + Nmine, [](const GroupType& a, const GroupType& b) {
        return a.get_base()->MinID < b.get_base()->MinID;
    });
    std::sort(std::execution::par_unseq, images, images + nimport, [](const GroupType& a, const GroupType& b) {
        return a.get_base()->MinID < b.get_base()->MinID;
    });

    /* merge the imported ones with the local ones */
    int start = 0;
    for(int i = 0; i < Nmine; i++) {
        const BaseGroup * prime = groups[i].get_base();
        for(;start < nimport; start++) {
            const BaseGroup * image = images[start].get_base();
            if(image->MinID >= prime->MinID)
                break;
        }
        int imagebegin = start;
        for(;start < nimport; start++) {
            const BaseGroup * image = images[start].get_base();
            if(image->MinID != prime->MinID)
                break;
            /* Note the type here should be GroupType*/
            groups[i].reduce(images[start]);
        }
        int imageend = start;
        /* update the images, such that they can be send back to the original task*/
        for(start = imagebegin; start < imageend; start++) {
            const BaseGroup * image = images[start].get_base();
            int save = image->OriginalIndex;
            memcpy(&images[start], &groups[i], sizeof(GroupType));
            reinterpret_cast<BaseGroup *>(&images[start])->OriginalIndex = save;
        }
    }

    /* reset the ordering of imported list, such that it can be properly returned */
    std::sort(std::execution::par_unseq, images, images + nimport, [](const GroupType& a, const GroupType& b) {
        return a.get_base()->OriginalIndex < b.get_base()->OriginalIndex;
    });

#ifdef DEBUG
    for(int i = 0; i < nimport; i++) {
        const BaseGroup * gi = images[i].get_base();
        if(gi->MinIDTask != ThisTask) {
            endrun(5, "Error in basegroup import: minidtask %d != ThisTask %d\n", gi->MinIDTask, ThisTask);
        }
    }
    GroupType * ghosts2 = mymalloc("TMP", GroupType, nmemb);

    /* Debug code does an intermediate copy so we can check consistency. Non DEBUG mode likes to live dangerously*/
    MPI_Alltoallv_smart(images, Recv_count, NULL, dtype,
                        ghosts2, Send_count, NULL, dtype,
                        Comm);
    for(int i = 0; i < nmemb - Nmine; i ++) {
        const BaseGroup * g1 = groups[i+ Nmine].get_base();
        const BaseGroup * g2 = ghosts2[i].get_base();
        if(g1->MinID != g2->MinID) {
            endrun(2, "g1 minID %lu, g2 minID %lu\n", g1->MinID, g2->MinID);
        }
        if(g1->MinIDTask != g2->MinIDTask) {
            endrun(2, "g1 minIDTask %d, g2 minIDTask %d\n", g1->MinIDTask, g2->MinIDTask);
        }
    }
    memcpy(groups + Nmine, ghosts2, sizeof(GroupType) * (nmemb - Nmine));
    myfree(ghosts2);
#else
    MPI_Alltoallv_smart(images, Recv_count, NULL, dtype,
                        groups + Nmine, Send_count, NULL, dtype,
                        Comm);
#endif

    myfree(images);

    MPI_Type_free(&dtype);

    /* At this point, each Group entry has the reduced attribute of the full group */
    /* And the local groups (MinIDTask == ThisTask) are placed at the begining of the list*/
    ta_free(Recv_count);
    ta_free(Send_count);
}

static void fof_radix_Group_TotalCountTaskDiffMinID(const void * a, void * radix, void * arg);
static void fof_radix_Group_OriginalTaskMinID(const void * a, void * radix, void * arg);

static void fof_assign_grnr(struct BaseGroup * base, const int NgroupsExt, MPI_Comm Comm)
{
    int i, j, NTask, ThisTask;
    int64_t ngr;
    MPI_Comm_size(Comm, &NTask);
    MPI_Comm_rank(Comm, &ThisTask);

    #pragma omp parallel for
    for(i = 0; i < NgroupsExt; i++)
    {
        base[i].OriginalTask = ThisTask;	/* original task */
    }

    mpsort_mpi(base, NgroupsExt, sizeof(base[0]),
            fof_radix_Group_TotalCountTaskDiffMinID, 24, NULL, Comm);

    /* assign group numbers
     * at this point, both Group are is sorted by length,
     * and the every time OriginalTask == MinIDTask, a list of ghost base is stored.
     * they shall get the same GrNr.
     * */
    ngr = 0;
    for(i = 0; i < NgroupsExt; i++)
    {
        if(base[i].OriginalTask == base[i].MinIDTask) {
            ngr++;
        }
        base[i].GrNr = ngr;
    }

    int64_t * ngra = ta_malloc("NGRA", int64_t, NTask);

    MPI_Allgather(&ngr, 1, MPI_INT64, ngra, 1, MPI_INT64, Comm);

    /* shift to the global grnr. */
    int64_t groffset = 0;
    #pragma omp parallel for reduction(+: groffset)
    for(j = 0; j < ThisTask; j++)
        groffset += ngra[j];
    #pragma omp parallel for
    for(i = 0; i < NgroupsExt; i++)
        base[i].GrNr += groffset;

    ta_free(ngra);

    /* bring the group list back into the original task, sorted by MinID */
    mpsort_mpi(base, NgroupsExt, sizeof(base[0]),
            fof_radix_Group_OriginalTaskMinID, 16, NULL, Comm);
}

int
fof_save_groups(FOFGroups * fof, const std::string OutputDir, const std::string FOFFileBase, int num, Cosmology * CP, double atime, const double * MassTable, int MetalReturnOn, MPI_Comm Comm)
{
    auto fname = OutputDir + "/" + FOFFileBase + "_" + zpad(num, 3);
    message(0, "Saving particle groups into %s\n", fname.c_str());

    return fof_save_particles(fof, fname, fof_params.FOFSaveParticles, CP, atime, MassTable, MetalReturnOn, Comm);
}

/*
 * Deal with seeding of particles At each FOF stage,
 * if seed_index is >= 0,  then that particle on seed_task
 * will be converted to a seed.
 *
 * */

static void fof_seed_make_one(struct Group * g, int ThisTask, const double atime, const RandTable * const rnd) {
   if(g->seed_task != ThisTask) {
        endrun(7771, "Seed does not belong to the right task");
    }
    int index = g->seed_index;
    /* Random generator for the initial mass*/
    blackhole_make_one(index, atime, rnd);
}

void fof_seed(FOFGroups * fof, ActiveParticles * act, double atime, const RandTable * const rnd, MPI_Comm Comm)
{
    int i, j, n, ntot;

    int NTask;
    MPI_Comm_size(Comm, &NTask);

    char * Marked = mymalloc2("SeedMark", char, fof->Ngroups);

    int Nexport = 0;
    #pragma omp parallel for reduction(+:Nexport)
    for(i = 0; i < fof->Ngroups; i++)
    {
        Marked[i] =
            (fof->Group[i].Mass >= fof_params.MinFoFMassForNewSeed)
        &&  (fof->Group[i].MassType[4] >= fof_params.MinMStarForNewSeed)
        &&  (fof->Group[i].LenType[5] == 0)
        &&  (fof->Group[i].seed_index >= 0);

        if(Marked[i]) Nexport ++;
    }
    struct Group * ExportGroups = mymalloc("Export", Group, Nexport);
    j = 0;
    for(i = 0; i < fof->Ngroups; i ++) {
        if(Marked[i]) {
            ExportGroups[j] = fof->Group[i];
            j++;
        }
    }
    myfree(Marked);

    std::sort(std::execution::par_unseq, ExportGroups, ExportGroups + Nexport, [](const Group& a, const Group& b) {
        return a.seed_task < b.seed_task;
    });

    int * Send_count = ta_malloc("Send_count", int, NTask);
    int * Recv_count = ta_malloc("Recv_count", int, NTask);

    memset(Send_count, 0, NTask * sizeof(int));
    for(i = 0; i < Nexport; i++) {
        Send_count[ExportGroups[i].seed_task]++;
    }

    MPI_Alltoall(Send_count, 1, MPI_INT, Recv_count, 1, MPI_INT, Comm);

    int Nimport = 0;

    for(j = 0;  j < NTask; j++)
    {
        Nimport += Recv_count[j];
    }

    struct Group * ImportGroups = mymalloc2("ImportGroups", Group, Nimport);

    MPI_Alltoallv_smart(ExportGroups, Send_count, NULL, MPI_TYPE_GROUP,
                        ImportGroups, Recv_count, NULL, MPI_TYPE_GROUP,
                        Comm);

    myfree(ExportGroups);
    ta_free(Recv_count);
    ta_free(Send_count);

    MPI_Allreduce(&Nimport, &ntot, 1, MPI_INT, MPI_SUM, Comm);

    message(0, "Making %d new black hole particles.\n", ntot);

    /* Do we have enough black hole slots to create this many black holes?
     * If not, allocate more slots. */
    if(Nimport + SlotsManager->info[5].size > SlotsManager->info[5].maxsize)
    {
        int *ActiveParticle_tmp=NULL;
        /* This is only called on a PM step, so the condition should never be true*/
        if(act->ActiveParticle) {
            ActiveParticle_tmp = mymalloc2("ActiveParticle_tmp", int, act->NumActiveParticle);
            memmove(ActiveParticle_tmp, act->ActiveParticle, act->NumActiveParticle * sizeof(int));
            myfree(act->ActiveParticle);
        }

        /*Now we can extend the slots! */
        int64_t atleast[6];
        int64_t i;
        for(i = 0; i < 6; i++)
            atleast[i] = SlotsManager->info[i].maxsize;
        atleast[5] += ntot*1.1;
        slots_reserve(1, atleast, SlotsManager);

        /*And now we need our memory back in the right place*/
        if(ActiveParticle_tmp) {
            act->ActiveParticle = mymalloc("ActiveParticle", int, (act->NumActiveParticle + PartManager->MaxPart - PartManager->NumPart));
            memmove(act->ActiveParticle, ActiveParticle_tmp, act->NumActiveParticle * sizeof(int));
            myfree(ActiveParticle_tmp);
        }
    }

    int ThisTask;
    MPI_Comm_rank(Comm, &ThisTask);

    for(n = 0; n < Nimport; n++)
    {
        fof_seed_make_one(&ImportGroups[n], ThisTask, atime, rnd);
    }

    myfree(ImportGroups);

    walltime_measure("/FOF/Seeding");
}


static void fof_radix_Group_TotalCountTaskDiffMinID(const void * a, void * radix, void * arg) {
    uint64_t * u = (uint64_t *) radix;
    struct BaseGroup * f = (struct BaseGroup *) a;
    u[0] = labs(f->OriginalTask - f->MinIDTask);
    u[1] = f->MinID;
    u[2] = UINT64_MAX - (f->Length);
}

static void fof_radix_Group_OriginalTaskMinID(const void * a, void * radix, void * arg) {
    uint64_t * u = (uint64_t *) radix;
    struct BaseGroup * f = (struct BaseGroup *) a;
    u[0] = f->MinID;
    u[1] = f->OriginalTask;
}
