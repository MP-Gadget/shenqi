#ifndef FOF2_H
#define FOF2_H
/* Internal header for the friends-of-friends treewalks in fof2.cpp, which are used by fof.cpp. */

#include <mpi.h>
#include "types.h"
#include "domain.h"

/* The MinID and MinIDTask of a particle: this pair serves as a halo label. */
struct fof_particle_list
{
    MyIDType MinID;
    int MinIDTask;
    int Pindex;

    /* Order by minID */
    bool operator< (const struct fof_particle_list& p2) const
    {
        if(MinID < p2.MinID)
            return true;
        return false;
    }
};

/* Find the halo label of every particle.
 * A tree containing only the particles with types in PrimaryLinkTypes is built. Then:
 * 1. Particles with types in PrimaryLinkTypes which are closer than LinkingLength are linked
 *    into groups, and labelled with the minimum particle ID in their group. (FOF_FIND_GROUPS)
 * 2. Particles with types in SecondaryLinkTypes are given the label
 *    of the nearest primary particle, if there is one close enough. (FOF_FIND_NEAREST)
 * HaloLabel must have PartManager->NumPart entries. MinID and MinIDTask are set here: Pindex is not touched.*/
void fof_label_particles(struct fof_particle_list * HaloLabel, DomainDecomp * ddecomp, const double LinkingLength, const int PrimaryLinkTypes, const int SecondaryLinkTypes, MPI_Comm Comm);

#endif
