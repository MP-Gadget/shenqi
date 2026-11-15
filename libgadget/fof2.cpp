/*! \file fof2.cpp
 *  \brief The treewalks of the parallel FoF group finder
 *
 *  These find the label of the halo each particle belongs to. The group catalogue is built from the labels in fof.cpp.
 */
#include <mpi.h>
#include <string.h>
#include <execution>
#include <algorithm>
#include <ranges>
#include <omp.h>

#include "types.h"
#include "utils/endrun.h"
#include "utils/mymalloc.h"
#include "utils/spinlocks.h"
#include "utils/system.h"

#include "walltime.h"
#include "domain.h"
#include "partmanager.h"
#include "forcetree2.h"
#include "treewalk2.h"

#include "fof.h"
#include "fof2.h"

#define LARGE 1e29

/* This function walks the particle tree starting at particle i until it reaches
 * a particle which has Head[i] = i, the root node (particles are initialised in
 * this state, so this is equivalent to finding a particle which has yet to be merged).
 * Once it reaches a root, it returns that particle number.
 * Arguments:
 *
 * stop: When this particle is reached, return -1. We use this to find an already merged tree.
 *
 * Returns:
 *      root particle if found
 *      -1 if stop particle reached
 */
static int
HEADl(int stop, int i, const int * const Head)
{
    int next = i;

    do {
        i = next;
        /* Reached stop, return*/
        if(i == stop)
            return -1;
        /* atomic read because we may change
         * this in update_root: not necessary on x86_64, but avoids tears elsewhere*/
        #pragma omp atomic read
        next = Head[i];
    } while(next != i);

    /* return unmerged particle*/
    return i;
}

/* Rewrite a tree so that all values in it point directly to the true root.
 * This means that the trees are O(1) deep and speeds up future accesses.
 * See https://arxiv.org/abs/1607.03224
 * Never write the root's own entry: called with i == r, the old loop wrote Head[r] = r and
 * undid a concurrent link of r below a smaller root (a lost union). */
static void
update_root(int i, const int r, int * Head)
{
    int t = i;
    if(i == r)
        return;
    do {
        i = t;
        #pragma omp atomic capture
        {
            t = Head[i];
            Head[i]= r;
        }
        /* Stop if we reached the top (new head is the same as the old)
         * or if the new head is less than or equal to the desired head, indicating
         * another thread changed us*/
    } while(t != i && (t > r));
}

/* For tests/test_fof.cpp only: update_root is static. */
void
fof_update_root_for_test(int i, int r, int * Head)
{
    update_root(i, r, Head);
}

/* Find the current head particle by walking the tree. No updates are done
 * so this can be performed from a threaded context. */
static int
HEAD(int i, const int * const Head)
{
    int r = i;
    while(Head[r] != r) {
        r = Head[r];
    }
    return r;
}

/* Merge the groups containing the particles target and other, which are both on this rank.*/
static void
fofp_merge(int target, int other, int * const Head, struct fof_particle_list * const HaloLabel, struct SpinLocks * const spin)
{
    int h1, h2;
    do {
        h1 = HEADl(-1, target, Head);
        /* Done if we find h1 along the path
         * (because other is already in the same halo) */
        h2 = HEADl(h1, other, Head);
        if(h2 < 0)
            return;
        /* Ensure that we always merge to the lower entry.
         * This avoids circular loops in the Head entries:
         * a -> b -> a */
        if(h1 > h2) {
            int tmp = h2;
            h2 = h1;
            h1 = tmp;
        }
     /* Atomic compare exchange to make h2 a subtree of h1.
      * Set Head[h2] = h1 iff Head[h2] is still h2. Otherwise loop.*/
    } while(!__atomic_compare_exchange(&Head[h2], &h2, &h1, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED));

    /* update MinID of h1: h2 is now just another child of h1
     * so we don't need to check that h2 changes its head.
     * It might happen that h1 is added to another halo at this point
     * and the addition gets the wrong MinID.
     * For this reason we recompute the MinIDs after the main treewalk.
     * We also lock h2 for a copy in case it is the h1 in another thread,
     * and may have inconsistent MinID and MinIDTask.*/

    /* Get a copy of h2 under the lock, which ensures
     * that MinID and MinIDTask do not change independently. */
    struct fof_particle_list h2label;
    lock_spinlock(h2, spin);
    h2label.MinID = HaloLabel[h2].MinID;
    h2label.MinIDTask = HaloLabel[h2].MinIDTask;
    unlock_spinlock(h2, spin);

    /* Now lock h1 so we don't change MinID but not MinIDTask.*/
    lock_spinlock(h1, spin);
    if(HaloLabel[h1].MinID > h2label.MinID)
    {
        HaloLabel[h1].MinID = h2label.MinID;
        HaloLabel[h1].MinIDTask = h2label.MinIDTask;
    }
    unlock_spinlock(h1, spin);

    /* h1 must be the root of other and target both:
     * do the splay to speed up future accesses.
     * We do not need to have h2 locked, because h2 is
     * now just another child of h1: these do not change the root,
     * they make the tree shallow.*/
    update_root(target, h1, Head);
    update_root(other, h1, Head);
}

/* Make a list of the particles on this rank for which the test function returns true.
 * The list should be freed with myfree. Returns the length of the list.*/
template <typename TestFunction>
static int64_t
fof_build_worklist(int ** WorkList, const int64_t NumPart, TestFunction test)
{
    *WorkList = mymalloc("FOFWorkList", int, NumPart + 1);
    auto iota = std::views::iota(0, (int) NumPart);
    auto end = std::copy_if(std::execution::par, iota.begin(), iota.end(), *WorkList, test);
    return end - *WorkList;
}

/************************ FOF_FIND_GROUPS: link the primary particles into groups ****************************/

/* Inputs to the primary treewalk.*/
class FOFPrimaryPriv : public ParamTypeBase {
    public:
    /* The linking length*/
    double LinkingLength;
    const particle_data * Parts;
    /* These are used to find the label of a particle which is being exported. */
    const int * Head;
    const struct fof_particle_list * HaloLabel;

    FOFPrimaryPriv(const double BoxSize, const double i_LinkingLength, const particle_data * const i_Parts, const int * const i_Head, const struct fof_particle_list * const i_HaloLabel):
        ParamTypeBase(BoxSize), LinkingLength(i_LinkingLength), Parts(i_Parts), Head(i_Head), HaloLabel(i_HaloLabel)
    {}
};

class FOFPrimaryQuery : public TreeWalkQueryBase<FOFPrimaryPriv>
{
    public:
    MyFloat Hsml;
    /* The label of the group this particle was in when the treewalk started. This is used when the query is exported.*/
    MyIDType MinID;
    int MinIDTask;
    /* The index of the particle on its own rank. This is only meaningful when the query is walking the tree of its own rank.
     * The local treewalk links particles with their indices and the exported treewalk with the label above.*/
    int Target;

    FOFPrimaryQuery(const particle_data& particle, const int * const i_NodeList, const FOFPrimaryPriv& priv):
    TreeWalkQueryBase<FOFPrimaryPriv>(particle, i_NodeList, priv), Hsml(priv.LinkingLength),
    MinID(IDTYPE_MAX), MinIDTask(-1), Target(&particle - priv.Parts)
    {
        /* The label is only used on other ranks, but an exported query is a copy of the local one, so always store it.
         * The queries are made before the threaded treewalk, so the head and its label are not changing here.*/
        const int head = HEAD(Target, priv.Head);
        MinID = priv.HaloLabel[head].MinID;
        MinIDTask = priv.HaloLabel[head].MinIDTask;
    }
};

/* Nothing is returned to the walking particle.*/
class FOFPrimaryResult : public TreeWalkResultBase<FOFPrimaryQuery> {
    public:
    using TreeWalkResultBase::TreeWalkResultBase;
};

/* The storage on the rank doing the visiting: the primary treewalk links the particles it finds.*/
class FOFPrimaryOutput {
    public:
    /* Head[i] is another particle in the same group as particle i, or i if it is the head of the group.*/
    int * Head;
    struct fof_particle_list * HaloLabel;
    /* The lock is used to protect MinID*/
    struct SpinLocks * spin;
    const particle_data * Parts;

    FOFPrimaryOutput(int * const i_Head, struct fof_particle_list * const i_HaloLabel, const particle_data * const i_Parts, const int64_t NumPart):
        Head(i_Head), HaloLabel(i_HaloLabel), Parts(i_Parts)
    {
        spin = init_spinlocks(NumPart);
    }

    ~FOFPrimaryOutput()
    {
        free_spinlocks(spin);
    }

    /* Nothing to do after the treewalk: the labels are made consistent in fof_label_primary.*/
    void postprocess(const int p, const FOFPrimaryResult& result, particle_data * const parts, const FOFPrimaryPriv * priv) {}
};

/* The types of particle which are linked are a runtime parameter, so the compile-time mask is all particles.
 * The tree is built just before the treewalk and contains only the primary particles. */
class FOFPrimaryLocalTreeWalk: public LocalNgbTreeWalk<FOFPrimaryLocalTreeWalk, FOFPrimaryQuery, FOFPrimaryResult, FOFPrimaryPriv, BaseNode, NGB_TREEFIND_ASYMMETRIC, ALLMASK>
{
    public:
    FOFPrimaryOutput * const visited;
    /* True if the query being walked was exported from another rank. Set in visit.*/
    bool ghost = false;

    FOFPrimaryLocalTreeWalk(const BaseNode * const Nodes, const FOFPrimaryQuery& input, FOFPrimaryOutput * const i_visited):
    LocalNgbTreeWalk<FOFPrimaryLocalTreeWalk, FOFPrimaryQuery, FOFPrimaryResult, FOFPrimaryPriv, BaseNode, NGB_TREEFIND_ASYMMETRIC, ALLMASK>(Nodes, input),
    visited(i_visited)
    {}

    /* The exported queries are copies of the local ones, so ngbiter cannot tell from the query
     * whether the walking particle is on this rank. Record it from the treewalk mode.*/
    template<TreeWalkReduceMode mode>
    FOFPrimaryResult visit(FOFPrimaryQuery& input, const FOFPrimaryPriv& priv, const struct particle_data * const parts)
    {
        ghost = (mode == TREEWALK_GHOSTS);
        return LocalNgbTreeWalk<FOFPrimaryLocalTreeWalk, FOFPrimaryQuery, FOFPrimaryResult, FOFPrimaryPriv, BaseNode, NGB_TREEFIND_ASYMMETRIC, ALLMASK>::template visit<mode>(input, priv, parts);
    }

    /* The tree does not contain all particle types, only the primary ones, so do not check the mask.*/
    template <typename TreeType>
    static void validate_tree(const TreeType * const tree) {}

    void ngbiter(FOFPrimaryQuery& input, const particle_data& particle, FOFPrimaryResult * output, const FOFPrimaryPriv& priv)
    {
        double dist[3];
        const double r2 = this->get_distance(input, particle, priv.BoxSize, dist);
        if(r2 > input.Hsml * input.Hsml)
            return;
        const int other = &particle - visited->Parts;

        if(!ghost) {
            /* Local FOF: both particles are on this rank.*/
            if(input.Target <= other)
                fofp_merge(input.Target, other, visited->Head, visited->HaloLabel, visited->spin);
        }
        else /* target is a ghost from another rank: give its label to the group of the particle we found.*/
        {
            const int head = HEAD(other, visited->Head);
            lock_spinlock(head, visited->spin);
            if(visited->HaloLabel[head].MinID > input.MinID)
            {
                visited->HaloLabel[head].MinID = input.MinID;
                visited->HaloLabel[head].MinIDTask = input.MinIDTask;
            }
            unlock_spinlock(head, visited->spin);
        }
    }
};

class FOFPrimaryTopTreeWalk: public TopTreeWalk<FOFPrimaryQuery, FOFPrimaryPriv, BaseNode, NGB_TREEFIND_ASYMMETRIC> { using TopTreeWalk::TopTreeWalk; };

class FOFPrimaryTreeWalk: public TreeWalk<FOFPrimaryTreeWalk, FOFPrimaryQuery, FOFPrimaryResult, FOFPrimaryLocalTreeWalk, FOFPrimaryTopTreeWalk, FOFPrimaryPriv, FOFPrimaryOutput, ForceTreeSimple> {
    public:
    using TreeWalk::TreeWalk;

    /* Give each local walk the group links of the rank doing the visiting, which it changes.*/
    FOFPrimaryLocalTreeWalk make_local_walk(const FOFPrimaryQuery& input)
    {
        return FOFPrimaryLocalTreeWalk(tree->get_nodes(), input, output);
    }
};

static void
fof_label_primary(struct fof_particle_list * HaloLabel, const ForceTreeSimple * const tree, const double LinkingLength, const int PrimaryLinkTypes, MPI_Comm Comm)
{
    int64_t i;
    int64_t link_across_tot;
    int ThisTask;
    MPI_Comm_rank(Comm, &ThisTask);
    particle_data * const parts = PartManager->Base;
    const int64_t NumPart = PartManager->NumPart;

    message(0, "Start linking particles (presently allocated=%g MB)\n", mymalloc_usedbytes() / (1024.0 * 1024.0));

    int * Head = mymalloc("FOF_Links", int, NumPart);
    char * PrimaryActive = mymalloc("FOFActive", char, NumPart);
    MyIDType * OldMinID = mymalloc("FOFActive", MyIDType, NumPart);

    #pragma omp parallel for
    for(i = 0; i < NumPart; i++)
    {
        Head[i] = i;
        OldMinID[i]= parts[i].ID;
        PrimaryActive[i] = 1;

        HaloLabel[i].MinID = parts[i].ID;
        HaloLabel[i].MinIDTask = ThisTask;
    }

    {
        FOFPrimaryPriv priv(PartManager->BoxSize, LinkingLength, parts, Head, HaloLabel);
        FOFPrimaryOutput output(Head, HaloLabel, parts, NumPart);
        struct SpinLocks * const spin = output.spin;
        FOFPrimaryTreeWalk tw("FOF_FIND_GROUPS", tree, priv, &output);
        do
        {
            double t0 = second();

            /* The particles which do the linking are the primary particles which changed their label last time.*/
            int * WorkList;
            const int64_t nwork = fof_build_worklist(&WorkList, NumPart, [parts, PrimaryActive, PrimaryLinkTypes](const int n) {
                if(parts[n].IsGarbage || parts[n].Swallowed)
                    return false;
                return ((1 << parts[n].Type) & PrimaryLinkTypes) && PrimaryActive[n];
            });
            tw.run_on_queue(WorkList, nwork, parts, Comm);
            myfree(WorkList);

            double t1 = second();
            /* This sets the MinID of the head particle to the minimum ID
             * of the child particles. We set this inside the treewalk,
             * but the locking allows a race, where the particle with MinID set
             * is no longer the one which is the true Head of the group.
             * So we must check it again here.*/
            #pragma omp parallel for
            for(i = 0; i < NumPart; i++) {
                int head = HEAD(i, Head);
                /* Don't check against ourself*/
                if(head == i)
                    continue;
                MyIDType headminid;
                #pragma omp atomic read
                headminid = HaloLabel[head].MinID;
                /* No atomic needed for i as this is not a head*/
                if(headminid > HaloLabel[i].MinID) {
                    lock_spinlock(head, spin);
                    if(HaloLabel[head].MinID > HaloLabel[i].MinID) {
                        #pragma omp atomic write
                        HaloLabel[head].MinID = HaloLabel[i].MinID;
                        HaloLabel[head].MinIDTask = HaloLabel[i].MinIDTask;
                    }
                    unlock_spinlock(head, spin);
                }
            }
            /* let's check out which particles have changed their MinID,
             * mark them for next round. */
            int64_t link_across = 0;
            #pragma omp parallel for reduction(+: link_across)
            for(i = 0; i < NumPart; i++) {
                int head = HEAD(i, Head);
                /* This loop sets the MinID of the children to the minID of the head.
                 * The minID of the head is set above and is stable at this point.*/
                if(i != head) {
                    HaloLabel[i].MinID = HaloLabel[head].MinID;
                    HaloLabel[i].MinIDTask = HaloLabel[head].MinIDTask;
                }
                MyIDType newMinID = HaloLabel[head].MinID;
                if(newMinID != OldMinID[i]) {
                    PrimaryActive[i] = 1;
                    OldMinID[i] = newMinID;
                    link_across ++;
                } else {
                    PrimaryActive[i] = 0;
                }
            }
            double t2 = second();

            MPI_Allreduce(&link_across, &link_across_tot, 1, MPI_INT64, MPI_SUM, Comm);
            message(0, "Linked %ld particles %g seconds postproc was %g seconds\n", link_across_tot, t1 - t0, t2 - t1);
        }
        while(link_across_tot > 0);
    }

    message(0, "Local groups found.\n");

    myfree(OldMinID);
    myfree(PrimaryActive);
    myfree(Head);
}

/************************ FOF_FIND_NEAREST: attach the secondary particles to the nearest primary particle ****************************/

/* Inputs to the secondary treewalk.*/
class FOFSecondaryPriv : public ParamTypeBase {
    public:
    const particle_data * Parts;
    /* The current search radius of each particle*/
    const float * hsml;
    /* The labels of the particles on the rank doing the visiting, which are read for the primary particles found.*/
    const struct fof_particle_list * HaloLabel;

    FOFSecondaryPriv(const double BoxSize, const particle_data * const i_Parts, const float * const i_hsml, const struct fof_particle_list * const i_HaloLabel):
        ParamTypeBase(BoxSize), Parts(i_Parts), hsml(i_hsml), HaloLabel(i_HaloLabel)
    {}
};

class FOFSecondaryQuery : public TreeWalkQueryBase<FOFSecondaryPriv>
{
    public:
    /* The search radius. This is reduced during the walk once a neighbour is found.*/
    MyFloat Hsml;

    FOFSecondaryQuery(const particle_data& particle, const int * const i_NodeList, const FOFSecondaryPriv& priv):
    TreeWalkQueryBase<FOFSecondaryPriv>(particle, i_NodeList, priv), Hsml(priv.hsml[&particle - priv.Parts])
    {}
};

class FOFSecondaryResult : public TreeWalkResultBase<FOFSecondaryQuery> {
    public:
    /* Distance to the nearest primary particle found by this walk and the label of that particle.*/
    MyFloat Distance = LARGE;
    MyIDType MinID = IDTYPE_MAX;
    int MinIDTask = -1;

    FOFSecondaryResult(const FOFSecondaryQuery& query):
        TreeWalkResultBase<FOFSecondaryQuery>(query) {}

    /* Keep the nearest of the neighbours found on each rank.*/
    FOFSecondaryResult& operator+=(const FOFSecondaryResult& other)
    {
        static_cast<TreeWalkResultBase<FOFSecondaryQuery>&>(*this) += static_cast<const TreeWalkResultBase<FOFSecondaryQuery>&>(other);
        if(other.Distance < Distance)
        {
            Distance = other.Distance;
            MinID = other.MinID;
            MinIDTask = other.MinIDTask;
        }
        return *this;
    }
};

class FOFSecondaryOutput {
    public:
    /* Distance to the nearest primary particle found so far. LARGE if none was found yet and -1 if we stopped looking.*/
    float * distance;
    /* The current search radius of each particle*/
    float * hsml;
    struct fof_particle_list * HaloLabel;
    /* Number of particles for each thread which need another iteration*/
    int64_t * npleft;
    /* We only search out to this distance.*/
    double MaxHsml;

    FOFSecondaryOutput(struct fof_particle_list * const i_HaloLabel, const double i_MaxHsml, const int64_t NumPart):
        HaloLabel(i_HaloLabel), MaxHsml(i_MaxHsml)
    {
        distance = mymalloc("FOF_SECONDARY->distance", float, NumPart);
        hsml = mymalloc("FOF_SECONDARY->hsml", float, NumPart);
        npleft = ta_malloc("NPLeft", int64_t, omp_get_max_threads());
    }

    ~FOFSecondaryOutput()
    {
        ta_free(npleft);
        myfree(hsml);
        myfree(distance);
    }

    /* Store the nearest neighbour found. If no neighbour was found, look again with a larger search radius.*/
    void
    postprocess(const int p, const FOFSecondaryResult& result, particle_data * const parts, const FOFSecondaryPriv * priv)
    {
        if(result.Distance < distance[p] && result.Distance >= 0 && result.Distance < 0.5 * LARGE)
        {
            distance[p] = result.Distance;
            HaloLabel[p].MinID = result.MinID;
            HaloLabel[p].MinIDTask = result.MinIDTask;
        }
        if(distance[p] > 0.5 * LARGE)
        {
            if(hsml[p] < MaxHsml)  /* we only search out to a maximum distance */
            {
                /* need to redo this particle */
                npleft[omp_get_thread_num()]++;
                hsml[p] *= 2.0;
            } else {
                distance[p] = -1;  /* we not continue to search for this particle */
            }
        }
    }
};

/* Finds the nearest primary particle. As for the primary treewalk, the tree contains only
 * the primary particles, whose types are a runtime parameter, so the compile-time mask is all particles.*/
class FOFSecondaryLocalTreeWalk: public LocalNgbTreeWalk<FOFSecondaryLocalTreeWalk, FOFSecondaryQuery, FOFSecondaryResult, FOFSecondaryPriv, BaseNode, NGB_TREEFIND_ASYMMETRIC, ALLMASK>
{
    public:
    using LocalNgbTreeWalk::LocalNgbTreeWalk;

    /* The tree does not contain all particle types, only the primary ones, so do not check the mask.*/
    template <typename TreeType>
    static void validate_tree(const TreeType * const tree) {}

    void ngbiter(FOFSecondaryQuery& input, const particle_data& particle, FOFSecondaryResult * output, const FOFSecondaryPriv& priv)
    {
        double dist[3];
        const double r2 = this->get_distance(input, particle, priv.BoxSize, dist);
        if(r2 > input.Hsml * input.Hsml)
            return;
        const double r = sqrt(r2);
        if(r < output->Distance)
        {
            const int other = &particle - priv.Parts;
            output->Distance = r;
            output->MinID = priv.HaloLabel[other].MinID;
            output->MinIDTask = priv.HaloLabel[other].MinIDTask;
        }
        /* No need to search nodes at a greater distance
         * now that we have a neighbour.*/
        input.Hsml = r;
    }
};

class FOFSecondaryTopTreeWalk: public TopTreeWalk<FOFSecondaryQuery, FOFSecondaryPriv, BaseNode, NGB_TREEFIND_ASYMMETRIC> { using TopTreeWalk::TopTreeWalk; };

class FOFSecondaryTreeWalk: public TreeWalk<FOFSecondaryTreeWalk, FOFSecondaryQuery, FOFSecondaryResult, FOFSecondaryLocalTreeWalk, FOFSecondaryTopTreeWalk, FOFSecondaryPriv, FOFSecondaryOutput, ForceTreeSimple> {
    public:
    using TreeWalk::TreeWalk;
};

static void
fof_label_secondary(struct fof_particle_list * HaloLabel, const ForceTreeSimple * const tree, const double LinkingLength, const int SecondaryLinkTypes, MPI_Comm Comm)
{
    int64_t n;
    particle_data * const parts = PartManager->Base;
    const int64_t NumPart = PartManager->NumPart;

    message(0, "Start finding nearest dm-particle (presently allocated=%g MB)\n",
            mymalloc_usedbytes() / (1024.0 * 1024.0));

    /* we only search out to a maximum distance */
    FOFSecondaryOutput output(HaloLabel, 4 * LinkingLength, NumPart);
    float * const distance = output.distance;
    float * const hsml = output.hsml;

    #pragma omp parallel for
    for(n = 0; n < NumPart; n++)
    {
        distance[n] = LARGE;
        hsml[n] = 0.4 * LinkingLength;

        if((parts[n].Type == 0 || parts[n].Type == 4 || parts[n].Type == 5) && hsml[n] < 0.5 * parts[n].Hsml) {
            /* use gas sml as a hint (faster convergence than 0.1 fof_params.FOFHaloComovingLinkingLength at high-z */
            hsml[n] = 0.5 * parts[n].Hsml;
        }
    }

    FOFSecondaryPriv priv(PartManager->BoxSize, parts, hsml, HaloLabel);
    FOFSecondaryTreeWalk tw("FOF_FIND_NEAREST", tree, priv, &output);

    int64_t ntot;
    int Niteration = 0;
    const int NumThreads = omp_get_max_threads();

    /* we will repeat the whole thing for those particles where we didn't find enough neighbours */
    message(0, "fof-nearest iteration started\n");

    do
    {
        memset(output.npleft, 0, sizeof(int64_t) * NumThreads);

        /* The secondary particles where we have not yet found a neighbour, and have not stopped looking.*/
        int * WorkList;
        const int64_t nwork = fof_build_worklist(&WorkList, NumPart, [parts, distance, SecondaryLinkTypes](const int n) {
            if(parts[n].IsGarbage || parts[n].Swallowed)
                return false;
            /* Exclude particles where we already found a neighbour*/
            if(distance[n] < 0.5 * LARGE)
                return false;
            return (bool) ((1 << parts[n].Type) & SecondaryLinkTypes);
        });
        /* This increases the search radius of the particles without a neighbour in the postprocess.*/
        tw.run_on_queue(WorkList, nwork, parts, Comm);
        myfree(WorkList);
        Niteration++;

        for(n = 1; n < NumThreads; n++)
            output.npleft[0] += output.npleft[n];

        MPI_Allreduce(&output.npleft[0], &ntot, 1, MPI_INT64, MPI_SUM, Comm);

        if(ntot < 0 || (ntot > 0 && Niteration > MAXITER))
            endrun(1159, "Failed to converge in fof-nearest: ntot %ld", ntot);
    }
    while(ntot > 0);
}

void
fof_label_particles(struct fof_particle_list * HaloLabel, DomainDecomp * ddecomp, const double LinkingLength, const int PrimaryLinkTypes, const int SecondaryLinkTypes, MPI_Comm Comm)
{
    /* We only need a tree containing primary linking particles only. No moments*/
    ForceTreeSimple dmtree(*PartManager, ddecomp, PrimaryLinkTypes, "");
    walltime_measure("/FOF/Build");

    /* Fill FOFP_List of primary */
    fof_label_primary(HaloLabel, &dmtree, LinkingLength, PrimaryLinkTypes, Comm);
    walltime_measure("/FOF/Primary");

    /* Fill FOFP_List of secondary */
    fof_label_secondary(HaloLabel, &dmtree, LinkingLength, SecondaryLinkTypes, Comm);
    message(0, "Attached gas and star particles to nearest dm particles.\n");
    walltime_measure("/FOF/Secondary");
}
