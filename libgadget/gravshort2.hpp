#ifndef GRAVSHORT_2_H
#define GRAVSHORT_2_H
/*! \file gravtree.c
 *  \brief Classes for gravitational (short-range) force computation. Included in both the CUDA and CPP files.
 */

#include <mpi.h>
#include <stdlib.h>
#include <math.h>

#include "types.h"
#include "utils/endrun.h"
#include "utils/mymalloc.h"
#include "localtreewalk2.h"

#include "forcetree.h"
#include "partmanager.h"
#include "gravity.h"

/* Class containing the fixed parameters of the gravity treewalk. */
 class GravTreeParams : public ParamTypeBase {
     public:
     /* Size of a PM cell, in internal units. Box / Nmesh */
     double cellsize;
     /* How many PM cells do we go
      * before we stop calculating the tree?*/
     double Rcut;
     /* Newton's constant in internal units*/
     double G;
     inttime_t Ti_Current;
     /* Matter density in internal units.
      * rho_0 = Omega0 * rho_crit
      * rho_crit = 3 H^2 /(8 pi G).
      * This is (rho_0)^(1/3) ,
      * Note: should account for
      * massive neutrinos, but doesn't. */
     double cbrtrho0;
     GravShortTable gravtab;
     /* Force softening*/
     double ForceSoftening;
     double ErrTolForceAcc;
     double BHOpeningAngle2;
     int TreeUseBH;

     GravTreeParams(const struct gravshort_tree_params TreeParams, const inttime_t i_Ti_Current, const double rho0, const PetaPM * const pm, const double BoxSize, GravShortTable& gravtab):
     ParamTypeBase(BoxSize), cellsize(BoxSize / pm->Nmesh), Rcut(TreeParams.Rcut * pm->Asmth * cellsize), G(pm->G), Ti_Current(i_Ti_Current),
     cbrtrho0(pow(rho0, 1.0 / 3)), gravtab(gravtab), ForceSoftening(FORCE_SOFTENING()),
     ErrTolForceAcc(TreeParams.ErrTolForceAcc), BHOpeningAngle2(TreeParams.BHOpeningAngle * TreeParams.BHOpeningAngle), TreeUseBH(TreeParams.TreeUseBH)
     {
        /* Enforce a maximum opening angle even for relative acceleration criterion, to avoid
        * pathological cases. Default value is 0.9, from Volker Springel.*/
        if(TreeUseBH == 0)
            BHOpeningAngle2 = TreeParams.MaxBHOpeningAngle * TreeParams.MaxBHOpeningAngle;
     }
 };

/* Class to store pointers to the outputs of the gravity code. */
class GravTreeOutput
{
    public:
     /* Pointer to the place to store accelerations*/
     MyFloat (*Accel)[3];
     int accelstorealloc;
     /* If this is true, we have all particles and need to update the gravitational potential */
     bool update_potential;
     GravTreeOutput(MyFloat (* AccelStore)[3], const size_t NumPart, const bool i_update_potential): Accel(AccelStore), update_potential(i_update_potential)
     {
         accelstorealloc = 0;
         if(!AccelStore) {
             Accel = (MyFloat (*) [3]) mymanagedmalloc("GravAccel", NumPart * sizeof(Accel[0]));
             accelstorealloc = 1;
         }
     }
     ~GravTreeOutput(void)
     {
         if(accelstorealloc)
             myfree(Accel);
     }

     /**
     * Postprocess - finalize quantities after tree walk completes.
     * Override to normalize results, compute derived quantities, etc.
     *
     * @param i Particle index
     * @param parts Array of particle data to index with i
     * @param priv Data structure for parameters of the gravity treewalk.
     */
     MYCUDAFN void postprocess(const int i, particle_data * const parts, const GravTreeParams * priv)
     {
         const double G = priv->G;
         Accel[i][0] *= G;
         Accel[i][1] *= G;
         Accel[i][2] *= G;

         if(update_potential) {
             /* On a PM step, update the stored full tree grav accel for the next PM step.
             * Needs to be done here so internal treewalk iterations don't get a partial acceleration.*/
             parts[i].FullTreeGravAccel[0] = Accel[i][0];
             parts[i].FullTreeGravAccel[1] = Accel[i][1];
             parts[i].FullTreeGravAccel[2] = Accel[i][2];
             /* calculate the potential */
             parts[i].Potential += parts[i].Mass / (priv->ForceSoftening / 2.8);
             /* remove self-potential */
             parts[i].Potential -= 2.8372975 * pow(parts[i].Mass, 2.0 / 3) * priv->cbrtrho0;
             parts[i].Potential *= G;
         }
     }
};

 /*Compute the absolute magnitude of the acceleration for a particle.*/
 MYCUDAFN static MyFloat
 grav_get_abs_accel(const struct particle_data& PP, const double G)
 {
     double aold=0;
     int j;
     for(j = 0; j < 3; j++) {
        double ax = PP.FullTreeGravAccel[j] + PP.GravPM[j];
        aold += ax*ax;
     }
     return sqrt(aold) / G;
 }

 class GravTreeQuery : public TreeWalkQueryBase<GravTreeParams> {
    public:
    MyFloat OldAcc;
    MYCUDAFN GravTreeQuery(const particle_data& particle, const int * const i_NodeList, const int firstnode, const GravTreeParams& priv) :
    TreeWalkQueryBase(particle, i_NodeList, firstnode, priv), OldAcc(grav_get_abs_accel(particle, priv.G)) {}
 };

class GravTreeResult : public TreeWalkResultBase<GravTreeQuery, GravTreeOutput> {
    public:
    MyFloat Acc[3] = {0};
    MyFloat Potential = 0;
    MYCUDAFN GravTreeResult(GravTreeQuery& query): TreeWalkResultBase(query), Potential(0) {}

    template<TreeWalkReduceMode mode>
    MYCUDAFN void reduce(const int place, const GravTreeOutput * output, struct particle_data * const parts)
    {
        TreeWalkResultBase<GravTreeQuery, GravTreeOutput>::reduce<mode>(place, output, parts);
        TREEWALK_REDUCE(output->Accel[place][0], Acc[0]);
        TREEWALK_REDUCE(output->Accel[place][1], Acc[1]);
        TREEWALK_REDUCE(output->Accel[place][2], Acc[2]);
        if(output->update_potential) {
            TREEWALK_REDUCE(parts[place].Potential, Potential);
        }
    }
};

/* Check whether a node should be discarded completely, its contents not contributing
 * to the acceleration. This happens if the node is further away than the short-range force cutoff.
 * Return 1 if the node should be discarded, 0 otherwise. */
static MYCUDAFN int
shall_we_discard_node(const double len, const double r2, const double center[3], const double inpos[3], const double BoxSize, const double rcut, const double rcut2)
{
    /* This checks the distance from the node center of mass
     * is greater than the cutoff. */
    if(r2 <= rcut2)
        return 0;
    /* check whether we can stop walking along this branch */
    const double eff_dist = rcut + 0.5 * len;
    /*This checks whether we are also outside this region of the oct-tree*/
    /* As long as one dimension is outside, we are fine*/
    for(int i=0; i < 3; i++)
        if(fabs(NEAREST(center[i] - inpos[i], BoxSize)) > eff_dist)
            return 1;
    return 0;
}

/* This function tests whether a node shall be opened (ie, should the next node be .
 * If it should be discarded, 0 is returned.
 * If it should be used, 1 is returned, otherwise zero is returned. */
static MYCUDAFN int
shall_we_open_node(const double len, const double mass, const double r2, const double center[3], const double inpos[3], const double BoxSize, const double aold, const int TreeUseBH, const double BHOpeningAngle2)
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

class GravLocalTreeWalk {
    public:
    /* A pointer to the force tree structure to walk.*/
    const struct NODE * const Nodes;

    MYCUDAFN GravLocalTreeWalk(const NODE * const Node, const GravTreeQuery& input):
    Nodes(Node) {}

    static void validate_tree(const ForceTree * const tree)
    {
        if(!force_tree_allocated(tree))
            endrun(0, "Tree has been freed before this treewalk.\n");
        /* Check whether the tree contains the particles we are looking for*/
        int mask = GASMASK + DMMASK + STARMASK + BHMASK; // Neutrinos may be absent.
        if((tree->mask & mask) != mask)
            endrun(5, "Gravity treewalk needs all particle types but tree mask is %d\n", tree->mask);

        if(!tree->moments_computed_flag)
            endrun(2, "Gravtree called before tree moments computed!\n");
    }

    /*! In the TreePM algorithm, the tree is walked only locally around the
     *  target coordinate.  Tree nodes that fall outside a box of half
     *  side-length Rcut= RCUT*ASMTH*MeshSize can be discarded. The short-range
     *  potential is modified by a complementary error function, multiplied
     *  with the Newtonian form. The resulting short-range suppression compared
     *  to the Newtonian force is tabulated, because looking up from this table
     *  is faster than recomputing the corresponding factor, despite the
     *  memory-access penalty (which reduces cache performance) incurred by the
     *  table.
     * Returns the number of particle-particle and particle-node interactions.
     */
    template<TreeWalkReduceMode mode>
    MYCUDAFN int64_t visit(const GravTreeQuery& input, GravTreeResult * output, const GravTreeParams& priv, const struct particle_data * const parts)
    {
        static_assert(mode != TREEWALK_TOPTREE, "Toptree should call toptree_visit, not visit.");

        /*Tree-opening constants*/
        const double cellsize = priv.cellsize;
        const double rcut = priv.Rcut;
        const double rcut2 = rcut * rcut;
        const double aold = priv.ErrTolForceAcc * input.OldAcc;

        //message(1, "BH: %d, opening angle %g aold %g\n", TreeUseBH, BHOpeningAngle2, aold);
        /*Start the tree walk*/
        int64_t listindex, ninteractions=0;

        /* Primary treewalk only ever has one nodelist entry*/
        for(listindex = 0; listindex < NODELISTLENGTH; listindex++)
        {
            /* Use the next node in the node list if we are doing a secondary walk.
             * For a primary walk the node list only ever contains one node. */
            int no = input.NodeList[listindex];
            const int startno = no;
            if(no < 0)
                break;

            while(no >= 0)
            {
                /* The tree always walks internal nodes*/
                const struct NODE * const nop = &Nodes[no];

                if constexpr(mode == TREEWALK_GHOSTS) {
                    if(nop->f.TopLevel && no != startno)  /* we reached a top-level node again, which means that we are done with the branch */
                        break;
                }

                double dx[3];
                for(int i = 0; i < 3; i++)
                    dx[i] = NEAREST(nop->mom.cofm[i] - input.Pos[i], priv.BoxSize);
                const double r2 = dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2];

                /* Discard this node, move to sibling*/
                if(shall_we_discard_node(nop->len, r2, nop->center, input.Pos, priv.BoxSize, rcut, rcut2))
                {
                    no = nop->sibling;
                    /* Don't add this node*/
                    continue;
                }

                /* This node accelerates the particle directly, and is not opened.*/
                const int open_node = shall_we_open_node(nop->len, nop->mom.mass, r2, nop->center, input.Pos, priv.BoxSize, aold, priv.TreeUseBH, priv.BHOpeningAngle2);

                if(!open_node)
                {
                    /* ok, node can be used */
                    no = nop->sibling;
                    /* Compute the acceleration and apply it to the output structure*/
                    apply_accn(output, dx, r2, nop->mom.mass, cellsize, priv.gravtab, priv.ForceSoftening);
                    ninteractions++;
                    continue;
                }

                /* Now we have a cell that needs to be opened.
                * If it contains particles we can add them directly here */
                if(nop->f.ChildType == PARTICLE_NODE_TYPE)
                {
                    /* Loop over child particles*/
                    for(int i = 0; i < nop->s.noccupied; i++) {
                        const int pp = nop->s.suns[i];
                        for(int j = 0; j < 3; j++)
                            dx[j] = NEAREST(parts[pp].Pos[j] - input.Pos[j], priv.BoxSize);
                        const double r2 = dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2];
                        /* Compute the acceleration and apply it to the output structure*/
                        apply_accn(output, dx, r2, parts[pp].Mass, cellsize, priv.gravtab, priv.ForceSoftening);
                        ninteractions++;
                    }
                    no = nop->sibling;
                    continue;
                }
                else if (nop->f.ChildType == PSEUDO_NODE_TYPE)
                {
                    // This should never happen in TREEWALK_GHOSTS mode! But we cannot endrun from the GPU.
                    #ifndef __CUDACC__
                    if constexpr(mode == TREEWALK_GHOSTS)
                        endrun(12312, "Secondary for particle from node %d found pseudo at %d.\n", input.NodeList[listindex], no);
                    #endif
                    /* Move to the sibling (likely also a pseudo node)*/
                    no = nop->sibling;
                    continue;
                }
                //NODE_NODE_TYPE
                /* This node contains other nodes and we need to open it.*/
                no = nop->s.suns[0];
            }
        }
        return ninteractions;
    }

    /* Add the acceleration from a node or particle to the output structure,
     * computing the short-range kernel and softening.*/
    MYCUDAFN void apply_accn(GravTreeResult * output, const double dx[3], const double r2, const double mass, const double cellsize, const GravShortTable& gravtab, const double h)
    {
        const double r = sqrt(r2);
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

        /* 0 means "r < table length". 1 means "r is outside table", tree acceleration is zero. */
        if(0 == gravtab.apply_short_range_window(r, &fac, &facpot, cellsize)) {
            for(int i = 0; i < 3; i++)
                output->Acc[i] += dx[i] * fac;
            output->Potential += facpot;
        }
    }
};

/* Note the NgbIter class is never used for the GravTree, so the final template argument has no effect. */
class GravTopTreeWalk : public TopTreeWalk<GravTreeQuery, GravTreeParams, NGB_TREEFIND_ASYMMETRIC> {
    using TopTreeWalk::TopTreeWalk;
    public:
    /*! Find exports. The tricky part of this routine is that tree nodes that would normally be discarded without opening must not be exported.
     */
    MYCUDAFN int toptree_visit(const int target, const GravTreeQuery& input, const GravTreeParams& priv, data_index * const DataIndexTable, const size_t BunchSize)
    {
        //message(1, "Starting toptree visit for target %d Nexport %ld\n", target, Nexport);
        /* Reset the exported particles for this target. */
        int64_t NThisParticleExport = 0;
        /*Tree-opening constants*/
        const double rcut = priv.Rcut;
        const double rcut2 = rcut * rcut;
        const double aold = priv.ErrTolForceAcc * input.OldAcc;

        /*Input particle data*/
        const double * inpos = input.Pos;

        /* For a top tree walk we always start from the first element of the tree. */
        int no = input.NodeList[0];
        while(no >= 0)
        {
            /* The tree always walks internal nodes*/
            const NODE * const nop = &Nodes[no];

            double dx[3];
            for(int i = 0; i < 3; i++)
                dx[i] = NEAREST(nop->mom.cofm[i] - inpos[i], priv.BoxSize);
            const double r2 = dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2];

            /* Discard this node, move to sibling*/
            if (shall_we_discard_node(nop->len, r2, nop->center, inpos, priv.BoxSize, rcut, rcut2) ||
            /* This node accelerates the particle directly, and is not opened, move to sibling.*/
            !shall_we_open_node(nop->len, nop->mom.mass, r2, nop->center, inpos, priv.BoxSize, aold, priv.TreeUseBH, priv.BHOpeningAngle2) )
            {
                no = nop->sibling;
                /* Don't add this node*/
                continue;
            }

            /* A pseudo particle that would normally be opened should now be exported. */
            if(nop->f.ChildType == PSEUDO_NODE_TYPE) {
                /* Export the pseudo particle*/
                if(!DataIndexTable)
                    NThisParticleExport = export_count(nop->s.suns[0], NThisParticleExport);
                else {
                    NThisParticleExport = export_particle(nop->s.suns[0], target, NThisParticleExport, DataIndexTable, BunchSize);
                    /* Exit the loop as we cannot export more particles.*/
                    if(NThisParticleExport < 0)
                        break;
                }
                /* Move sideways*/
                no = nop->sibling;
                continue;
            }
            /* Only walk toptree nodes here, move to sibling if we found a toptree leaf.
             * This is a local toptree leaf, which would normally be opened.
             */
            if((nop->f.TopLevel && !nop->f.InternalTopLevel))
            {
                no = nop->sibling;
                continue;
            }
            /* Open the toptree node. */
            no = nop->s.suns[0];
        }
    #if defined DEBUG && not defined __CUDACC__
        if(NThisParticleExport > 1000)
            message(5, "%ld exports for particle %d! Odd.\n", NThisParticleExport, target);
    #endif
        /* If we filled up, this partial toptree walk will be discarded and the toptree loop exited.*/
        //message(5, "Export buffer full for particle %d with %ld (%lu) exports\n", target, NThisParticleExport, Nexport);
        return NThisParticleExport;
    }
};

#ifdef USE_CUDA
void
grav_short_tree_cuda(const ActiveParticles * act, ForceTree * tree, GravTreeParams * priv, GravTreeOutput * output, particle_data * const parts, const size_t MaxExportBufferBytes, MPI_Comm comm);
#endif

#endif
