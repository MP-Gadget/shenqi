#include <mpi.h>
#include <stdlib.h>
#include <math.h>

#include "libgadget/types.h"
#include "utils/endrun.h"
#include "utils/mymalloc.h"

#include "forcetree.h"
#include "treewalk2.h"
#include "localtreewalk2.h"
#include "timestep.h"
#include "walltime.h"
#include "partmanager.h"
#include "gravity.h"

/*! \file gravtree.c
 *  \brief main driver routines for gravitational (short-range) force computation
 *
 *  This file contains the code for the gravitational force computation by
 *  means of the tree algorithm. To this end, a tree force is computed for all
 *  active local particles, and particles are exported to other processors if
 *  needed, where they can receive additional force contributions. If the
 *  TreePM algorithm is enabled, the force computed will only be the
 *  short-range part.
 */
 class GravTreePriv : public ParamTypeBase {
     public:
     /* Size of a PM cell, in internal units. Box / Nmesh */
     const double cellsize;
     /* How many PM cells do we go
      * before we stop calculating the tree?*/
     const double Rcut;
     /* Newton's constant in internal units*/
     const double G;
     const inttime_t Ti_Current;
     /* Matter density in internal units.
      * rho_0 = Omega0 * rho_crit
      * rho_crit = 3 H^2 /(8 pi G).
      * This is (rho_0)^(1/3) ,
      * Note: should account for
      * massive neutrinos, but doesn't. */
     const double cbrtrho0;
     /* Pointer to the place to store accelerations*/
     MyFloat (*Accel)[3];
     /* If this is true, we have all particles and need to update the gravitational potential */
     const bool update_potential;
     int accelstorealloc;

     GravTreePriv(const double Rcut, const inttime_t i_Ti_Current, const double rho0, const PetaPM * const pm, const double BoxSize, MyFloat (* AccelStore)[3], const bool i_update_potential, const int64_t NumPart):
     ParamTypeBase(BoxSize), cellsize(BoxSize / pm->Nmesh), Rcut(Rcut * pm->Asmth * cellsize), G(pm->G), Ti_Current(i_Ti_Current),
     cbrtrho0(pow(rho0, 1.0 / 3)), Accel(AccelStore), update_potential(i_update_potential)
     {
         accelstorealloc = 0;
         if(!AccelStore) {
             Accel = (MyFloat (*) [3]) mymalloc2("GravAccel", NumPart * sizeof(Accel[0]));
             accelstorealloc = 1;
         }
     }

     ~GravTreePriv()
     { if(accelstorealloc)
         myfree(Accel);
     }
 };

 #define GRAV_GET_PRIV(tw) ((struct GravShortPriv *) ((tw)->priv))

 /*Compute the absolute magnitude of the acceleration for a particle.*/
 MyFloat
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

 class GravTreeQuery : public TreeWalkQueryBase<GravTreePriv> {
     public:
    const MyFloat OldAcc;
    GravTreeQuery(const particle_data& particle, const int * const i_NodeList, const int firstnode, const GravTreePriv& priv) :
    TreeWalkQueryBase(particle, i_NodeList, firstnode, priv), OldAcc(grav_get_abs_accel(particle, priv.G)) {}
 };

class GravTreeResult : public TreeWalkResultBase<GravTreePriv> {
    public:
    MyFloat Acc[3];
    MyFloat Potential;
    void reduce(const int place, const TreeWalkReduceMode mode, const GravTreePriv& priv, struct particle_data * const parts)
    {
        TreeWalkResultBase::reduce(place, mode, priv, parts);
        TREEWALK_REDUCE(priv.Accel[place][0], Acc[0]);
        TREEWALK_REDUCE(priv.Accel[place][1], Acc[1]);
        TREEWALK_REDUCE(priv.Accel[place][2], Acc[2]);
        if(priv.update_potential)
            TREEWALK_REDUCE(Part[place].Potential, Potential);
    }

    /* Add the acceleration from a node or particle to the output structure,
     * computing the short-range kernel and softening.*/
    void apply_accn(const double dx[3], const double r2, const double mass, const double cellsize)
    {
        const double r = sqrt(r2);

        const double h = FORCE_SOFTENING();
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

        if(0 == grav_apply_short_range_window(r, &fac, &facpot, cellsize)) {
            int i;
            for(i = 0; i < 3; i++)
                Acc[i] += dx[i] * fac;
            Potential += facpot;
        }
    }
};

static struct gravshort_tree_params TreeParams;
/*Softening length*/
static double GravitySoftening;

/* gravitational softening length
 * (given in terms of an `equivalent' Plummer softening length)
 */
double FORCE_SOFTENING(void)
{
    /* Force is Newtonian beyond this.*/
    return 2.8 * GravitySoftening;
}

/*! Sets the (comoving) softening length, converting from units of the mean separation to comoving internal units. */
void
gravshort_set_softenings(double MeanSeparation)
{
    GravitySoftening = TreeParams.FractionalGravitySoftening * MeanSeparation;
    /* 0: Gas is collisional */
    message(0, "GravitySoftening = %g\n", GravitySoftening);
}

/*This is a helper for the tests*/
void set_gravshort_treepar(struct gravshort_tree_params tree_params)
{
    TreeParams = tree_params;
}

struct gravshort_tree_params get_gravshort_treepar(void)
{
    return TreeParams;
}

/* Sets up the module*/
void
set_gravshort_tree_params(ParameterSet * ps)
{
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    if(ThisTask == 0) {
        TreeParams.BHOpeningAngle = param_get_double(ps, "BHOpeningAngle");
        TreeParams.ErrTolForceAcc = param_get_double(ps, "ErrTolForceAcc");
        TreeParams.TreeUseBH= param_get_int(ps, "TreeUseBH");
        TreeParams.Rcut = param_get_double(ps, "TreeRcut");
        TreeParams.FractionalGravitySoftening = param_get_double(ps, "GravitySoftening");
        TreeParams.MaxBHOpeningAngle = param_get_double(ps, "MaxBHOpeningAngle");
    }
    MPI_Bcast(&TreeParams, sizeof(struct gravshort_tree_params), MPI_BYTE, 0, MPI_COMM_WORLD);
}

/* Check whether a node should be discarded completely, its contents not contributing
 * to the acceleration. This happens if the node is further away than the short-range force cutoff.
 * Return 1 if the node should be discarded, 0 otherwise. */
static int
shall_we_discard_node(const double len, const double r2, const double center[3], const double inpos[3], const double BoxSize, const double rcut, const double rcut2)
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

/* This function tests whether a node shall be opened (ie, should the next node be .
 * If it should be discarded, 0 is returned.
 * If it should be used, 1 is returned, otherwise zero is returned. */
static int
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

class GravLocalTreeWalk : public LocalTreeWalk<TreeWalkNgbIterBase<GravTreeQuery, GravTreeResult, GravTreePriv>, GravTreeQuery, GravTreeResult, GravTreePriv> {
    /*! In the TreePM algorithm, the tree is walked only locally around the
     *  target coordinate.  Tree nodes that fall outside a box of half
     *  side-length Rcut= RCUT*ASMTH*MeshSize can be discarded. The short-range
     *  potential is modified by a complementary error function, multiplied
     *  with the Newtonian form. The resulting short-range suppression compared
     *  to the Newtonian force is tabulated, because looking up from this table
     *  is faster than recomputing the corresponding factor, despite the
     *  memory-access penalty (which reduces cache performance) incurred by the
     *  table.
     */
    int visit(const GravTreeQuery& input, GravTreeResult * output, const GravTreePriv& priv, const struct particle_data * const parts)
    {
        const double BoxSize = tree->BoxSize;

        /*Tree-opening constants*/
        const double cellsize = priv.cellsize;
        const double rcut = priv.Rcut;
        const double rcut2 = rcut * rcut;
        const double aold = TreeParams.ErrTolForceAcc * input.OldAcc;
        const int TreeUseBH = TreeParams.TreeUseBH;
        double BHOpeningAngle2 = TreeParams.BHOpeningAngle * TreeParams.BHOpeningAngle;
        /* Enforce a maximum opening angle even for relative acceleration criterion, to avoid
         * pathological cases. Default value is 0.9, from Volker Springel.*/
        if(TreeUseBH == 0)
            BHOpeningAngle2 = TreeParams.MaxBHOpeningAngle * TreeParams.MaxBHOpeningAngle;

        /*Input particle data*/
        const double * inpos = input.Pos;

        /*Start the tree walk*/
        int listindex, ninteractions=0;

        /* Primary treewalk only ever has one nodelist entry*/
        for(listindex = 0; listindex < NODELISTLENGTH; listindex++)
        {
            int numcand = 0;
            /* Use the next node in the node list if we are doing a secondary walk.
             * For a primary walk the node list only ever contains one node. */
            int no = input.NodeList[listindex];
            int startno = no;
            if(no < 0)
                break;

            while(no >= 0)
            {
                /* The tree always walks internal nodes*/
                struct NODE *nop = &tree->Nodes[no];

                if(mode == TREEWALK_GHOSTS && nop->f.TopLevel && no != startno)  /* we reached a top-level node again, which means that we are done with the branch */
                    break;

                int i;
                double dx[3];
                for(i = 0; i < 3; i++)
                    dx[i] = NEAREST(nop->mom.cofm[i] - inpos[i], BoxSize);
                const double r2 = dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2];

                /* Discard this node, move to sibling*/
                if(shall_we_discard_node(nop->len, r2, nop->center, inpos, BoxSize, rcut, rcut2))
                {
                    no = nop->sibling;
                    /* Don't add this node*/
                    continue;
                }

                /* This node accelerates the particle directly, and is not opened.*/
                int open_node = shall_we_open_node(nop->len, nop->mom.mass, r2, nop->center, inpos, BoxSize, aold, TreeUseBH, BHOpeningAngle2);

                if(!open_node)
                {
                    /* ok, node can be used */
                    no = nop->sibling;
                    if(mode != TREEWALK_TOPTREE) {
                        /* Compute the acceleration and apply it to the output structure*/
                        output->apply_accn(dx, r2, nop->mom.mass, cellsize);
                    }
                    continue;
                }

                if(mode == TREEWALK_TOPTREE) {
                    if(nop->f.ChildType == PSEUDO_NODE_TYPE) {
                        /* Export the pseudo particle*/
                        if(-1 == export_particle(nop->s.suns[0]))
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
                            ngblist[numcand++] = pp;
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
            int i;
            for(i = 0; i < numcand; i++)
            {
                int pp = ngblist[i];
                double dx[3];
                int j;
                for(j = 0; j < 3; j++)
                    dx[j] = NEAREST(Part[pp].Pos[j] - inpos[j], BoxSize);
                const double r2 = dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2];
                /* Compute the acceleration and apply it to the output structure*/
                output->apply_accn(dx, r2, Part[pp].Mass, cellsize);
            }
            ninteractions = numcand;
        }
        treewalk_add_counters(ninteractions);
        return 1;
    }
};

class GravTreeWalk : public TreeWalk <GravTreeQuery, GravTreeResult, GravLocalTreeWalk, GravTreePriv> {
    protected:
    /**
    * Postprocess - finalize quantities after tree walk completes.
    * Override to normalize results, compute derived quantities, etc.
    *
    * @param i Particle index
    */
    void postprocess(const int i, particle_data * const part)
    {
        const double G = priv.G;
        priv.Accel[i][0] *= G;
        priv.Accel[i][1] *= G;
        priv.Accel[i][2] *= G;

        if(priv.update_potential) {
            /* On a PM step, update the stored full tree grav accel for the next PM step.
            * Needs to be done here so internal treewalk iterations don't get a partial acceleration.*/
            Part[i].FullTreeGravAccel[0] = priv.Accel[i][0];
            Part[i].FullTreeGravAccel[1] = priv.Accel[i][1];
            Part[i].FullTreeGravAccel[2] = priv.Accel[i][2];
            /* calculate the potential */
            Part[i].Potential += Part[i].Mass / (FORCE_SOFTENING() / 2.8);
            /* remove self-potential */
            Part[i].Potential -= 2.8372975 * pow(Part[i].Mass, 2.0 / 3) * priv.cbrtrho0;
            Part[i].Potential *= G;
        }
    }
    public:
        GravTreeWalk(const char * const name, const ForceTree * const tree, const GravTreePriv& priv)
            : TreeWalk(name, tree, priv, false) {
                if(!tree->moments_computed_flag)
                    endrun(2, "Gravtree called before tree moments computed!\n");
            };
};

/*! This function computes the gravitational forces for all active particles from all particles in the tree.
 * Particles are only exported to other processors when really
 *  needed, thereby allowing a good use of the communication buffer.
 *  NeutrinoTracer = All.HybridNeutrinosOn && (atime <= All.HybridNuPartTime);
 *  rho0 = CP.Omega0 * 3 * CP.Hubble * CP.Hubble / (8 * M_PI * G)
 *  ActiveParticle should contain only gravitationally active particles.
 *  If this tree contains all particles, as specified by the full_particle_tree_flag, we calculate the short-
 * range gravitational potential and update the fulltreegravaccel. Note that in practice
 * for hierarchical gravity only active particles are in the tree and so this is
 * only true on PM steps where all particles are active.
 */
void
grav_short_tree2(const ActiveParticles * act, PetaPM * pm, ForceTree * tree, MyFloat (* AccelStore)[3], double rho0, inttime_t Ti_Current)
{

    GravTreePriv priv(TreeParams.Rcut, Ti_Current, rho0, pm, tree->BoxSize, AccelStore, tree->full_particle_tree_flag, PartManager->NumPart);
    GravTreeWalk tw("GRAVTREE", tree, priv);
    tw.run(act->ActiveParticle, act->NumActiveParticle, PartManager->Base);

    /* Now the force computation is finished */
    /*  gather some diagnostic information */

    double timetree = tw.timecomp0 + tw.timecomp1 + tw.timecomp2 + tw.timecomp3;
    walltime_add("/Tree/WalkTop", tw.timecomp0);
    walltime_add("/Tree/WalkPrim", tw.timecomp1);
    walltime_add("/Tree/WalkSec", tw.timecomp2);
    walltime_add("/Tree/Reduce", tw.timecommsumm);
    walltime_add("/Tree/PostPre", tw.timecomp3);
    walltime_add("/Tree/Wait", tw.timewait1);

    double timeall = walltime_measure(WALLTIME_IGNORE);

    walltime_add("/Tree/Misc", timeall - (timetree + tw.timewait1 + tw.timecommsumm));

    tw.print_stats();

    /* TreeUseBH > 1 means use the BH criterion on the initial timestep only,
     * avoiding the fully open O(N^2) case.*/
    if(TreeParams.TreeUseBH > 1)
        TreeParams.TreeUseBH = 0;
}
