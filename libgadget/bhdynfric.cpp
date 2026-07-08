#include <math.h>
#include <execution>
#include <algorithm>
#include <ranges>
#include "timestep.h"
#include "treewalk2.h"
#include "densitykernel.hpp"
#include "bhdynfric.h"
#include "walltime.h"
#include "utils/system.h"
#include "utils/endrun.h"

#define BHPOTVALUEINIT 1.0e29

static struct BlackholeDynFricParams
{
    int BH_DynFrictionMethod;/*0 for off; 1 for Star Only; 2 for DM+Star;*/
    int BH_DFBoostFactor; /*Optional boost factor for DF */
    double BH_DFbmax; /* the maximum impact range, in physical unit of kpc. */
    int BlackHoleRepositionEnabled; /* If true, enable repositioning the BH to the potential minimum. If false, do dynamic friction.*/
} blackhole_dynfric_params;

/*Set the parameters of the BH module*/
void set_blackhole_dynfric_params(ParameterSet * ps)
{
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    if(ThisTask == 0) {
        blackhole_dynfric_params.BH_DynFrictionMethod = param_get_int(ps, "BH_DynFrictionMethod");
        blackhole_dynfric_params.BH_DFBoostFactor = param_get_int(ps, "BH_DFBoostFactor");
        blackhole_dynfric_params.BH_DFbmax = param_get_double(ps, "BH_DFbmax");
        blackhole_dynfric_params.BlackHoleRepositionEnabled = param_get_int(ps, "BlackHoleRepositionEnabled");
        if(blackhole_dynfric_params.BH_DynFrictionMethod > 2)
            message(0, "BH_DynFrictionMethod %d no longer supported: only stars and DM will cause dynamical friction\n", blackhole_dynfric_params.BH_DynFrictionMethod);
    }
    MPI_Bcast(&blackhole_dynfric_params, sizeof(struct BlackholeDynFricParams), MPI_BYTE, 0, MPI_COMM_WORLD);
}

int
BHGetRepositionEnabled(void)
{
    return blackhole_dynfric_params.BlackHoleRepositionEnabled;
}

/* Parameters used in the dynamic friction treewalk. */
class BHDynFricPriv : public ParamTypeBase {
    public:
    /* Time factors*/
    KickFactorData kf;
    inttime_t Ti_Current; /* current time*/
    int BH_DynFrictionMethod;
    BHDynFricPriv(double BoxSize, KickFactorData& kf_i, inttime_t Ti_Current_i, int BH_DynFrictionMethod_i): ParamTypeBase(BoxSize), kf(kf_i), Ti_Current(Ti_Current_i), BH_DynFrictionMethod(BH_DynFrictionMethod_i) {};
};

/* Computes the BH velocity dispersion for kinetic feedback*/
class BHDynFricOutput {
public:
    size_t ZeroDF = 0; // Counter for zero density BHs
    double ZeroDFMass = 0; /* Total mass of BHs with zero DF density*/
    bh_particle_data * BhParts;
    bool BlackHoleRepositionEnabled;

    BHDynFricOutput(bool BlackHoleReposition, slots_manager_type * slotsmanager): BhParts(slotsmanager->bh_slot()), BlackHoleRepositionEnabled(BlackHoleReposition) {}

    void
    postprocess(int n, particle_data * const parts, const BHDynFricPriv * priv)
    {
        bh_particle_data& Bhpart = BhParts[parts[n].PI];
        if(Bhpart.DF_SurroundingDensity > 0){
            /* normalize velocity/dispersion */
            Bhpart.DF_SurroundingRmsVel /= Bhpart.DF_SurroundingDensity;
            Bhpart.DF_SurroundingRmsVel = sqrt(Bhpart.DF_SurroundingRmsVel);
            for(int j = 0; j < 3; j++)
                Bhpart.DF_SurroundingVel[j] /= Bhpart.DF_SurroundingDensity;
        }
        else {
            #pragma omp atomic update
            ZeroDF++;
            #pragma omp atomic update
            ZeroDFMass += Bhpart.Mass;
        }
    }
};

class BHDynFricQuery : public TreeWalkQueryBase<BHDynFricPriv>
{
    public:
    MyFloat Hsml;

    MYCUDAFN BHDynFricQuery(const particle_data& particle, const int * const i_NodeList, const int firstnode, const BHDynFricPriv& priv):
    TreeWalkQueryBase<BHDynFricPriv>(particle, i_NodeList, firstnode, priv), Hsml(particle.Hsml) {}
};

class BHReposResult : public TreeWalkResultBase<BHDynFricQuery, BHDynFricOutput> {
    public:
    /* Minimum potential for diagnostics*/
    MyFloat BH_MinPotPos[3] = {-1,-1,-1};
    MyFloat BH_MinPotVel[3] = {0,0,0};
    MyFloat BH_MinPot = BHPOTVALUEINIT;

    MYCUDAFN BHReposResult(const BHDynFricQuery& query):
        TreeWalkResultBase<BHDynFricQuery, BHDynFricOutput>(query) {}

    template<TreeWalkReduceMode mode>
    MYCUDAFN void reduce(int place, const BHDynFricOutput * output, struct particle_data * const parts)
    {
        bh_particle_data& BhPart = output->BhParts[parts[place].PI];

        if(BhPart.MinPot > BH_MinPot)
        {
            BhPart.JumpToMinPot = output->BlackHoleRepositionEnabled;
            BhPart.MinPot = BH_MinPot;
            for(int k = 0; k < 3; k++) {
                /* Movement occurs in drift.c */
                BhPart.MinPotPos[k] = BH_MinPotPos[k];
                BhPart.MinPotVel[k] = BH_MinPotVel[k];
            }
        }
    }
};

class BHDynFricResult : public BHReposResult {
    public:
    MyFloat SurroundingVel[3];
    MyFloat SurroundingDensity;
    MyFloat SurroundingRmsVel;

    template<TreeWalkReduceMode mode>
    MYCUDAFN void reduce(int place, const BHDynFricOutput * output, struct particle_data * const parts) {
        bh_particle_data& BhPart = output->BhParts[parts[place].PI];
        TREEWALK_REDUCE(BhPart.DF_SurroundingDensity, SurroundingDensity);
        for(int j = 0; j < 3; j++)
            TREEWALK_REDUCE(BhPart.DF_SurroundingVel[j], SurroundingVel[j]);
        TREEWALK_REDUCE(BhPart.DF_SurroundingRmsVel, SurroundingRmsVel);
        /* Find minimum potential*/
        static_cast<BHReposResult *>(this)->reduce<mode>(place, output, parts);
    }
};

/* We want all particles here*/
template <typename DensityKernel>
class BHReposLocalTreeWalk: public LocalNgbTreeWalk<BHReposLocalTreeWalk<DensityKernel>, BHDynFricQuery, BHReposResult, BHDynFricPriv, NGB_TREEFIND_ASYMMETRIC, ALLMASK>
{
    public:
    DensityKernel dynfric_kernel;

    MYCUDAFN BHReposLocalTreeWalk(const NODE * const Nodes, const BHDynFricQuery& input):
    LocalNgbTreeWalk<BHReposLocalTreeWalk<DensityKernel>, BHDynFricQuery, BHReposResult, BHDynFricPriv, NGB_TREEFIND_ASYMMETRIC, ALLMASK>(Nodes, input), dynfric_kernel(input.Hsml)
    { }

    /*! This function is the 'core' of the SPH force computation. A target
     *  particle is specified which may either be local, or reside in the
     *  communication buffer.     *
     * @param input  Query data
     * @param output Result accumulator
     */
    MYCUDAFN void ngbiter(const BHDynFricQuery& input, const particle_data& particle, BHReposResult * output, const BHDynFricPriv& priv)
    {
        double dist[3];
        double r2 = this->get_distance(input, particle, priv.BoxSize, dist);

        /* Find the black hole potential minimum. */
        if(r2 < dynfric_kernel.H * dynfric_kernel.H && particle.Potential < output->BH_MinPot)
        {
            output->BH_MinPot = particle.Potential;
            for(int d = 0; d < 3; d++) {
                output->BH_MinPotPos[d] = particle.Pos[d];
                output->BH_MinPotVel[d] = particle.Vel[d];
            }
        }
    }
};

template <typename DensityKernel, int treemask>
class BHDynFricLocalTreeWalk: public LocalNgbTreeWalk<BHDynFricLocalTreeWalk<DensityKernel, treemask>, BHDynFricQuery, BHDynFricResult, BHDynFricPriv, NGB_TREEFIND_ASYMMETRIC, treemask>
{
    public:
    DensityKernel dynfric_kernel;

    MYCUDAFN BHDynFricLocalTreeWalk(const NODE * const Nodes, const BHDynFricQuery& input):
    LocalNgbTreeWalk<BHDynFricLocalTreeWalk<DensityKernel, treemask>, BHDynFricQuery, BHDynFricResult, BHDynFricPriv, NGB_TREEFIND_ASYMMETRIC, treemask>(Nodes, input), dynfric_kernel(input.Hsml)
    { }

    /*! This function is the 'core' of the SPH force computation. A target
     *  particle is specified which may either be local, or reside in the
     *  communication buffer.     *
     * @param input  Query data
     * @param output Result accumulator
     */
    MYCUDAFN void ngbiter(const BHDynFricQuery& input, const particle_data& particle, BHDynFricResult * output, const BHDynFricPriv& priv)
    {
        double dist[3];
        double r2 = this->get_distance(input, particle, priv.BoxSize, dist);

        BHReposResult * potoutput = static_cast<BHReposResult *>(output);
        if(r2 >= dynfric_kernel.H * dynfric_kernel.H)
            return;

        /* Find the black hole potential minimum. */
        if(particle.Potential < potoutput->BH_MinPot)
        {
            potoutput->BH_MinPot = particle.Potential;
            for(int d = 0; d < 3; d++) {
                potoutput->BH_MinPotPos[d] = particle.Pos[d];
                potoutput->BH_MinPotVel[d] = particle.Vel[d];
            }
        }

        /* Collect Star/+DM/+Gas density/velocity for DF computation */
        if(particle.Type == 4 || (particle.Type == 1 && priv.BH_DynFrictionMethod > 1)) {
            double u = sqrt(r2) / dynfric_kernel.H;
            double wk = dynfric_kernel.wk(u);
            output->SurroundingDensity += (particle.Mass * wk);
            MyFloat VelPred[3];
            priv.kf.DM_VelPred(particle, VelPred);
            for (int k = 0; k < 3; k++){
                output->SurroundingVel[k] += (particle.Mass * wk * VelPred[k]);
                output->SurroundingRmsVel += (particle.Mass * wk * pow(VelPred[k], 2));
            }
        }
    }
};

class BHDynFricTopTreeWalk: public TopTreeWalk<BHDynFricQuery, BHDynFricPriv, NGB_TREEFIND_ASYMMETRIC> { using TopTreeWalk::TopTreeWalk; };

class BHReposTreeWalkCubic: public TreeWalk<BHReposTreeWalkCubic, BHDynFricQuery, BHReposResult, BHReposLocalTreeWalk<CubicDensityKernel>, BHDynFricTopTreeWalk, BHDynFricPriv, BHDynFricOutput> {
    public:
    using TreeWalk::TreeWalk;
};

class BHReposTreeWalkQuartic: public TreeWalk<BHReposTreeWalkQuartic, BHDynFricQuery, BHReposResult, BHReposLocalTreeWalk<QuarticDensityKernel>, BHDynFricTopTreeWalk, BHDynFricPriv, BHDynFricOutput> {
    public:
    using TreeWalk::TreeWalk;
};

class BHReposTreeWalkQuintic: public TreeWalk<BHReposTreeWalkQuintic, BHDynFricQuery, BHReposResult, BHReposLocalTreeWalk<QuinticDensityKernel>, BHDynFricTopTreeWalk, BHDynFricPriv, BHDynFricOutput> {
    public:
    using TreeWalk::TreeWalk;
};

int
blackhole_dynfric_treemask(void)
{
    /* dynamical friction uses: stars, DM if BH_DynFrictionMethod > 1, gas if BH_DynFrictionMethod  == 3.
     * The BH do not contribute dynamic friction but are here so the potential minimum is updated. */
    int treemask = STARMASK + BHMASK;
    /* Don't necessarily need dark matter */
    if(blackhole_dynfric_params.BH_DynFrictionMethod > 1)
        treemask += DMMASK;
    /* For repositioning we want all particles*/
    if(blackhole_dynfric_params.BlackHoleRepositionEnabled)
        treemask = ALLMASK;
    return treemask;
}

/* The template specialisations for the dynamical friction treewalks also need a treemask.
 * dynamical friction uses: stars, DM if BH_DynFrictionMethod > 1.
 * The BH do not contribute dynamic friction but are here so the potential minimum is updated.
 *  treemask = STARMASK + BHMASK
 *  if(blackhole_dynfric_params.BH_DynFrictionMethod > 1)
    treemask += DMMASK;
*/
class BHDynFricTreeWalkCubic: public TreeWalk<BHDynFricTreeWalkCubic, BHDynFricQuery, BHDynFricResult, BHDynFricLocalTreeWalk<CubicDensityKernel, STARMASK + BHMASK>, BHDynFricTopTreeWalk, BHDynFricPriv, BHDynFricOutput> {
    public:
    using TreeWalk::TreeWalk;
};

class BHDynFricTreeWalkCubicDM: public TreeWalk<BHDynFricTreeWalkCubicDM, BHDynFricQuery, BHDynFricResult, BHDynFricLocalTreeWalk<CubicDensityKernel, STARMASK + BHMASK + DMMASK>, BHDynFricTopTreeWalk, BHDynFricPriv, BHDynFricOutput> {
    public:
    using TreeWalk::TreeWalk;
};

class BHDynFricTreeWalkQuartic: public TreeWalk<BHDynFricTreeWalkQuartic, BHDynFricQuery, BHDynFricResult, BHDynFricLocalTreeWalk<QuarticDensityKernel, STARMASK + BHMASK>, BHDynFricTopTreeWalk, BHDynFricPriv, BHDynFricOutput> {
    public:
    using TreeWalk::TreeWalk;
};

class BHDynFricTreeWalkQuarticDM: public TreeWalk<BHDynFricTreeWalkQuarticDM, BHDynFricQuery, BHDynFricResult, BHDynFricLocalTreeWalk<QuarticDensityKernel, STARMASK + BHMASK + DMMASK>, BHDynFricTopTreeWalk, BHDynFricPriv, BHDynFricOutput> {
    public:
    using TreeWalk::TreeWalk;
};

class BHDynFricTreeWalkQuintic: public TreeWalk<BHDynFricTreeWalkQuintic, BHDynFricQuery, BHDynFricResult, BHDynFricLocalTreeWalk<QuinticDensityKernel, STARMASK + BHMASK>, BHDynFricTopTreeWalk, BHDynFricPriv, BHDynFricOutput> {
    public:
    using TreeWalk::TreeWalk;
};

class BHDynFricTreeWalkQuinticDM: public TreeWalk<BHDynFricTreeWalkQuinticDM, BHDynFricQuery, BHDynFricResult, BHDynFricLocalTreeWalk<QuinticDensityKernel, STARMASK + BHMASK + DMMASK>, BHDynFricTopTreeWalk, BHDynFricPriv, BHDynFricOutput> {
    public:
    using TreeWalk::TreeWalk;
};

void blackhole_init_potential(int * ActiveBlackHoles, const int64_t NumActiveBlackHoles, struct particle_data * const parts)
{
    /* Initialise the minimum potential*/
    for(int i = 0; i < NumActiveBlackHoles; i++) {
        int n = ActiveBlackHoles ? ActiveBlackHoles[i] : i;
        /* Note that the potential is only updated when it is from all particles.
        * In particular this means that it is not updated for hierarchical gravity
        * when the number of active particles is less than the total number of particles
        * (because then the tree does not contain all forces). */
        BHP(n).MinPot = parts[n].Potential;
        for(int j = 0; j < 3; j++) {
            BHP(n).MinPotPos[j] = parts[n].Pos[j];
        }
    }
}

/* Simple treewalk that just finds the local potential minimum for BH repositioning.*/
void
blackhole_minpot(int * ActiveBlackHoles, const int64_t NumActiveBlackHoles, DomainDecomp * ddecomp, BHDynFricPriv& priv, MPI_Comm comm)
{
    /* Repositioning uses all particles: in practice it will usually be stars, gas or BH.*/
    ForceTree tree[1] = {0};
    message(0, "Building tree with all particles for repositioning\n");
    force_tree_rebuild_mask(tree, ddecomp, ALLMASK, "");
    walltime_measure("/BH/BuildRepos");

    blackhole_init_potential(ActiveBlackHoles, NumActiveBlackHoles, Part);

    BHDynFricOutput output(blackhole_dynfric_params.BlackHoleRepositionEnabled, SlotsManager);

    switch(GetDensityKernelType()) {
        case DENSITY_KERNEL_CUBIC_SPLINE:
            {
                BHReposTreeWalkCubic tw("BH_REPOS", tree, priv, &output);
                tw.run(ActiveBlackHoles, NumActiveBlackHoles, PartManager->Base, comm);
            }
            break;
        case DENSITY_KERNEL_QUARTIC_SPLINE:
            {
                BHReposTreeWalkQuartic tw("BH_REPOS", tree, priv, &output);
                tw.run(ActiveBlackHoles, NumActiveBlackHoles, PartManager->Base, comm);
            }
            break;
        default: //DENSITY_KERNEL_QUINTIC_SPLINE
            BHReposTreeWalkQuintic tw("BH_REPOS", tree, priv, &output);
            tw.run(ActiveBlackHoles, NumActiveBlackHoles, PartManager->Base, comm);
    }
    force_tree_free(tree);
    walltime_measure("/BH/Repos");
}


class DynFricActivePred {
public:
     const particle_data * parts;
     const bh_particle_data * bhparts;
     inttime_t Ti_Current;
     int * ActiveBlackHoles;
     MYCUDAFN bool operator()(int n) const {
         int i = ActiveBlackHoles[n];
         if(parts[i].IsGarbage || parts[i].Swallowed || parts[i].Type != 5)
             return false;
         return is_timebin_active(bhparts[parts[i].PI].TimeBinDynFric, Ti_Current);
     }
};


/* Returns total number of dynamic-friction active particles over all processors*/
int
blackhole_dynfric_num_active(int * ActiveBlackHoles, int64_t NumActiveBlackHoles, const inttime_t Ti_Current, int ** DynFricActive)
{
    if (blackhole_dynfric_params.BH_DynFrictionMethod == 0)
        return 0;

    *DynFricActive = mymanagedmalloc("DynFricActive", int, NumActiveBlackHoles);
    DynFricActivePred pred{PartManager->Base, SlotsManager->bh_slot(), Ti_Current, ActiveBlackHoles};
    /* This is the C++20 equivalent of a counting_iterator.*/
    auto iota = std::views::iota(0, static_cast<int>(NumActiveBlackHoles));
    auto end = std::copy_if(std::execution::par, iota.begin(), iota.end(), *DynFricActive, pred);
    int64_t nactive = end - *DynFricActive;
    return nactive;
}

void
blackhole_dynfric(int * ActiveBlackHoles, int64_t NumActiveBlackHoles, DomainDecomp * ddecomp, KickFactorData& kf, inttime_t Ti_Current, MPI_Comm comm)
{
    BHDynFricPriv priv(PartManager->BoxSize, kf, Ti_Current, blackhole_dynfric_params.BH_DynFrictionMethod);
    if (blackhole_dynfric_params.BH_DynFrictionMethod == 0) {
        /* If there is no dynamic friction, do repositioning, and
         * run a special walk to find the potential minimum.*/
        if(blackhole_dynfric_params.BlackHoleRepositionEnabled)
            blackhole_minpot(ActiveBlackHoles, NumActiveBlackHoles, ddecomp, priv, comm);
        return;
    }
    int * DynFricActive = NULL;
    int64_t ndynfricactive = blackhole_dynfric_num_active(ActiveBlackHoles, NumActiveBlackHoles, Ti_Current, &DynFricActive);
    int64_t totdynfric;
    MPI_Allreduce(&ndynfricactive, &totdynfric, 1, MPI_INT64, MPI_SUM, comm);
    if(!totdynfric) {
        myfree(DynFricActive);
        return;
    }

    blackhole_init_potential(DynFricActive, ndynfricactive, Part);

    /* dynamical friction uses: stars, DM if BH_DynFrictionMethod > 1 gas if BH_DynFrictionMethod  == 3.
     * The DM in dynamic friction and accretion doesn't really do anything, so could perhaps be removed from the treebuild later.*/
    ForceTree tree[1] = {0};
    int treemask = blackhole_dynfric_treemask();
    message(0, "Building dynamic friction tree with types %d\n", treemask);

    force_tree_rebuild_mask(tree, ddecomp, treemask, "");
    walltime_measure("/BH/BuildDF");

    BHDynFricOutput output(blackhole_dynfric_params.BlackHoleRepositionEnabled, SlotsManager);

    switch(GetDensityKernelType()) {
        case DENSITY_KERNEL_CUBIC_SPLINE:
            {
                if(blackhole_dynfric_params.BH_DynFrictionMethod > 1) {
                    BHDynFricTreeWalkCubicDM tw("BH_DYNFRIC_DM", tree, priv, &output);
                    tw.run_on_queue(DynFricActive, ndynfricactive, PartManager->Base, comm);
                }
                else {
                    BHDynFricTreeWalkCubic tw("BH_DYNFRIC", tree, priv, &output);
                    tw.run_on_queue(DynFricActive, ndynfricactive, PartManager->Base, comm);
                }
            }
            break;
        case DENSITY_KERNEL_QUARTIC_SPLINE:
            {
                if(blackhole_dynfric_params.BH_DynFrictionMethod > 1) {
                    BHDynFricTreeWalkQuarticDM tw("BH_DYNFRIC_DM", tree, priv, &output);
                    tw.run_on_queue(DynFricActive, ndynfricactive, PartManager->Base, comm);
                }
                else {
                    BHDynFricTreeWalkQuartic tw("BH_DYNFRIC", tree, priv, &output);
                    tw.run_on_queue(DynFricActive, ndynfricactive, PartManager->Base, comm);
                }
            }
            break;
        default: //DENSITY_KERNEL_QUINTIC_SPLINE
            if(blackhole_dynfric_params.BH_DynFrictionMethod > 1) {
                BHDynFricTreeWalkQuinticDM tw("BH_DYNFRIC_DM", tree, priv, &output);
                tw.run_on_queue(DynFricActive, ndynfricactive, PartManager->Base, comm);
            }
            else {
                BHDynFricTreeWalkQuintic tw("BH_DYNFRIC", tree, priv, &output);
                tw.run_on_queue(DynFricActive, ndynfricactive, PartManager->Base, comm);
            }
    }
    force_tree_free(tree);
    myfree(DynFricActive);

    size_t totalzerodf;
    double totalzeromass;
    MPI_Reduce(&output.ZeroDF, &totalzerodf, 1, MPI_INT64, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&output.ZeroDFMass, &totalzeromass, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if(totalzerodf > 0)
        message(0, "Dynamic Friction density is zero for %ld BHs avg mass %g.\n", totalzerodf, totalzeromass/totalzerodf);
}

/*************************************************************************************/
/* Compute the DF acceleration in the BH from stored quantities*/
static void
blackhole_compute_dfaccel(const int n, const double atime, const double Grav)
{
    int j;
    for(j = 0; j < 3; j++)
        BHP(n).DFAccel[j] = 0;
    /***********************************************************************************/
    /* This is Gizmo's implementation of dynamic friction                              */
    /* c.f. section 3.1 in http://www.tapir.caltech.edu/~phopkins/public/notes_blackholes.pdf */
    /* Compute dynamic friction accel when DF turned on                                */
    /* averaged value for colomb logarithm and integral over the distribution function */
    /* acc_friction = -4*pi*G^2 * Mbh * log(lambda) * rho * f_of_x * bhvel / |bhvel^3| */
    /*       f_of_x = [erf(x) - 2*x*exp(-x^2)/sqrt(pi)]                                */
    /*       lambda = b_max * v^2 / G / (M+m)                                          */
    /*        b_max = Size of system (e.g. Rvir)                                       */
    /*            v = Relative velocity of BH with respect to the environment          */
    /*            M = Mass of BH                                                       */
    /*            m = individual mass elements composing the large system (e.g. m<<M)  */
    /*            x = v/sqrt(2)/sigma                                                  */
    /*        sigma = width of the max. distr. of the host system                      */
    /*                (e.g. sigma = v_disp / 3                                         */
    if(BHP(n).DF_SurroundingDensity > 0){
        /* Calculate Coulumb Logarithm */
        double bhvel = 0;
        for(j = 0; j < 3; j++)
            bhvel += pow(Part[n].Vel[j] - BHP(n).DF_SurroundingVel[j], 2);
        bhvel = sqrt(bhvel);

        if(!isfinite(bhvel)) {
            endrun(6, "Bad bhvel %g vel %g %g %g DF vel %g %g %g id %ld n %d mass %g\n",
                bhvel, Part[n].Vel[0], Part[n].Vel[1], Part[n].Vel[2],
                BHP(n).DF_SurroundingVel[0], BHP(n).DF_SurroundingVel[1], BHP(n).DF_SurroundingVel[2],Part[n].ID, n, Part[n].Mass);
        }
        /* There is no parameter in physical unit, so I kept everything in code unit */

        double x = bhvel / sqrt(2) / (BHP(n).DF_SurroundingRmsVel / 3);
        /* First term is approximation of the error function */
        const double a_erf = 8 * (M_PI - 3) / (3 * M_PI * (4. - M_PI));
        double f_of_x = x / fabs(x) * sqrt(1 - exp(-x * x * (4 / M_PI + a_erf * x * x)
            / (1 + a_erf * x * x))) - 2 * x / sqrt(M_PI) * exp(-x * x);
        /* Floor at zero */
        if (f_of_x < 0)
            f_of_x = 0;

        double lambda = 1. + blackhole_dynfric_params.BH_DFbmax * pow((bhvel/atime),2) / Grav / Part[n].Mass;

        for(j = 0; j < 3; j++)
        {
            /* prevent DFAccel from exploding */
            if(bhvel > 0){
                BHP(n).DFAccel[j] = - 4. * M_PI * Grav * Grav * Part[n].Mass * BHP(n).DF_SurroundingDensity * log(lambda) * f_of_x * (Part[n].Vel[j] - BHP(n).DF_SurroundingVel[j]) / pow(bhvel, 3);
                BHP(n).DFAccel[j] *= atime;  // convert to code unit of acceleration
                BHP(n).DFAccel[j] *= blackhole_dynfric_params.BH_DFBoostFactor; // Add a boost factor
            }
        }
// #ifdef DEBUG
        // message(2,"x=%e, log(lambda)=%e, fof_x=%e, Mbh=%e, ratio=%e \n",
           // x,log(lambda),f_of_x,Part[n].Mass,BHP(n).DFAccel[0]/Part[n].FullTreeGravAccel[0]);
// #endif
    }
}

/* Compute the DF acceleration for all active black holes*/
void
blackhole_dfaccel(int * ActiveBlackHoles, size_t NumActiveBlackHoles, const double atime, const double GravInternal)
{
    if (blackhole_dynfric_params.BH_DynFrictionMethod == 0)
        return;

    #pragma omp parallel for
    for(size_t i = 0; i < NumActiveBlackHoles; i++) {
        int n = i;
        if(ActiveBlackHoles)
            n = ActiveBlackHoles[i];
        blackhole_compute_dfaccel(n, atime, GravInternal);
    }
}
