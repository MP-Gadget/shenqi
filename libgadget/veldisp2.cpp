#include <math.h>
#include <omp.h>
#include "libgadget/slotsmanager.h"
#include "veldisp.h"
#include "treewalk2.h"
#include "localtreewalk2.h"
#include "walltime.h"
#include "sfr_eff.h"
#include "density2.h"
#include "utils/endrun.h"
#include "utils/mymalloc.h"
#include "densitykernel.hpp"

/* For the wind hsml loop*/
#define NWINDHSML 5 /* Number of densities to evaluate for wind weight ngbiter*/
#define NUMDMNGB 40 /*Number of DM ngb to evaluate vel dispersion */
#define MAXDMDEVIATION 1

/* Computes the BH velocity dispersion for kinetic feedback*/
class BHVelDispPriv : public ParamTypeBase {
    public:
    /* Time factors*/
    KickFactorData kf;
    BHVelDispPriv(double BoxSize, KickFactorData& kf_i): ParamTypeBase(BoxSize), kf(kf_i) {};
};

class BHVelDispQuery : public TreeWalkQueryBase<BHVelDispPriv>
{
    public:
    MyFloat Hsml;
    MyFloat Vel[3];
    MYCUDAFN BHVelDispQuery(const particle_data& particle, const int * const i_NodeList, const int firstnode, const BHVelDispPriv& priv):
    TreeWalkQueryBase<BHVelDispPriv>(particle, i_NodeList, firstnode, priv), Hsml(particle.Hsml)
    {
            Vel[0] = particle.Vel[0];
            Vel[1] = particle.Vel[1];
            Vel[2] = particle.Vel[2];
    };
    static MYCUDAFN bool haswork(const particle_data& particle){
        /*Black hole not being swallowed*/
        if(!TreeWalkQueryBase::haswork(particle))
            return false;
        return (particle.Type == 5);
    }
};

class BHVelDispResult : public TreeWalkResultBase<BHVelDispQuery> {
    public:
    /* used for AGN kinetic feedback */
    MyFloat V2sumDM = 0;

    MyFloat V1sumDM[3] = {};
    MyFloat NumDM = 0;

    MYCUDAFN BHVelDispResult(const BHVelDispQuery& query):
        TreeWalkResultBase<BHVelDispQuery>(query) {}

    MYCUDAFN BHVelDispResult& operator +=(const BHVelDispResult& other)
    {
        static_cast<TreeWalkResultBase<BHVelDispQuery>&>(*this) += static_cast<const TreeWalkResultBase<BHVelDispQuery>& >(other);

        V2sumDM += other.V2sumDM;
        for(int i = 0; i < 3; i++) {
            V1sumDM[i] += other.V1sumDM[i];
        }
        NumDM += other.NumDM;
        return *this;
    }
};

/* Computes the BH velocity dispersion for kinetic feedback*/
class BHVelDispOutput {
    public:
        /* Pointers to the BH particle data array*/
        bh_particle_data * BhParts;

    BHVelDispOutput(slots_manager_type * SlotsManager) : BhParts(SlotsManager->bh_slot())
    {}

    ~BHVelDispOutput()
    {}

    void
    postprocess(int i, const BHVelDispResult& result, particle_data * const parts, const BHVelDispPriv * priv)
    {
        if (result.NumDM <= 0)
            return;
        double vdisp = result.V2sumDM/result.NumDM;
        for(int d = 0; d<3; d++)
            vdisp -= pow(result.V1sumDM[d]/result.NumDM,2);
        if(vdisp > 0)
            BhParts[parts[i].PI].VDisp = sqrt(vdisp / 3);
    }
};

template <typename DensityKernel>
class BHVelDispLocalTreeWalk: public LocalNgbTreeWalk<BHVelDispLocalTreeWalk<DensityKernel>, BHVelDispQuery, BHVelDispResult, BHVelDispPriv, NGB_TREEFIND_ASYMMETRIC, DMMASK>
{
    public:
    DensityKernel feedback_kernel;

    MYCUDAFN BHVelDispLocalTreeWalk(const NODE * const Nodes, const BHVelDispQuery& input):
    LocalNgbTreeWalk<BHVelDispLocalTreeWalk<DensityKernel>, BHVelDispQuery, BHVelDispResult, BHVelDispPriv, NGB_TREEFIND_ASYMMETRIC, DMMASK>(Nodes, input), feedback_kernel(input.Hsml)
    { }

    /*! This function is the 'core' of the SPH force computation. A target
     *  particle is specified which may either be local, or reside in the
     *  communication buffer.     *
     * @param input  Query data
     * @param output Result accumulator
     */
    MYCUDAFN void ngbiter(const BHVelDispQuery& input, const particle_data& particle, BHVelDispResult * output, const BHVelDispPriv& priv)
    {
        double dist[3];
        double r2 = this->get_distance(input, particle, priv.BoxSize, dist);

        /* Check we are within the density kernel*/
        if(r2 <= 0 || !(r2 < input.Hsml * input.Hsml))
            return;

        /* collect info for sigmaDM and Menc for kinetic feedback */
        output->NumDM += 1;
        MyFloat VelPred[3];
        priv.kf.DM_VelPred(particle, VelPred);
        for(int d = 0; d < 3; d++){
            double vel = VelPred[d] - input.Vel[d];
            output->V1sumDM[d] += vel;
            output->V2sumDM += vel * vel;
        }
    }
};

class BHVelDispTopTreeWalk: public TopTreeWalk<BHVelDispQuery, BHVelDispPriv, NGB_TREEFIND_ASYMMETRIC> { using TopTreeWalk::TopTreeWalk; };

class BHVelDispTreeWalkCubic: public TreeWalk<BHVelDispTreeWalkCubic, BHVelDispQuery, BHVelDispResult, BHVelDispLocalTreeWalk<CubicDensityKernel>, BHVelDispTopTreeWalk, BHVelDispPriv, BHVelDispOutput> {
    public:
    using TreeWalk::TreeWalk;
};

class BHVelDispTreeWalkQuartic: public TreeWalk<BHVelDispTreeWalkQuartic, BHVelDispQuery, BHVelDispResult, BHVelDispLocalTreeWalk<QuarticDensityKernel>, BHVelDispTopTreeWalk, BHVelDispPriv, BHVelDispOutput> {
    public:
    using TreeWalk::TreeWalk;
};

class BHVelDispTreeWalkQuintic: public TreeWalk<BHVelDispTreeWalkQuintic, BHVelDispQuery, BHVelDispResult, BHVelDispLocalTreeWalk<QuinticDensityKernel>, BHVelDispTopTreeWalk, BHVelDispPriv, BHVelDispOutput> {
    public:
    using TreeWalk::TreeWalk;
};

/* Compute the DM velocity dispersion for black holes*/
static void
blackhole_veldisp(const ActiveParticles * act, ForceTree * tree, KickFactorData& kf, MPI_Comm comm)
{
    /* This treewalk uses only DM */
    if(!tree->tree_allocated_flag)
        endrun(0, "DM Tree not allocated for veldisp\n");

    BHVelDispPriv priv(tree->BoxSize, kf);

    BHVelDispOutput output(SlotsManager);

    switch(GetDensityKernelType()) {
        case DENSITY_KERNEL_CUBIC_SPLINE:
            {
                BHVelDispTreeWalkCubic tw("BH_VDISP", tree, priv, &output);
                tw.run(act->ActiveParticle, act->NumActiveParticle, PartManager->Base, comm);
            }
            break;
        case DENSITY_KERNEL_QUARTIC_SPLINE:
            {
                BHVelDispTreeWalkQuartic tw("BH_VDISP", tree, priv, &output);
                tw.run(act->ActiveParticle, act->NumActiveParticle, PartManager->Base, comm);
            }
            break;
        default: //DENSITY_KERNEL_QUINTIC_SPLINE
            BHVelDispTreeWalkQuintic tw("BH_VDISP", tree, priv, &output);
            tw.run(act->ActiveParticle, act->NumActiveParticle, PartManager->Base, comm);
    }
    walltime_measure("/BH/VDisp");

}

/* Now comes the classes for the wind particle velocity dispersion. */

/* this evaluator walks the tree and sums the total mass of surrounding gas
 * particles as described in VS08. */
/* it also calculates the velocity dispersion of the nearest 40 DM or gas particles */

class WindVDispPriv : public ParamTypeBase {
    public:
    double Time;
    double hubble;
    KickFactorData kf;
    /* Lower and upper bounds on smoothing length: pointers to arrays allocated and managed in Output.*/
    MyFloat *Left, *Right, *DMRadius;
    WindVDispPriv(double BoxSize, double i_Time, double i_hubble, KickFactorData& kf_i, MyFloat * Left_i, MyFloat * Right_i, MyFloat * DMRadius_i):
    ParamTypeBase(BoxSize), Time(i_Time), hubble(i_hubble), kf(kf_i), Left(Left_i), Right(Right_i), DMRadius(DMRadius_i)
    {}
};

inline double
vdispeffdmradius(const int i, double left, double right, double DMRadius, const double BoxSize)
{
    /*The asymmetry is because it is free to compute extra densities for h < Hsml, but not for h > Hsml*/
    if (right > 0.99*BoxSize){
        right = DMRadius;
    }
    if(left == 0)
        left = 0.1 * DMRadius;
    /*Evenly split in volume*/
    double rvol = pow(right, 3);
    double lvol = pow(left, 3);
    return pow((1.0*i+1)/(1.0*NWINDHSML+1) * (rvol - lvol) + lvol, 1./3);
}

/* Code to compute velocity dispersions*/
class WindVDispQuery : public TreeWalkQueryBase<WindVDispPriv>{
    public:
    double Mass;
    double DMRadius[NWINDHSML];
    double Vel[3];
    double Hsml;

    MYCUDAFN WindVDispQuery(const particle_data& particle, const int * const i_NodeList, const int firstnode, const WindVDispPriv& priv):
    TreeWalkQueryBase<WindVDispPriv>(particle, i_NodeList, firstnode, priv), Mass(particle.Mass)
    {
        for(int i=0; i<3; i++)
            Vel[i] = particle.Vel[i];
        const int pi = particle.PI;
        for(int i = 0; i<NWINDHSML; i++){
            DMRadius[i] = vdispeffdmradius(i, priv.Left[pi], priv.Right[pi], priv.DMRadius[pi], priv.BoxSize);
        }
        Hsml = DMRadius[NWINDHSML-1];
    }
};

class WindVDispResult : public TreeWalkResultBase<WindVDispQuery>{
    public:
    double V1sum[NWINDHSML][3] = {};
    double V2sum[NWINDHSML] = {};
    double NumNgb[NWINDHSML] = {};
    int maxcmpte = NWINDHSML;

    MYCUDAFN WindVDispResult(const WindVDispQuery& query):
        TreeWalkResultBase<WindVDispQuery>(query) {}

    MYCUDAFN WindVDispResult& operator +=(const WindVDispResult& other)
    {
        static_cast<TreeWalkResultBase<WindVDispQuery>&>(*this) += static_cast<const TreeWalkResultBase<WindVDispQuery>&>(other);

        if(maxcmpte > other.maxcmpte) {
            maxcmpte = other.maxcmpte;
        }
        for(int i = 0; i < maxcmpte; i++) {
            NumNgb[i] += other.NumNgb[i];
            V2sum[i] += other.V2sum[i];
            for(int j = 0; j < 3; j++)
                V1sum[i][j] += other.V1sum[i][j];
        }
        return *this;
    }

    double GetNumNgb() const
    {
        return NumNgb[0];
    }
};

class WindVDispOutput {
    public:
    /* Lower and upper bounds on smoothing length*/
    MyFloat *Left, *Right, *DMRadius;
    double BoxSize;
    particle_data * parts;
    sph_particle_data * SphParts;
    bool verbose = false;

    WindVDispOutput(const double BoxSize, const int * WorkSet, const int64_t WorkSetSize, particle_data * parts_i, slots_manager_type * slotsmanager):
    parts(parts_i), SphParts(slotsmanager->sph_slot())
    {
        Left = mymalloc("VDISP->Left", MyFloat, SlotsManager->info[0].size);
        Right = mymalloc("VDISP->Right", MyFloat, SlotsManager->info[0].size);
        DMRadius = mymalloc("VDISP->DMRadius", MyFloat, SlotsManager->info[0].size);
        /*Initialise the arrays */
        #pragma omp parallel for
        for (int i = 0; i < WorkSetSize; i++) {
            const int n = WorkSet ? WorkSet[i] : i;
            if(parts[n].Type == 0) {
                const int pi = parts[n].PI;
                DMRadius[pi] = parts[n].Hsml;
                Left[pi] = 0;
                Right[pi] = BoxSize;
            }
        }
    }

    ~WindVDispOutput()
    {
        myfree(DMRadius);
        myfree(Right);
        myfree(Left);
    }

    MYCUDAFN int
    postprocess(const int place, WindVDispResult& result, particle_data * const parts, const WindVDispPriv * priv)
    {
        const int pi = parts[place].PI;
        double evaldmradius[NWINDHSML];
        for(int j = 0; j < result.maxcmpte; j++){
            evaldmradius[j] = vdispeffdmradius(j, Left[pi], Right[pi], DMRadius[pi], priv->BoxSize);
        }
        int close = 0;
        DMRadius[pi] = ngb_narrow_down(&Right[pi], &Left[pi], evaldmradius, result.NumNgb, result.maxcmpte, NUMDMNGB, &close, priv->BoxSize);
        const double numngb = result.NumNgb[close];
        result.NumNgb[0] = numngb;

        /*  If we have 40 neighbours, or if DMRadius is narrow, set vdisp and be done. Otherwise return 0 and add to redo queue. */
        if((numngb >= (NUMDMNGB - MAXDMDEVIATION) && numngb <= (NUMDMNGB + MAXDMDEVIATION)) ||
        (Right[pi] - Left[pi] < 5e-6 * Left[pi])) {
            if(Right[pi] - Left[pi] < 5e-6 * Left[pi])
                message(1, "Tight dm hsml for id %ld ngb %g radius %g\n",parts[place].ID, numngb, evaldmradius[close]);
            double vdisp = result.V2sum[close] / numngb;
            for(int d = 0; d < 3; d++){
                vdisp -= pow(result.V1sum[close][d] / numngb,2);
            }
            if(vdisp > 0) {
                if(parts[place].Type == 0)
                    SphParts[pi].VDisp = sqrt(vdisp / 3);
            }
            return 1;
        }
        /* Add to redo queue */
        return 0;
    }
};

/* This is split out because we want to run it before the treewalk is constructed,
 * and because it depends on a variety of bits of data. */
class VDispWork {
public:
     const particle_data * parts;
     const sph_particle_data * SphParts;
     double ddrift;
     double sfr_dens_thresh;
     MYCUDAFN bool operator()(int i) const {
         const particle_data& particle = parts[i];
         if(particle.IsGarbage || particle.Swallowed)
             return false;
         /* Only want gas*/
         if(particle.Type != 0)
             return false;
         /* We only want VDisp for gas particles that may be star-forming over the next PM timestep.
          * Use DtHsml.*/
         double densfac = (particle.Hsml + particle.DtHsml * ddrift)/particle.Hsml;
         if(densfac > 1)
             densfac = 1;
         if(SphParts[particle.PI].Density/(densfac * densfac * densfac) < 0.1 * sfr_dens_thresh)
             return false;
         /* Update veldisp only on a gravitationally active timestep,
          * since this is the frequency at which the gravitational acceleration,
          * which is the only thing DM contributes to, could change. This is probably
          * overly conservative, because most of the acceleration will be from other stars. */
         //     if(!is_timebin_active(particle.TimeBinGravity, particle.Ti_drift))
         //         return false;
         return true;
     }
};

int64_t build_vdisp_queue(int ** WorkSet, int * active_set, int64_t size, const double Time, DriftKickTimes * times, TimeBinMgr * timebinmgr)
{
    *WorkSet = mymanagedmalloc("ActiveQueue", int, size);
    double ddrift = timebinmgr->get_exact_drift_factor(times->Ti_Current, times->Ti_Current + times->PM_length);
    VDispWork haswork{PartManager->Base, SlotsManager->sph_slot(), ddrift, sfr_density_threshold(Time)};
    /* This is a standard stream compaction algorithm. It evaluates the haswork function
    * for every particle in the active set, stores the results in an array of flags, counts the non-zero flags,
    * and then scatters each particle integer to the right index in the final array. All is parallelized. */
    if(active_set) {
        auto end = std::copy_if(std::execution::par,
            active_set, active_set + size, *WorkSet, haswork);
        return end - *WorkSet;
    }
    else { // Need to handle this separately.
        /* The GPU code has a counting_iterator from thrust which avoids allocating the memory. This is the C++20 equivalent.*/
        auto iota = std::views::iota(0, (int) size);
        auto end = std::copy_if(std::execution::par, iota.begin(), iota.end(), *WorkSet, haswork);
        return end - *WorkSet;
    }
}

class WindVDispLocalTreeWalk: public LocalNgbTreeWalk<WindVDispLocalTreeWalk, WindVDispQuery, WindVDispResult, WindVDispPriv, NGB_TREEFIND_ASYMMETRIC, DMMASK>
{
    public:
    MYCUDAFN WindVDispLocalTreeWalk(const NODE * const Nodes, const WindVDispQuery& input):
    LocalNgbTreeWalk<WindVDispLocalTreeWalk, WindVDispQuery, WindVDispResult, WindVDispPriv, NGB_TREEFIND_ASYMMETRIC, DMMASK>(Nodes, input) {}

    /*! This function is the 'core' of the SPH force computation. A target
     *  particle is specified which may either be local, or reside in the
     *  communication buffer.     *
     * @param input  Query data
     * @param output Result accumulator
     */
    MYCUDAFN void ngbiter(WindVDispQuery& input, const particle_data& particle, WindVDispResult * output, const WindVDispPriv& priv)
    {
        double dist[3];
        double r2 = this->get_distance(input, particle, priv.BoxSize, dist);

        /* Check we are within the density kernel*/
        if(r2 <= 0 || !(r2 < input.DMRadius[NWINDHSML-1] * input.DMRadius[NWINDHSML-1]))
            return;

        const double r = sqrt(r2);

        const double atime = priv.Time;
        for (int i = 0; i < output->maxcmpte; i++) {
            if(r < input.DMRadius[i]) {
                output->NumNgb[i] += 1;
                int d;
                MyFloat VelPred[3];
                priv.kf.DM_VelPred(particle, VelPred);
                for(d = 0; d < 3; d ++) {
                    /* Add hubble flow to relative velocity. Use predicted velocity to current time.
                     * The I particle is active so always at current time.*/
                    double vel = VelPred[d] - input.Vel[d] + priv.hubble * atime * atime * dist[d];
                    output->V1sum[i][d] += vel;
                    output->V2sum[i] += vel * vel;
                }
            }
        }
        /* Update the search radius here, Hsml in QueryType, if we know that it is too large. */
        for(int i = 0; i<NWINDHSML; i++){
            if(output->NumNgb[i] > NUMDMNGB){
                output->maxcmpte = i+1;
                input.Hsml = input.DMRadius[i];
                break;
            }
        }
        /*
        message(1, "ThisTask = %d %ld ngb=%d NGB=%d TotalWeight=%g V2sum=%g V1sum=%g %g %g\n",
        ThisTask, I->ID, numngb, O->Ngb, O->TotalWeight, O->V2sum,
        O->V1sum[0], O->V1sum[1], O->V1sum[2]);
        */
    }
};

class WindVDispTopTreeWalk: public TopTreeWalk<WindVDispQuery, WindVDispPriv, NGB_TREEFIND_ASYMMETRIC> { using TopTreeWalk::TopTreeWalk; };

class WindVDispTreeWalk: public TreeWalk<WindVDispTreeWalk, WindVDispQuery, WindVDispResult, WindVDispLocalTreeWalk, WindVDispTopTreeWalk, WindVDispPriv, WindVDispOutput> {
    public:
    using TreeWalk::TreeWalk;
};

/* Find the 1D DM velocity dispersion of all gas particles by running a density loop.
 * Stores it in VDisp in the slots structure.*/
void
winds_find_vel_disp(const ActiveParticles * act, const double Time, const double hubble, DriftKickTimes * times, TimeBinMgr * timebinmgr, DomainDecomp * ddecomp)
{
    int * ActiveVDisp = NULL;
    /* Build the queue to check that we have something to do before we rebuild the tree.*/
    KickFactorData kf(times, timebinmgr);
    int64_t NumVDisp = build_vdisp_queue(&ActiveVDisp, act->ActiveParticle, act->NumActiveParticle, Time, times, timebinmgr);

    int64_t totvdisp, totbh;
    /* Check for black holes*/
    MPI_Allreduce(&SlotsManager->info[5].size, &totbh, 1, MPI_INT64, MPI_SUM, ddecomp->DomainComm);
    /* If this queue is empty, nothing to do for winds.*/
    MPI_Allreduce(&NumVDisp, &totvdisp, 1, MPI_INT64, MPI_SUM, ddecomp->DomainComm);

    ForceTree tree[1] = {{0}};
    if(totvdisp > 0 || totbh > 0)
        force_tree_rebuild_mask(tree, ddecomp, DMMASK, "");

    /* Compute the black hole velocity dispersions if needed*/
    if(totbh)
        blackhole_veldisp(act, tree, kf, ddecomp->DomainComm);

    if(totvdisp == 0) {
        force_tree_free(tree);
        myfree(ActiveVDisp);
        return;
    }
    WindVDispOutput output(PartManager->BoxSize, ActiveVDisp, NumVDisp, PartManager->Base, SlotsManager);
    WindVDispPriv priv(PartManager->BoxSize, Time, hubble, kf, output.Left, output.Right, output.DMRadius);
    report_memory_usage("WIND_VDISP");
    /* Find densities*/
    WindVDispTreeWalk tw("WIND_VDISP", tree, priv, &output);
    tw.do_hsml_loop(ActiveVDisp, NumVDisp, true, PartManager->Base, ddecomp->DomainComm);
    force_tree_free(tree);
    myfree(ActiveVDisp);
    walltime_measure("/Cooling/VDisp");
}
