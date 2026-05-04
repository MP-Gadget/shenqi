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

/* Computes the BH velocity dispersion for kinetic feedback*/
class BHVelDispOutput {
    public:
    /* temporary computed for kinetic feedback energy threshold*/
    MyFloat * NumDM;
    MyFloat (*V1sumDM)[3];
    MyFloat * V2sumDM;

    BHVelDispOutput(slots_manager_type * SlotsManager)
    {
        NumDM = (MyFloat *) mymalloc("NumDM", SlotsManager->info[5].size * sizeof(MyFloat));
        V1sumDM = (MyFloat (*) [3]) mymalloc("V1sumDM", 3* SlotsManager->info[5].size * sizeof(V1sumDM[0]));
        V2sumDM = (MyFloat *) mymalloc("V2sumDM", SlotsManager->info[5].size * sizeof(MyFloat));
    }

    ~BHVelDispOutput()
    {
        myfree(V2sumDM);
        myfree(V1sumDM);
        myfree(NumDM);
    }

    void
    postprocess(int i, particle_data * const parts, const BHVelDispPriv * priv)
    {
        const int PI = parts[i].PI;
        /*************************************************************************/
        /* decide whether to release KineticFdbkEnergy*/
        const double numdm = NumDM[PI];
        if (numdm > 0) {
            double vdisp = V2sumDM[PI]/numdm;
            for(int d = 0; d<3; d++)
                vdisp -= pow(V1sumDM[PI][d]/numdm,2);
            if(vdisp > 0)
                BHP(i).VDisp = sqrt(vdisp / 3);
        }
    }

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

class BHVelDispResult : public TreeWalkResultBase<BHVelDispQuery, BHVelDispOutput> {
    public:
    /* used for AGN kinetic feedback */
    MyFloat V2sumDM;
    MyFloat V1sumDM[3];
    MyFloat NumDM;

    template<TreeWalkReduceMode mode>
    MYCUDAFN void reduce(int place, const BHVelDispOutput * output, struct particle_data * const parts)
    {
        int PI = Part[place].PI;
        for (int k = 0; k < 3; k++){
            TREEWALK_REDUCE(output->V1sumDM[PI][k], V1sumDM[k]);
        }
        TREEWALK_REDUCE(output->NumDM[PI], NumDM);
        TREEWALK_REDUCE(output->V2sumDM[PI], V2sumDM);
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
blackhole_veldisp(const ActiveParticles * act, ForceTree * tree, KickFactorData& kf)
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
                tw.run(act->ActiveParticle, act->NumActiveParticle, PartManager->Base, MPI_COMM_WORLD);
            }
            break;
        case DENSITY_KERNEL_QUARTIC_SPLINE:
            {
                BHVelDispTreeWalkQuartic tw("BH_VDISP", tree, priv, &output);
                tw.run(act->ActiveParticle, act->NumActiveParticle, PartManager->Base, MPI_COMM_WORLD);
            }
            break;
        default: //DENSITY_KERNEL_QUINTIC_SPLINE
            BHVelDispTreeWalkQuintic tw("BH_VDISP", tree, priv, &output);
            tw.run(act->ActiveParticle, act->NumActiveParticle, PartManager->Base, MPI_COMM_WORLD);
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


class WindVDispOutput {
    public:
    /* Maximum index where NumNgb is valid. */
    int * maxcmpte;
    MyFloat (* V2sum)[NWINDHSML];
    MyFloat (* V1sum)[NWINDHSML][3];
    MyFloat (* NumNgb) [NWINDHSML];
    /* Lower and upper bounds on smoothing length*/
    MyFloat *Left, *Right, *DMRadius;
    double BoxSize;
    particle_data * parts;
    sph_particle_data * SphParts;
    bool verbose = false;

    WindVDispOutput(const double BoxSize, const int * WorkSet, const int64_t WorkSetSize, particle_data * parts_i, slots_manager_type * slotsmanager):
    parts(parts_i), SphParts(reinterpret_cast<sph_particle_data *>(slotsmanager->info[0].ptr))
    {
        Left = (MyFloat *) mymalloc("VDISP->Left", SlotsManager->info[0].size * sizeof(MyFloat));
        Right = (MyFloat *) mymalloc("VDISP->Right", SlotsManager->info[0].size * sizeof(MyFloat));
        DMRadius = (MyFloat *) mymalloc("VDISP->DMRadius", SlotsManager->info[0].size * sizeof(MyFloat));
        NumNgb = (MyFloat (*) [NWINDHSML]) mymalloc("VDISP->NumNgb", SlotsManager->info[0].size * sizeof(NumNgb[0]));
        V1sum = (MyFloat (*) [NWINDHSML][3]) mymalloc("VDISP->V1Sum", SlotsManager->info[0].size * sizeof(V1sum[0]));
        V2sum = (MyFloat (*) [NWINDHSML]) mymalloc("VDISP->V2Sum", SlotsManager->info[0].size * sizeof(V2sum[0]));
        maxcmpte = (int *) mymalloc("maxcmpte", SlotsManager->info[0].size * sizeof(int));

        /*Initialise the arrays */
        #pragma omp parallel for
        for (int i = 0; i < WorkSetSize; i++) {
            const int n = WorkSet ? WorkSet[i] : i;
            if(parts[n].Type == 0) {
                const int pi = parts[n].PI;
                DMRadius[pi] = parts[n].Hsml;
                Left[pi] = 0;
                Right[pi] = BoxSize;
                maxcmpte[pi] = NUMDMNGB;
            }
        }
    }

    ~WindVDispOutput()
    {
        myfree(maxcmpte);
        myfree(V2sum);
        myfree(V1sum);
        myfree(NumNgb);
        myfree(DMRadius);
        myfree(Right);
        myfree(Left);
    }

    double GetNumNgb(const int i)
    {
        return NumNgb[parts[i].PI][NWINDHSML-1];
    }

    MYCUDAFN int
    postprocess(const int i, particle_data * const parts, const WindVDispPriv * priv)
    {
        const int pi = parts[i].PI;
        const int maxcmpt = maxcmpte[pi];
        double evaldmradius[NWINDHSML];
        for(int j = 0; j < maxcmpt; j++){
            evaldmradius[j] = vdispeffdmradius(j, Left[pi], Right[pi], DMRadius[pi], priv->BoxSize);
        }
        int close = 0;
        DMRadius[pi] = ngb_narrow_down(&Right[pi], &Left[pi], evaldmradius, NumNgb[pi], maxcmpt, NUMDMNGB, &close, priv->BoxSize);
        double numngb = NumNgb[pi][close];
        /* Ensure the largest entry is set to the current best NumNgb. */
        NumNgb[pi][NWINDHSML-1] = numngb;

        /*  If we have 40 neighbours, or if DMRadius is narrow, set vdisp and be done. Otherwise return 0 and add to redo queue. */
        if((numngb >= (NUMDMNGB - MAXDMDEVIATION) && numngb <= (NUMDMNGB + MAXDMDEVIATION)) ||
        (Right[pi] - Left[pi] < 5e-6 * Left[pi])) {
            if(Right[pi] - Left[pi] < 5e-6 * Left[pi])
                message(1, "Tight dm hsml for id %ld ngb %g radius %g\n",parts[i].ID, numngb, evaldmradius[close]);
            double vdisp = V2sum[pi][close] / numngb;
            for(int d = 0; d < 3; d++){
                vdisp -= pow(V1sum[pi][close][d] / numngb,2);
            }
            if(vdisp > 0) {
                if(parts[i].Type == 0)
                    SphParts[pi].VDisp = sqrt(vdisp / 3);
            }
            return 1;
        }
        /* Add to redo queue */
        return 0;
    }
};

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
    *WorkSet = (int *) mymanagedmalloc("ActiveQueue", size * sizeof(int));
    double ddrift = timebinmgr->get_exact_drift_factor(times->Ti_Current, times->Ti_Current + times->PM_length);
    VDispWork haswork{PartManager->Base, reinterpret_cast<sph_particle_data *>(SlotsManager->info[0].ptr), ddrift, sfr_density_threshold(Time)};
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

class WindVDispResult : public TreeWalkResultBase<WindVDispQuery, WindVDispOutput>{
    public:
    double V1sum[NWINDHSML][3];
    double V2sum[NWINDHSML];
    double NumNgb[NWINDHSML];
    int maxcmpte = NWINDHSML;
    template<TreeWalkReduceMode mode>
    MYCUDAFN void reduce(int place, const WindVDispOutput * output, struct particle_data * const parts)
    {
        int pi = Part[place].PI;
        if constexpr(mode == TREEWALK_PRIMARY)
            output->maxcmpte[pi] = maxcmpte;
        else if (output->maxcmpte[pi] > maxcmpte)
            output->maxcmpte[pi] = maxcmpte;
        for (int i = 0; i < maxcmpte; i++){
            TREEWALK_REDUCE(output->NumNgb[pi][i], NumNgb[i]);
            TREEWALK_REDUCE(output->V2sum[pi][i], V2sum[i]);
            for(int k = 0; k < 3; k ++) {
                TREEWALK_REDUCE(output->V1sum[pi][i][k], V1sum[i][k]);
            }
        }
    //     message(1, "Reduce ID=%ld, NGB_first=%d NGB_last=%d maxcmpte = %d, left = %g, right = %g\n",
    //             Part[place].ID, O->Ngb[0],O->Ngb[O->maxcmpte-1],WINDP(place, Windd).maxcmpte,WINDP(place, Windd).Left,WINDP(place, Windd).Right);
    }
};

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
    MPI_Allreduce(&SlotsManager->info[5].size, &totbh, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);
    /* If this queue is empty, nothing to do for winds.*/
    MPI_Allreduce(&NumVDisp, &totvdisp, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);

    ForceTree tree[1] = {{0}};
    if(totvdisp > 0 || totbh > 0)
        force_tree_rebuild_mask(tree, ddecomp, DMMASK, NULL);

    /* Compute the black hole velocity dispersions if needed*/
    if(totbh)
        blackhole_veldisp(act, tree, kf);

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
    tw.do_hsml_loop(ActiveVDisp, NumVDisp, true, PartManager->Base);
    force_tree_free(tree);
    myfree(ActiveVDisp);
    walltime_measure("/Cooling/VDisp");
}
