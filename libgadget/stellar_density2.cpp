#include <math.h>
#include <omp.h>
#include "metal_return.h"
#include "treewalk2.h"
#include "density2.h"
#include "densitykernel.hpp"
#include "partmanager.h"
#include "slotsmanager.h"
#include "walltime.h"
#include "utils/endrun.h"
#include "utils/mymalloc.h"

/* This file contains the stellar density treewalk, which finds the SPH volume
 * weights around each star used to distribute the returned metals. It is a
 * density-like treewalk with an Hsml iteration, ported to the new C++ treewalk. */

/* Number of densities to evaluate simultaneously*/
#define NHSML 10

/* Parameters for the stellar density treewalk. */
class StellarDensityPriv : public ParamTypeBase {
    public:
    /*!< Desired number of SPH neighbours */
    double DesNumNgb;
    /* Maximum allowed deviation from the desired number of neighbours*/
    double MaxNgbDeviation;
    /* If true, weight the SPH volume by the kernel*/
    int SPHWeighting;
    sph_particle_data * SphParts;
    /* Lower and upper bounds on smoothing length: pointers to arrays allocated and managed in the Output.*/
    MyFloat *Left, *Right;
    StellarDensityPriv(double BoxSize, double DesNumNgb_i, double MaxNgbDeviation_i, int SPHWeighting_i, slots_manager_type * slotsmanager, MyFloat * Left_i, MyFloat * Right_i):
        ParamTypeBase(BoxSize), DesNumNgb(DesNumNgb_i), MaxNgbDeviation(MaxNgbDeviation_i), SPHWeighting(SPHWeighting_i), SphParts(slotsmanager->sph_slot()), Left(Left_i), Right(Right_i) {}
};

/* Get Hsml for one of the NHSML evaluations, evenly spaced in volume between Left and Right. */
MYCUDAFN static inline double
stellareffhsml(const int i, double left, double right, const double Hsml, const double BoxSize)
{
    /* Use slightly past the current Hsml as the right most boundary*/
    if(right > 0.99*BoxSize)
        right = Hsml * ((1.+NHSML)/NHSML);
    /* Use 1/2 of current Hsml for left. The asymmetry is because it is free
     * to compute extra densities for h < Hsml, but not for h > Hsml.*/
    if(left == 0)
        left = 0.1 * Hsml;
    /* From left + 1/N  to right - 1/N, evenly spaced in volume,
     * since NumNgb ~ h^3.*/
    double rvol = pow(right, 3);
    double lvol = pow(left, 3);
    return pow((1.*i+1)/(1.*NHSML+1) * (rvol - lvol) + lvol, 1./3);
}

class StellarDensityOutput {
    public:
    /* Current number of neighbours*/
    MyFloat (*NumNgb)[NHSML];
    MyFloat (*VolumeSPH)[NHSML];
    /* Lower and upper bounds on smoothing length*/
    MyFloat *Left, *Right;
    /* Maximum index where NumNgb is valid. */
    int * maxcmpte;
    /* The SPH volume weight for each star: the output of this treewalk.*/
    MyFloat * StarVolumeSPH;
    particle_data * parts;
    bool verbose = false;

    StellarDensityOutput(MyFloat * StarVolumeSPH_i, const ForceTree * const tree, const int * WorkSet, const int64_t WorkSetSize, particle_data * const parts_i, slots_manager_type * slotsmanager):
    StarVolumeSPH(StarVolumeSPH_i), parts(parts_i)
    {
        typedef MyFloat NumNgbArray[NHSML];
        const int64_t nstar = slotsmanager->info[4].size;
        Left = mymalloc("DENS_PRIV->Left", MyFloat, nstar);
        Right = mymalloc("DENS_PRIV->Right", MyFloat, nstar);
        NumNgb = mymalloc("DENS_PRIV->NumNgb", NumNgbArray, nstar);
        VolumeSPH = mymalloc("DENS_PRIV->VolumeSPH", NumNgbArray, nstar);
        maxcmpte = mymalloc("maxcmpte", int, nstar);

        /*Initialise the bounds for the active stars*/
        #pragma omp parallel for
        for(int64_t i = 0; i < WorkSetSize; i++) {
            const int p = WorkSet ? WorkSet[i] : i;
            const int pi = parts[p].PI;
            Left[pi] = 0;
            Right[pi] = tree->BoxSize;
            /* If somehow Hsml has become zero through underflow, use something non-zero
             * to make sure we converge. */
            if(parts[p].Hsml == 0) {
                const int fat = force_get_father(p, tree);
                parts[p].Hsml = tree->Nodes[fat].len;
                if(parts[p].Hsml == 0)
                    parts[p].Hsml = tree->BoxSize / pow(PartManager->NumPart, 1./3)/4.;
            }
        }
    }

    ~StellarDensityOutput()
    {
        myfree(maxcmpte);
        myfree(VolumeSPH);
        myfree(NumNgb);
        myfree(Right);
        myfree(Left);
    }

    double GetNumNgb(const int i)
    {
        const int pi = parts[i].PI;
        return NumNgb[pi][maxcmpte[pi]-1];
    }

    /* Narrow the Hsml bounds from the computed neighbour numbers and store the
     * SPH volume weight. Returns 1 if the star is done, 0 if it needs to be redone.*/
    int
    postprocess(const int i, particle_data * const parts, const StellarDensityPriv * priv)
    {
        const int pi = parts[i].PI;
        const int maxcmpt = maxcmpte[pi];
        double evalhsml[NHSML];
        evalhsml[0] = stellareffhsml(0, Left[pi], Right[pi], parts[i].Hsml, priv->BoxSize);
        for(int j = 1; j < maxcmpt; j++)
            evalhsml[j] = stellareffhsml(j, Left[pi], Right[pi], parts[i].Hsml, priv->BoxSize);

        int close = 0;
        parts[i].Hsml = ngb_narrow_down(&Right[pi], &Left[pi], evalhsml, NumNgb[pi], maxcmpt, priv->DesNumNgb, &close, priv->BoxSize);
        const double numngb = NumNgb[pi][close];

        /* Save the volume weight for the metal return*/
        StarVolumeSPH[pi] = VolumeSPH[pi][close];

        /* now check whether we had enough neighbours */
        if(numngb < (priv->DesNumNgb - priv->MaxNgbDeviation) ||
                numngb > (priv->DesNumNgb + priv->MaxNgbDeviation))
        {
            /* This condition is here to prevent the density code looping forever if it encounters
             * multiple particles at the same position. If this happens you likely have worse
             * problems anyway, so warn also. */
            if((Right[pi] - Left[pi]) < 1.0e-4 * Left[pi])
            {
                /* If this happens probably the exchange is screwed up and all your particles have moved to (0,0,0)*/
                message(1, "Very tight Hsml bounds for i=%d ID=%lu type %d Hsml=%g Left=%g Right=%g Ngbs=%g des = %g Right-Left=%g pos=(%g|%g|%g)\n",
                 i, parts[i].ID, parts[i].Type, evalhsml[0], Left[pi], Right[pi], numngb, priv->DesNumNgb, Right[pi] - Left[pi], parts[i].Pos[0], parts[i].Pos[1], parts[i].Pos[2]);
                return 1;
            }
            if(verbose)
                message(1, "i=%d ID=%lu Hsml=%g lastdhsml=%g Left=%g Right=%g Ngbs=%g Right-Left=%g pos=(%g|%g|%g)\n",
                 i, parts[i].ID, parts[i].Hsml, evalhsml[close], Left[pi], Right[pi], numngb, Right[pi] - Left[pi], parts[i].Pos[0], parts[i].Pos[1], parts[i].Pos[2]);
            /* More work needed: add this particle to the redo queue*/
            return 0;
        }
        if(StarVolumeSPH[pi] == 0)
            endrun(3, "i = %d pi = %d StarVolumeSPH %g hsml %g\n", i, pi, StarVolumeSPH[pi], parts[i].Hsml);
        return 1;
    }
};

class StellarDensityQuery : public TreeWalkQueryBase<StellarDensityPriv>
{
    public:
    /* The search radius: the largest evaluation radius. Reduced during the
     * walk once we know we have enough neighbours at a smaller radius. */
    MyFloat Hsml;
    MyFloat HsmlEval[NHSML];

    MYCUDAFN StellarDensityQuery(const particle_data& particle, const int * const i_NodeList, const int firstnode, const StellarDensityPriv& priv):
    TreeWalkQueryBase<StellarDensityPriv>(particle, i_NodeList, firstnode, priv)
    {
        const int pi = particle.PI;
        for(int i = 0; i < NHSML; i++)
            HsmlEval[i] = stellareffhsml(i, priv.Left[pi], priv.Right[pi], particle.Hsml, priv.BoxSize);
        Hsml = HsmlEval[NHSML-1];
    }

    static MYCUDAFN bool haswork(const particle_data& particle) {
        if(!TreeWalkQueryBase::haswork(particle))
            return false;
        /* Stars only. The mass return condition is checked when building the initial queue.*/
        return particle.Type == 4;
    }
};

class StellarDensityResult : public TreeWalkResultBase<StellarDensityQuery, StellarDensityOutput> {
    public:
    MyFloat VolumeSPH[NHSML];
    MyFloat Ngb[NHSML];
    int maxcmpte = NHSML;

    template<TreeWalkReduceMode mode>
    MYCUDAFN void reduce(int place, const StellarDensityOutput * output, struct particle_data * const parts)
    {
        const int pi = parts[place].PI;
        if constexpr(mode == TREEWALK_PRIMARY)
            output->maxcmpte[pi] = maxcmpte;
        else if(output->maxcmpte[pi] > maxcmpte)
            output->maxcmpte[pi] = maxcmpte;
        for(int i = 0; i < maxcmpte; i++) {
            TREEWALK_REDUCE(output->NumNgb[pi][i], Ngb[i]);
            TREEWALK_REDUCE(output->VolumeSPH[pi][i], VolumeSPH[i]);
        }
    }
};

template <typename DensityKernel>
class StellarDensityLocalTreeWalk: public LocalNgbTreeWalk<StellarDensityLocalTreeWalk<DensityKernel>, StellarDensityQuery, StellarDensityResult, StellarDensityPriv, NGB_TREEFIND_ASYMMETRIC, GASMASK>
{
    public:
    MYCUDAFN StellarDensityLocalTreeWalk(const NODE * const Nodes, const StellarDensityQuery& input):
    LocalNgbTreeWalk<StellarDensityLocalTreeWalk<DensityKernel>, StellarDensityQuery, StellarDensityResult, StellarDensityPriv, NGB_TREEFIND_ASYMMETRIC, GASMASK>(Nodes, input)
    { }

    /*! This function computes the number of neighbours and the SPH volume
     * weight around a star for each of the NHSML evaluation radii.
     * @param input  Query data
     * @param output Result accumulator
     */
    MYCUDAFN void ngbiter(StellarDensityQuery& input, const particle_data& particle, StellarDensityResult * output, const StellarDensityPriv& priv)
    {
        double dist[3];
        const double r2 = this->get_distance(input, particle, priv.BoxSize, dist);

        if(!(r2 < input.HsmlEval[output->maxcmpte-1] * input.HsmlEval[output->maxcmpte-1]))
            return;

        const double r = sqrt(r2);
        for(int i = 0; i < output->maxcmpte; i++) {
            if(r2 < input.HsmlEval[i] * input.HsmlEval[i])
            {
                DensityKernel kernel(input.HsmlEval[i]);
                const double wk = kernel.wk(r / input.HsmlEval[i]);
                output->Ngb[i] += wk * kernel.volume();
                /* For stars we need the total weighting, sum(w_k m_k / rho_k).*/
                double thisvol = particle.Mass / priv.SphParts[particle.PI].Density;
                if(priv.SPHWeighting)
                    thisvol *= wk;
                output->VolumeSPH[i] += thisvol;
            }
        }
        /* If there is an entry which is above desired DesNumNgb,
         * we don't need to search past it. After this point
         * all entries in the Ngb table above O->Ngb are invalid.*/
        for(int i = 0; i < NHSML; i++) {
            if(output->Ngb[i] > priv.DesNumNgb) {
                output->maxcmpte = i+1;
                input.Hsml = input.HsmlEval[i];
                break;
            }
        }
    }
};

class StellarDensityTopTreeWalk: public TopTreeWalk<StellarDensityQuery, StellarDensityPriv, NGB_TREEFIND_ASYMMETRIC> { using TopTreeWalk::TopTreeWalk; };

class StellarDensityTreeWalkCubic: public TreeWalk<StellarDensityTreeWalkCubic, StellarDensityQuery, StellarDensityResult, StellarDensityLocalTreeWalk<CubicDensityKernel>, StellarDensityTopTreeWalk, StellarDensityPriv, StellarDensityOutput> {
    public:
    using TreeWalk::TreeWalk;
};

class StellarDensityTreeWalkQuartic: public TreeWalk<StellarDensityTreeWalkQuartic, StellarDensityQuery, StellarDensityResult, StellarDensityLocalTreeWalk<QuarticDensityKernel>, StellarDensityTopTreeWalk, StellarDensityPriv, StellarDensityOutput> {
    public:
    using TreeWalk::TreeWalk;
};

class StellarDensityTreeWalkQuintic: public TreeWalk<StellarDensityTreeWalkQuintic, StellarDensityQuery, StellarDensityResult, StellarDensityLocalTreeWalk<QuinticDensityKernel>, StellarDensityTopTreeWalk, StellarDensityPriv, StellarDensityOutput> {
    public:
    using TreeWalk::TreeWalk;
};

/* This selects the stars with significant enrichment this timestep.
 * It matches metals_haswork in metal_return.cpp, which is used for
 * the metal return treewalk that runs after this one. */
class StellarDensityWork {
public:
     const particle_data * parts;
     const star_particle_data * StarParts;
     const MyFloat * MassReturn;
     MYCUDAFN bool operator()(int n) const {
         const particle_data& particle = parts[n];
         if(particle.IsGarbage || particle.Swallowed)
             return false;
         if(particle.Type != 4)
             return false;
         /* Don't do enrichment from all stars, just those with significant enrichment*/
         if(MassReturn[particle.PI] < 1e-3 * (particle.Mass + StarParts[particle.PI].TotalMassReturned))
             return false;
         return true;
     }
};

static int64_t
build_stellar_density_queue(int ** WorkSet, int * active_set, const int64_t size, MyFloat * MassReturn)
{
    if(size == 0) {
        *WorkSet = mymanagedmalloc("StarDensQueue", int, 1);
        return 0;
    }
    *WorkSet = mymanagedmalloc("StarDensQueue", int, size);
    StellarDensityWork haswork{PartManager->Base, SlotsManager->star_slot(), MassReturn};
    if(active_set) {
        auto end = std::copy_if(std::execution::par, active_set, active_set + size, *WorkSet, haswork);
        return end - *WorkSet;
    }
    /* This is the C++20 equivalent of a counting_iterator.*/
    auto iota = std::views::iota(0, (int) size);
    auto end = std::copy_if(std::execution::par, iota.begin(), iota.end(), *WorkSet, haswork);
    return end - *WorkSet;
}

void
stellar_density(const ActiveParticles * act, MyFloat * StarVolumeSPH, MyFloat * MassReturn, const ForceTree * const tree, const int SPHWeighting, const double MaxNgbDeviation)
{
    int * StarQueue = NULL;
    const int64_t NumQueue = build_stellar_density_queue(&StarQueue, act->ActiveParticle, act->NumActiveParticle, MassReturn);

    StellarDensityOutput output(StarVolumeSPH, tree, StarQueue, NumQueue, PartManager->Base, SlotsManager);
    StellarDensityPriv priv(tree->BoxSize, GetNumNgb(GetDensityKernelType()), MaxNgbDeviation, SPHWeighting, SlotsManager, output.Left, output.Right);

    switch(GetDensityKernelType()) {
        case DENSITY_KERNEL_CUBIC_SPLINE:
            {
                StellarDensityTreeWalkCubic tw("STELLAR_DENSITY", tree, priv, &output);
                tw.do_hsml_loop(StarQueue, NumQueue, true, PartManager->Base, MPI_COMM_WORLD);
                tw.print_stats("/SPH/Metals/Density", MPI_COMM_WORLD);
            }
            break;
        case DENSITY_KERNEL_QUARTIC_SPLINE:
            {
                StellarDensityTreeWalkQuartic tw("STELLAR_DENSITY", tree, priv, &output);
                tw.do_hsml_loop(StarQueue, NumQueue, true, PartManager->Base, MPI_COMM_WORLD);
                tw.print_stats("/SPH/Metals/Density", MPI_COMM_WORLD);
            }
            break;
        default: //DENSITY_KERNEL_QUINTIC_SPLINE
            StellarDensityTreeWalkQuintic tw("STELLAR_DENSITY", tree, priv, &output);
            tw.do_hsml_loop(StarQueue, NumQueue, true, PartManager->Base, MPI_COMM_WORLD);
            tw.print_stats("/SPH/Metals/Density", MPI_COMM_WORLD);
    }
    myfree(StarQueue);
}
