#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#include "libgadget/localtreewalk2.h"
#include "libgadget/partmanager.h"
#include "physconst.h"
#include "walltime.h"
#include "density.h"
#include "treewalk2.h"
#include "timefac.h"
#include "slotsmanager.h"
#include "timestep.h"
#include "utils/endrun.h"
#include "utils/mymalloc.h"
#include "gravity.h"
#include "winds.h"

static struct density_params DensityParams;

/*Set cooling module parameters from a cooling_params struct for the tests*/
void
set_densitypar(struct density_params dp)
{
    DensityParams = dp;
}

/*Set the parameters of the density module*/
void
set_density_params(ParameterSet * ps)
{
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    if(ThisTask == 0) {
        DensityParams.DensityKernelType = (enum DensityKernelType) param_get_enum(ps, "DensityKernelType");
        DensityParams.MaxNumNgbDeviation = param_get_double(ps, "MaxNumNgbDeviation");
        DensityParams.DensityResolutionEta = param_get_double(ps, "DensityResolutionEta");
        DensityParams.MinGasHsmlFractional = param_get_double(ps, "MinGasHsmlFractional");

        DensityKernel kernel;
        density_kernel_init(&kernel, 1.0, DensityParams.DensityKernelType);
        message(1, "The Density Kernel type is %s\n", kernel.name);
        message(1, "The Density resolution is %g * mean separation, or %g neighbours\n",
                    DensityParams.DensityResolutionEta, GetNumNgb(GetDensityKernelType()));
        /*These two look like black hole parameters but they are really neighbour finding parameters*/
        DensityParams.BlackHoleNgbFactor = param_get_double(ps, "BlackHoleNgbFactor");
        DensityParams.BlackHoleMaxAccretionRadius = param_get_double(ps, "BlackHoleMaxAccretionRadius");
    }
    MPI_Bcast(&DensityParams, sizeof(struct density_params), MPI_BYTE, 0, MPI_COMM_WORLD);
}

double
GetNumNgb(enum DensityKernelType KernelType)
{
    DensityKernel kernel;
    density_kernel_init(&kernel, 1.0, KernelType);
    return density_kernel_desnumngb(&kernel, DensityParams.DensityResolutionEta);
}

enum DensityKernelType
GetDensityKernelType(void)
{
    return DensityParams.DensityKernelType;
}

/* The evolved entropy at drift time: evolved dlog a.
 * Used to predict pressure and entropy for SPH */
MyFloat
SPH_EntVarPred(const int p_i, const DriftKickTimes * times)
{
        const int bin = Part[p_i].TimeBinHydro;
        const int PI = Part[p_i].PI;
        const double dloga = dloga_from_dti(times->Ti_Current - times->Ti_kick[bin], times->Ti_Current);
        double EntVarPred = SphP[PI].Entropy + SphP[PI].DtEntropy * dloga;
        /*Entropy limiter for the predicted entropy: makes sure entropy stays positive. */
        if(EntVarPred < 0.05*SphP[PI].Entropy)
            EntVarPred = 0.05 * SphP[PI].Entropy;
        /* Just in case*/
        if(EntVarPred <= 0)
            return 0;
        EntVarPred = exp(1./GAMMA * log(EntVarPred));
//         EntVarPred = pow(EntVarPred, 1/GAMMA);
        return EntVarPred;
}

/* Structure storing the pre-computed kick factors which
 * used for making the predicted velocities.*/
class KickFactorData
{
    public:
    double FgravkickB;
    double gravkicks[TIMEBINS+1];
    double hydrokicks[TIMEBINS+1];

    /* Initialise the grav and hydrokick arrays for the current kick times.*/
    KickFactorData(const DriftKickTimes * const times, Cosmology * CP)
    {
        int i;
        /* Factor this out since all particles have the same PM kick time*/
        FgravkickB = get_exact_gravkick_factor(CP, times->PM_kick, times->Ti_Current);
        memset(gravkicks, 0, sizeof(gravkicks[0])*(TIMEBINS+1));
        memset(hydrokicks, 0, sizeof(hydrokicks[0])*(TIMEBINS+1));
        /* Compute the factors to move a current kick times velocity to the drift time velocity.
         * We need to do the computation for all timebins up to the maximum because even inactive
         * particles may have interactions. */
        #pragma omp parallel for
        for(i = times->mintimebin; i <= TIMEBINS; i++)
        {
            gravkicks[i] = get_exact_gravkick_factor(CP, times->Ti_kick[i], times->Ti_Current);
            hydrokicks[i] = get_exact_hydrokick_factor(CP, times->Ti_kick[i], times->Ti_Current);
        }
    }

    /* Get the predicted velocity for a particle
     * at the current Force computation time ti,
     * which always coincides with the Drift inttime.
     * For hydro forces.*/
    void
    SPH_VelPred(const particle_data& particle, MyFloat * VelPred)
    {
        int j;
        const double * const HydroAccel = ((struct sph_particle_data *)SlotsManager->info[0].ptr)[particle.PI].HydroAccel;
        /* Notice that the kick time for gravity and hydro may be different! So the prediction is also different*/
        for(j = 0; j < 3; j++) {
            VelPred[j] = particle.Vel[j] + gravkicks[particle.TimeBinGravity] * particle.FullTreeGravAccel[j]
                + particle.GravPM[j] * FgravkickB + hydrokicks[particle.TimeBinHydro] * HydroAccel[j];
        }
    }

    /* Get the predicted velocity for a particle
     * at the current Force computation time ti,
     * which always coincides with the Drift inttime.
     * For hydro forces.*/
    void
    DM_VelPred(int i, MyFloat * VelPred)
    {
        int j;
        for(j = 0; j < 3; j++)
            VelPred[j] = Part[i].Vel[j] + gravkicks[Part[i].TimeBinGravity] * Part[i].FullTreeGravAccel[j]+ Part[i].GravPM[j] * FgravkickB;
    }

};


class DensityPriv : ParamTypeBase {
    public:
    const bool update_hsml;
    /* Are there potentially black holes?*/
    const bool BlackHoleOn;
    const bool DoEgyDensity;

    const DriftKickTimes * const times;
    /* The gradient of the density, used sometimes during star formation.
     * May be NULL.*/
    MyFloat * GradRho;
    /* Current number of neighbours*/
    MyFloat *NumNgb;
    /* Lower and upper bounds on smoothing length*/
    MyFloat *Left, *Right;
    MyFloat (*Rot)[3];
    /* This is the DhsmlDensityFactor for the pure density,
     * not the entropy weighted density.
     * If DensityIndependentSphOn = 0 then DhsmlEgyDensityFactor and DhsmlDensityFactor
     * are the same and this is not used.
     * If DensityIndependentSphOn = 1 then this is used to set DhsmlEgyDensityFactor.*/
    MyFloat * DhsmlDensityFactor;

    /*!< Desired number of SPH neighbours */
    double DesNumNgb;
    /*!< minimum allowed SPH smoothing length */
    double MinGasHsml;
    /* Predicted quantities computed during for density and reused during hydro.*/
    struct sph_pred_data SPH_predicted;

    /* For computing the predicted quantities dynamically during the treewalk.*/
    KickFactorData kf;

    DensityPriv(const bool i_update_hsml, const bool i_DoEgyDensity, const bool i_BlackHoleOn, const DriftKickTimes * const i_times, const bool GradRho_mag, const double BoxSize, Cosmology * CP, const ActiveParticles * const act, const struct part_manager_type * const PartManager):
    update_hsml(i_update_hsml), BlackHoleOn(i_BlackHoleOn), DoEgyDensity(i_DoEgyDensity), times(i_times), kf(i_times, CP)
    {
        Left = (MyFloat *) mymalloc("DENS_PRIV->Left", PartManager->NumPart * sizeof(MyFloat));
        Right = (MyFloat *) mymalloc("DENS_PRIV->Right", PartManager->NumPart * sizeof(MyFloat));
        NumNgb = (MyFloat *) mymalloc("DENS_PRIV->NumNgb", PartManager->NumPart * sizeof(MyFloat));
        Rot = (MyFloat (*) [3]) mymalloc("DENS_PRIV->Rot", SlotsManager->info[0].size * sizeof(Rot[0]));
        /* This one stores the gradient for h finding. The factor stored in SPHP->DhsmlEgyDensityFactor depends on whether PE SPH is enabled.*/
        DhsmlDensityFactor = (MyFloat *) mymalloc("DhsmlDensity", PartManager->NumPart * sizeof(MyFloat));

        DesNumNgb = GetNumNgb(DensityParams.DensityKernelType);
        MinGasHsml = DensityParams.MinGasHsmlFractional * (FORCE_SOFTENING()/2.8);

        if(GradRho_mag)
            GradRho = (MyFloat *) mymalloc("SPH_GradRho", sizeof(MyFloat) * 3 * SlotsManager->info[0].size);
        else
            GradRho = NULL;

        int i;
        /* Init Left and Right: this has to be done before treewalk */
        #pragma omp parallel for
        for(i = 0; i < act->NumActiveParticle; i++)  {
            int p_i = act->ActiveParticle ? act->ActiveParticle[i] : i;
            /* We only really need active particles with work
            * but I don't want to read the particle table here*/
            Right[p_i] = BoxSize;
            NumNgb[p_i] = 0;
            Left[p_i] = 0;
        }

        /* If all particles are active, easiest to compute all the predicted velocities immediately*/
        if(!act->ActiveParticle || act->NumActiveHydro > 0.1 * (SlotsManager->info[0].size + SlotsManager->info[5].size)) {
            SPH_predicted.EntVarPred = (MyFloat *) mymalloc2("EntVarPred", sizeof(MyFloat) * SlotsManager->info[0].size);
            #pragma omp parallel for
            for(i = 0; i < PartManager->NumPart; i++)
                if(Part[i].Type == 0 && !Part[i].IsGarbage)
                    SPH_predicted.EntVarPred[Part[i].PI] = SPH_EntVarPred(i, times);
        }
        /* But if only some particles are active, the pow function in EntVarPred is slow and we have a lot of overhead, because we are doing 5500^3 exps for 5 particles.
        * So instead we compute it for active particles and use an atomic to guard the changes inside the loop.
        * For sufficiently small particle numbers the memset dominates and it is fastest to just compute each predicted entropy as we need it.*/
        else if(act->NumActiveHydro > 0.0001 * (SlotsManager->info[0].size + SlotsManager->info[5].size)){
            SPH_predicted.EntVarPred = (MyFloat *) mymalloc2("EntVarPred", sizeof(MyFloat) * SlotsManager->info[0].size);
            memset(SPH_predicted.EntVarPred, 0, sizeof(SPH_predicted.EntVarPred[0]) * SlotsManager->info[0].size);
            #pragma omp parallel for
            for(i = 0; i < act->NumActiveParticle; i++)
            {
                int p_i = act->ActiveParticle ? act->ActiveParticle[i] : i;
                if(Part[p_i].Type == 0 && !Part[p_i].IsGarbage)
                    SPH_predicted.EntVarPred[Part[p_i].PI] = SPH_EntVarPred(p_i, times);
            }
        }
    }

    ~DensityPriv()
    {
        if(GradRho)
            myfree(GradRho);
        myfree(DhsmlDensityFactor);
        myfree(Rot);
        myfree(NumNgb);
        myfree(Right);
        myfree(Left);
    }
};

/*! Structure for communication during the density computation. Holds data that is sent to other processors.
*/
class TreeWalkNgbIterDensity : public TreeWalkNgbIterBase{
    public:
        DensityKernel kernel;
        double kernel_volume;
};

class DensityQuery : public TreeWalkQueryBase<DensityPriv>
{
    public:
        double Vel[3];
        MyFloat Hsml;
        int Type;

        DensityQuery(const particle_data& particle, const int * const i_NodeList, const int firstnode, DensityPriv& priv):
        TreeWalkQueryBase<DensityPriv>(particle, i_NodeList, firstnode, priv), Hsml(particle.Hsml), Type(particle.Type)
        {
            if(particle.Type != 0)
            {
                Vel[0] = particle.Vel[0];
                Vel[1] = particle.Vel[1];
                Vel[2] = particle.Vel[2];
            }
            else
                priv.kf.SPH_VelPred(particle, Vel);
        }

};

class DensityResult : public TreeWalkResultBase<DensityPriv> {
    /*These are only used for density independent SPH*/
    public:
        MyFloat EgyRho;
        MyFloat DhsmlEgyDensity;

        MyFloat Rho;
        MyFloat DhsmlDensity;
        MyFloat Ngb;
        MyFloat Div;
        MyFloat Rot[3];
        /*Only used if sfr_need_to_compute_sph_grad_rho is true*/
        MyFloat GradRho[3];

        void reduce(int place, enum TreeWalkReduceMode mode, const DensityPriv& priv)
        {
            TREEWALK_REDUCE(priv.NumNgb[place], Ngb);
            TREEWALK_REDUCE(priv.DhsmlDensityFactor[place], DhsmlDensity);

            if(Part[place].Type == 0)
            {
                TREEWALK_REDUCE(SPHP(place).Density, Rho);

                TREEWALK_REDUCE(SPHP(place).DivVel, Div);
                int pi = Part[place].PI;
                TREEWALK_REDUCE(priv.Rot[pi][0], Rot[0]);
                TREEWALK_REDUCE(priv.Rot[pi][1], Rot[1]);
                TREEWALK_REDUCE(priv.Rot[pi][2], Rot[2]);

                MyFloat * gradrho = priv.GradRho;

                if(gradrho) {
                    TREEWALK_REDUCE(gradrho[3*pi], GradRho[0]);
                    TREEWALK_REDUCE(gradrho[3*pi+1], GradRho[1]);
                    TREEWALK_REDUCE(gradrho[3*pi+2], GradRho[2]);
                }

                /*Only used for density independent SPH*/
                if(priv.DoEgyDensity) {
                    TREEWALK_REDUCE(SPHP(place).EgyWtDensity, EgyRho);
                    TREEWALK_REDUCE(SPHP(place).DhsmlEgyDensityFactor, DhsmlEgyDensity);
                }
            }
            else if(Part[place].Type == 5)
            {
                TREEWALK_REDUCE(BHP(place).Density, Rho);
                TREEWALK_REDUCE(BHP(place).DivVel, Div);
            }
        }

};

class DensityLocalTreeWalk: LocalTreeWalk<TreeWalkNgbIterDensity, DensityQuery, DensityResult>
{
    public:
        /*
        *  This function represents the core of the SPH density computation.
        *
        *  The neighbours of the particle in the Query are enumerated, and results
        *  are stored into the Result object.
        *
        *  Upon start-up we initialize the iterator with the density kernels used in
        *  the computation. The assumption is the density kernels are slow to
        *  initialize.
        *
        */
        void ngbiter(
                DensityQuery& input,
                DensityResult * output,
                TreeWalkNgbIterDensity * iter)
        {
            if(iter->other == -1) {
                const double h = input.Hsml;
                density_kernel_init(&iter->kernel, h, DensityParams.DensityKernelType);
                iter->kernel_volume = density_kernel_volume(&iter->kernel);

                iter->Hsml = h;
                iter->mask = GASMASK; /* gas only */
                iter->symmetric = NGB_TREEFIND_ASYMMETRIC;
                return;
            }
            const int other = iter->other;
            const double r = iter->r;
            const double r2 = iter->r2;
            const double * dist = iter->dist;

            if(Part[other].Mass == 0) {
                endrun(12, "Density found zero mass particle %d type %d id %ld pos %g %g %g\n",
                    other, Part[other].Type, Part[other].ID, Part[other].Pos[0], Part[other].Pos[1], Part[other].Pos[2]);
            }

            if(r2 < iter->kernel.HH)
            {
                /* For the BH we wish to exclude wind particles from the density,
                * because they are excluded from the accretion treewalk.*/
                if(input.Type == 5 && winds_is_particle_decoupled(other))
                    return;

                const double u = r * iter->kernel.Hinv;
                const double wk = density_kernel_wk(&iter->kernel, u);
                output->Ngb += wk * iter->kernel_volume;

                const double dwk = density_kernel_dwk(&iter->kernel, u);

                const double mass_j = Part[other].Mass;

                output->Rho += (mass_j * wk);

                /* Hinv is here because O->DhsmlDensity is drho / dH.
                * nothing to worry here */
                double density_dW = density_kernel_dW(&iter->kernel, u, wk, dwk);
                output->DhsmlDensity += mass_j * density_dW;

                double EntVarPred;
                MyFloat VelPred[3];
                priv.kf.SPH_VelPred(other, VelPred);

                if(priv->SPH_predicted->EntVarPred) {
                    #pragma omp atomic read
                    EntVarPred = priv->SPH_predicted->EntVarPred[Part[other].PI];
                    /* Lazily compute the predicted quantities. We can do this
                    * with minimal locking since nothing happens should we compute them twice.
                    * Zero can be the special value since there should never be zero entropy.*/
                    if(EntVarPred == 0) {
                        EntVarPred = SPH_EntVarPred(other, priv->times);
                        #pragma omp atomic write
                        priv->SPH_predicted->EntVarPred[Part[other].PI] = EntVarPred;
                    }
                }
                else
                    EntVarPred = SPH_EntVarPred(other, priv->times);

                if(priv->DoEgyDensity) {
                    output->EgyRho += mass_j * EntVarPred * wk;
                    output->DhsmlEgyDensity += mass_j * EntVarPred * density_dW;
                }

                if(r > 0)
                {
                    double fac = mass_j * dwk / r;
                    double dv[3];
                    double rot[3];
                    int d;
                    for(d = 0; d < 3; d ++) {
                        dv[d] = input.Vel[d] - VelPred[d];
                    }
                    output->Div += -fac * dotproduct(dist, dv);

                    crossproduct(dv, dist, rot);
                    for(d = 0; d < 3; d ++) {
                        output->Rot[d] += fac * rot[d];
                    }
                    if(priv->GradRho) {
                        for (d = 0; d < 3; d ++)
                            output->GradRho[d] += fac * dist[d];
                    }
                }
            }
        }


};

class DensityTreeWalk: public TreeWalk<DensityQuery, DensityResult, DensityLocalTreeWalk, DensityPriv> {
    bool haswork(int n)
    {
        /* Don't want a density for swallowed black hole particles*/
        if(Part[n].Swallowed)
            return 0;
        if(Part[n].Type == 0 || Part[n].Type == 5)
            return 1;
        return 0;
    }

    /* Returns 1 if we are done and do not need to loop. 0 if we need to repeat.*/
    int
    density_check_neighbours (int i)
    {
        /* now check whether we had enough neighbours */
        int tid = omp_get_thread_num();
        double desnumngb = priv.DesNumNgb;

        if(priv.BlackHoleOn && Part[i].Type == 5)
            desnumngb = desnumngb * DensityParams.BlackHoleNgbFactor;

        MyFloat * Left = priv.Left;
        MyFloat * Right = priv.Right;
        MyFloat * NumNgb = priv.NumNgb;

        if(maxnumngb[tid] < NumNgb[i])
            maxnumngb[tid] = NumNgb[i];
        if(minnumngb[tid] > NumNgb[i])
            minnumngb[tid] = NumNgb[i];

        if(Niteration >= MAXITER - 5)
        {
             message(1, "i=%d ID=%lu Hsml=%g Left=%g Right=%g Ngbs=%g Right-Left=%g\n   pos=(%g|%g|%g)\n",
                 i, Part[i].ID, Part[i].Hsml, Left[i], Right[i],
                 NumNgb[i], Right[i] - Left[i], Part[i].Pos[0], Part[i].Pos[1], Part[i].Pos[2]);
        }

        if(NumNgb[i] < (desnumngb - DensityParams.MaxNumNgbDeviation) ||
                (NumNgb[i] > (desnumngb + DensityParams.MaxNumNgbDeviation)))
        {
            /* This condition is here to prevent the density code looping forever if it encounters
             * multiple particles at the same position. If this happens you likely have worse
             * problems anyway, so warn also. */
            if((Right[i] - Left[i]) < 1.0e-5 * Left[i])
            {
                /* If this happens probably the exchange is screwed up and all your particles have moved to (0,0,0)*/
                message(1, "Very tight Hsml bounds for i=%d ID=%lu Hsml=%g Left=%g Right=%g Ngbs=%g Right-Left=%g pos=(%g|%g|%g)\n",
                 i, Part[i].ID, Part[i].Hsml, Left[i], Right[i], NumNgb[i], Right[i] - Left[i], Part[i].Pos[0], Part[i].Pos[1], Part[i].Pos[2]);
                Part[i].Hsml = Right[i];
                return 1;
            }

            /* If we need more neighbours, move the lower bound up. If we need fewer, move the upper bound down.*/
            if(NumNgb[i] < desnumngb) {
                    Left[i] = Part[i].Hsml;
            } else {
                    Right[i] = Part[i].Hsml;
            }

            /* Next step is geometric mean of previous. */
            if((Right[i] < tree->BoxSize && Left[i] > 0) || (Part[i].Hsml * 1.26 > 0.99 * tree->BoxSize))
                Part[i].Hsml = cbrt(0.5 * (pow(Left[i], 3) + pow(Right[i], 3)));
            else
            {
                if(!(Right[i] < tree->BoxSize) && Left[i] == 0)
                    endrun(8188, "Cannot occur. Check for memory corruption: i=%d L = %g R = %g N=%g. Type %d, Pos %g %g %g hsml %g Box %g\n", i, Left[i], Right[i], NumNgb[i], Part[i].Type, Part[i].Pos[0], Part[i].Pos[1], Part[i].Pos[2], Part[i].Hsml, tree->BoxSize);

                MyFloat DensFac = priv.DhsmlDensityFactor[i];
                double fac = 1.26;
                if(NumNgb[i] > 0)
                    fac = 1 - (NumNgb[i] - desnumngb) / (NUMDIMS * NumNgb[i]) * DensFac;

                /* Find the initial bracket using the kernel gradients*/
                if(Right[i] > 0.99 * tree->BoxSize && Left[i] > 0)
                    if(DensFac <= 0 || fabs(NumNgb[i] - desnumngb) >= 0.5 * desnumngb || fac > 1.26)
                        fac = 1.26;

                if(Right[i] < 0.99*tree->BoxSize && Left[i] == 0)
                    if(DensFac <=0 || fac < 1./3)
                        fac = 1./3;

                Part[i].Hsml *= fac;
            }

            if(priv.BlackHoleOn && Part[i].Type == 5)
                if(Left[i] > DensityParams.BlackHoleMaxAccretionRadius)
                {
                    Part[i].Hsml = DensityParams.BlackHoleMaxAccretionRadius;
                    return 1;
                }

            if(Right[i] < priv.MinGasHsml) {
                Part[i].Hsml = priv.MinGasHsml;
                return 1;
            }
            /* More work needed: add this particle to the redo queue*/
            NPRedo[tid][NPLeft[tid]] = i;
            NPLeft[tid] ++;
            if(NPLeft[tid] > Redo_thread_alloc)
                endrun(5, "Particle %ld on thread %d exceeded allocated size of redo queue %ld\n", NPLeft[tid], tid, Redo_thread_alloc);
            return 0;
        }
        else {
            /* We might have got here by serendipity, without bounding.*/
            if(priv.BlackHoleOn && Part[i].Type == 5)
                if(Part[i].Hsml > DensityParams.BlackHoleMaxAccretionRadius)
                    Part[i].Hsml = DensityParams.BlackHoleMaxAccretionRadius;
            if(Part[i].Hsml < priv.MinGasHsml)
                Part[i].Hsml = priv.MinGasHsml;
            return 1;
        }
    }

    void
    postprocess(int i)
    {
        MyFloat * DhsmlDens = &(priv.DhsmlDensityFactor[i]);
        double density = -1;
        if(Part[i].Type == 0)
            density = SPHP(i).Density;
        else if(Part[i].Type == 5)
            density = BHP(i).Density;
        if(density <= 0 && priv.NumNgb[i] > 0) {
            endrun(12, "Particle %d type %d has bad density: %g\n", i, Part[i].Type, density);
        }
        *DhsmlDens *= Part[i].Hsml / (NUMDIMS * density);
        *DhsmlDens = 1 / (1 + *DhsmlDens);

        /* Uses DhsmlDensityFactor and changes Hsml, hence the location.*/
        if(priv.update_hsml) {
            int done = density_check_neighbours(i, tw);
            /* If we are done repeating, update the hmax in the parent node,
            * if that type is in the tree.*/
            if(done && (tree->mask & (1<<Part[i].Type)))
                update_tree_hmax_father(tree, i, Part[i].Pos, Part[i].Hsml);
        }

        if(Part[i].Type == 0)
        {
            int PI = Part[i].PI;
            /*Compute the EgyWeight factors, which are only useful for density independent SPH */
            if(priv.DoEgyDensity) {
                double EntPred;
                if(priv.SPH_predicted->EntVarPred)
                    EntPred = priv.SPH_predicted->EntVarPred[Part[i].PI];
                else
                    EntPred = SPH_EntVarPred(i, priv.times);
                if(EntPred <= 0 || SPHP(i).EgyWtDensity <=0)
                    endrun(12, "Particle %d has bad predicted entropy: %g or EgyWtDensity: %g, Particle ID = %ld, pos %g %g %g, vel %g %g %g, mass = %g, density = %g, MaxSignalVel = %g, Entropy = %g, DtEntropy = %g \n", i, EntPred, SPHP(i).EgyWtDensity, Part[i].ID, Part[i].Pos[0], Part[i].Pos[1], Part[i].Pos[2], Part[i].Vel[0], Part[i].Vel[1], Part[i].Vel[2], Part[i].Mass, SPHP(i).Density, SPHP(i).MaxSignalVel, SPHP(i).Entropy, SPHP(i).DtEntropy);
                SPHP(i).DhsmlEgyDensityFactor *= Part[i].Hsml/ (NUMDIMS * SPHP(i).EgyWtDensity);
                SPHP(i).DhsmlEgyDensityFactor *= - (*DhsmlDens);
                SPHP(i).EgyWtDensity /= EntPred;
            }
            else
                SPHP(i).DhsmlEgyDensityFactor = *DhsmlDens;

            MyFloat * Rot = priv.Rot[PI];
            SPHP(i).CurlVel = sqrt(Rot[0] * Rot[0] + Rot[1] * Rot[1] + Rot[2] * Rot[2]) / SPHP(i).Density;

            SPHP(i).DivVel /= SPHP(i).Density;
            Part[i].DtHsml = (1.0 / NUMDIMS) * SPHP(i).DivVel * Part[i].Hsml;
        }
        else if(Part[i].Type == 5)
        {
            BHP(i).DivVel /= BHP(i).Density;
            Part[i].DtHsml = (1.0 / NUMDIMS) * BHP(i).DivVel * Part[i].Hsml;
        }
    }
}

/*! \file density.c
 *  \brief SPH density computation and smoothing length determination
 *
 *  This file contains the "first SPH loop", where the SPH densities and some
 *  auxiliary quantities are computed.  There is also functionality that
 *  corrects the smoothing length if needed.
 */

/*! This function computes the local density for each active SPH particle, the
 * number of neighbours in the current smoothing radius, and the divergence
 * and rotation of the velocity field.  The pressure is updated as well.  If a
 * particle with its smoothing region is fully inside the local domain, it is
 * not exported to the other processors. The function also detects particles
 * that have a number of neighbours outside the allowed tolerance range. For
 * these particles, the smoothing length is adjusted accordingly, and the
 * density() computation is called again.  Note that the smoothing length is
 * not allowed to fall below the lower bound set by MinGasHsml (this may mean
 * that one has to deal with substantially more than normal number of
 * neighbours.)
 */
void
density2(const ActiveParticles * act, int update_hsml, int DoEgyDensity, int BlackHoleOn, const DriftKickTimes times, Cosmology * CP, struct sph_pred_data * SPH_predicted, MyFloat * GradRho_mag, const ForceTree * const tree)
{
    DensityPriv priv(update_hsml, DoEgyDensity, BlackHoleOn, &times, GradRho_mag, tree->BoxSize, CP, act, PartManager);
    TreeWalk tw(tree, "DENSITY", priv);

    //tw->visit = (TreeWalkVisitFunction) treewalk_visit_nolist_ngbiter;
    //tw->NoNgblist = 1;
    walltime_measure("/SPH/Density/Init");

    /* Do the treewalk with looping for hsml*/
    tw.do_hsml_loop(act->ActiveParticle, act->NumActiveParticle, update_hsml);

    if(GradRho_mag) {
        int64_t i;
        #pragma omp parallel for
        for(i = 0; i < SlotsManager->info[0].size; i++)
        {
            MyFloat * gr = tw.priv.GradRho + (3*i);
            GradRho_mag[i] = sqrt(gr[0]*gr[0] + gr[1] * gr[1] + gr[2] * gr[2]);
        }
    }

    /* collect some timing information */

    double timeall = walltime_measure(WALLTIME_IGNORE);
    double timecomp = tw.timecomp0 + tw.timecomp3 + tw.timecomp1 + tw.timecomp2;
    walltime_add("/SPH/Density/WalkTop", tw.timecomp0);
    walltime_add("/SPH/Density/WalkPrim", tw.timecomp1);
    walltime_add("/SPH/Density/WalkSec", tw.timecomp2);
    walltime_add("/SPH/Density/PostPre", tw.timecomp3);
    // walltime_add("/SPH/Density/Compute", timecomp);
    walltime_add("/SPH/Density/Wait", tw.timewait1);
    walltime_add("/SPH/Density/Reduce", tw.timecommsumm);
    walltime_add("/SPH/Density/Misc", timeall - (timecomp + tw.timewait1 + tw.timecommsumm));
}

void
slots_free_sph_pred_data(struct sph_pred_data * sph_scratch)
{
    if(sph_scratch->EntVarPred)
        myfree(sph_scratch->EntVarPred);
    sph_scratch->EntVarPred = NULL;
}

/* Set the initial smoothing length for gas and BH*/
void
set_init_hsml(ForceTree * tree, DomainDecomp * ddecomp, const double MeanGasSeparation)
{
    /* Need moments because we use them to set Hsml*/
    force_tree_calc_moments(tree, ddecomp);
    if(!tree->Father)
        endrun(5, "tree Father array not allocated at initial hsml!\n");
    const double DesNumNgb = GetNumNgb(GetDensityKernelType());
    int i;
    #pragma omp parallel for
    for(i = 0; i < PartManager->NumPart; i++)
    {
        /* These initial smoothing lengths are only used for SPH-like particles.*/
        if(Part[i].Type != 0 && Part[i].Type != 5)
            continue;

        if(Part[i].IsGarbage)
            continue;
        int no = i;

        do {
            int p = force_get_father(no, tree);

            if(p < tree->firstnode)
                break;

            /* Check that we didn't somehow get a bad set of nodes*/
            if(p > tree->numnodes + tree->firstnode)
                endrun(5, "Bad init father: i=%d, mass = %g type %d hsml %g no %d len %g father %d, numnodes %ld firstnode %ld\n",
                    i, Part[i].Mass, Part[i].Type, Part[i].Hsml, no, tree->Nodes[no].len, p, tree->numnodes, tree->firstnode);
            no = p;
        } while(10 * DesNumNgb * Part[i].Mass > tree->Nodes[no].mom.mass);

        /* Validate the tree node contents*/
        if(tree->Nodes[no].len > tree->BoxSize || tree->Nodes[no].mom.mass < Part[i].Mass)
            endrun(5, "Bad tree moments: i=%d, mass = %g type %d hsml %g no %d len %g treemass %g\n",
                    i, Part[i].Mass, Part[i].Type, Part[i].Hsml, no, tree->Nodes[no].len, tree->Nodes[no].mom.mass);
        Part[i].Hsml = MeanGasSeparation;
        if(no >= tree->firstnode) {
            double testhsml = tree->Nodes[no].len * pow(3.0 / (4 * M_PI) * DesNumNgb * Part[i].Mass / tree->Nodes[no].mom.mass, 1.0 / 3);
            /* recover from a poor initial guess */
            if (testhsml < 500. * MeanGasSeparation)
                Part[i].Hsml = testhsml;
        }

        if(Part[i].Hsml <= 0)
            endrun(5, "Bad hsml guess: i=%d, mass = %g type %d hsml %g no %d len %g treemass %g\n",
                    i, Part[i].Mass, Part[i].Type, Part[i].Hsml, no, tree->Nodes[no].len, tree->Nodes[no].mom.mass);
    }
}
