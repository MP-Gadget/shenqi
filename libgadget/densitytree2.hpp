#ifndef DENSITYTREE2_HPP
#define DENSITYTREE2_HPP
#include "density2.h"
#include "localtreewalk2.h"
#include "utils/mymalloc.h"
#include "winds.h"

class DensityPriv : public ParamTypeBase {
    public:
    bool update_hsml;
    /* Are there potentially black holes?*/
    bool BlackHoleOn;
    bool DoEgyDensity;

    DriftKickTimes times;
    enum DensityKernelType DensityKernelType;  /* 0 for Cubic Spline,  (recmd NumNgb = 33)
                               1 for Quintic spline (recmd  NumNgb = 97) */

    /*!< Desired number of SPH neighbours */
    double DesNumNgb;
    double DesNumNgbBH;
    /*!< minimum allowed SPH smoothing length */
    double MinGasHsml;
    /* For computing the predicted quantities dynamically during the treewalk.*/
    KickFactorData kf;
    /*!< Predicted entropy at current particle drift time for SPH computation*/
    MyFloat * EntVarPred;

    DensityPriv(const struct density_params DensityParams, const bool i_update_hsml, const bool i_DoEgyDensity, const bool i_BlackHoleOn, DriftKickTimes * i_times, const double BoxSize, Cosmology * CP, const ActiveParticles * const act, const struct part_manager_type * const i_PartManager):
    ParamTypeBase(i_PartManager->BoxSize), update_hsml(i_update_hsml), BlackHoleOn(i_BlackHoleOn), DoEgyDensity(i_DoEgyDensity), times(*i_times),
    DensityKernelType(DensityParams.DensityKernelType), DesNumNgb(GetNumNgb(DensityKernelType)), DesNumNgbBH(DesNumNgb * DensityParams.BlackHoleNgbFactor),
    MinGasHsml(DensityParams.MinGasHsmlFractional * (FORCE_SOFTENING()/2.8)), kf(i_times, CP), EntVarPred(NULL)
    {
        struct particle_data * parts = i_PartManager->Base;

        /* If all particles are active, easiest to compute all the predicted velocities immediately*/
        if(!act->ActiveParticle || act->NumActiveHydro > 0.1 * (SlotsManager->info[0].size + SlotsManager->info[5].size)) {
            EntVarPred = (MyFloat *) mymanagedmalloc("EntVarPred", sizeof(MyFloat) * SlotsManager->info[0].size);
            #pragma omp parallel for
            for(int64_t i = 0; i < PartManager->NumPart; i++)
                if(parts[i].Type == 0 && !parts[i].IsGarbage)
                    EntVarPred[parts[i].PI] = SPH_EntVarPred(parts[i], &times);
        }
        /* But if only some particles are active, the pow function in EntVarPred is slow and we have a lot of overhead, because we are doing 5500^3 exps for 5 particles.
        * So instead we compute it for active particles and use an atomic to guard the changes inside the loop.
        * For sufficiently small particle numbers the memset dominates and it is fastest to just compute each predicted entropy as we need it.*/
        else if(act->NumActiveHydro > 0.0001 * (SlotsManager->info[0].size + SlotsManager->info[5].size)){
            EntVarPred = (MyFloat *) mymanagedmalloc("EntVarPred", sizeof(MyFloat) * SlotsManager->info[0].size);
            memset(EntVarPred, 0, sizeof(EntVarPred[0]) * SlotsManager->info[0].size);
            #pragma omp parallel for
            for(int64_t i = 0; i < act->NumActiveParticle; i++)
            {
                int p_i = act->ActiveParticle ? act->ActiveParticle[i] : i;
                if(parts[p_i].Type == 0 && !parts[p_i].IsGarbage)
                    EntVarPred[parts[p_i].PI] = SPH_EntVarPred(parts[p_i], &times);
            }
        }
    }
};


class DensityOutput {
    public:
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
    /* Whether to output extra debugging information during postprocess. */
    double MaxNumNgbDeviation;
    bool verbose;

    DensityOutput(const bool GradRho_mag, const int64_t NumPart, const int64_t NumGasSlots, const double BoxSize, const double MaxNgbDeviation):
    MaxNumNgbDeviation(MaxNgbDeviation), verbose(false)
    {
        Left = (MyFloat *) mymanagedmalloc("DENS_PRIV->Left", NumPart * sizeof(MyFloat));
        Right = (MyFloat *) mymanagedmalloc("DENS_PRIV->Right", NumPart * sizeof(MyFloat));
        NumNgb = (MyFloat *) mymanagedmalloc("DENS_PRIV->NumNgb", NumPart * sizeof(MyFloat));
        Rot = (MyFloat (*) [3]) mymanagedmalloc("DENS_PRIV->Rot", NumGasSlots * sizeof(Rot[0]));
        /* This one stores the gradient for h finding. The factor stored in SPHP->DhsmlEgyDensityFactor depends on whether PE SPH is enabled.*/
        DhsmlDensityFactor = (MyFloat *) mymanagedmalloc("DhsmlDensity", NumPart * sizeof(MyFloat));
        if(GradRho_mag)
            GradRho = (MyFloat *) mymanagedmalloc("SPH_GradRho", sizeof(MyFloat) * 3 * NumGasSlots);
        else
            GradRho = NULL;

        /* Init Left and Right: this has to be done before treewalk */
        #pragma omp parallel for
        for(int64_t i = 0; i < NumPart; i++)  {
            Right[i] = BoxSize;
            NumNgb[i] = 0;
            Left[i] = 0;
        }
    }

    ~DensityOutput()
    {
        if(GradRho)
            myfree(GradRho);
        myfree(DhsmlDensityFactor);
        myfree(Rot);
        myfree(NumNgb);
        myfree(Right);
        myfree(Left);
    }

    MYCUDAFN int
    postprocess(const int i, struct particle_data * const parts, const DensityPriv * priv)
    {
        int done = 0;
        MyFloat * DhsmlDens = &(DhsmlDensityFactor[i]);
        double density = -1;
        if(parts[i].Type == 0)
            density = SphP[parts[i].PI].Density;
        else if(parts[i].Type == 5)
            density = BhP[parts[i].PI].Density;
        if(density <= 0 && NumNgb[i] > 0) {
            endrun(12, "Particle %d type %d has bad density: %g\n", i, parts[i].Type, density);
        }
        *DhsmlDens *= parts[i].Hsml / (NUMDIMS * density);
        *DhsmlDens = 1 / (1 + *DhsmlDens);

        /* Uses DhsmlDensityFactor and changes Hsml, hence the location.*/
        if(priv->update_hsml) {
            done = density_check_neighbours(i, verbose, parts, priv);
        }

        if(parts[i].Type == 0)
        {
            int PI = parts[i].PI;
            /*Compute the EgyWeight factors, which are only useful for density independent SPH */
            if(priv->DoEgyDensity) {
                double EntPred;
                if(priv->EntVarPred)
                    EntPred = priv->EntVarPred[parts[i].PI];
                else
                    EntPred = SPH_EntVarPred(parts[i], &priv->times);
                if(EntPred <= 0 || SphP[parts[i].PI].EgyWtDensity <=0)
                    endrun(12, "Particle %d has bad predicted entropy: %g or EgyWtDensity: %g, Particle ID = %ld, pos %g %g %g, vel %g %g %g, mass = %g, density = %g, MaxSignalVel = %g, Entropy = %g, DtEntropy = %g \n", i, EntPred, SphP[parts[i].PI].EgyWtDensity, parts[i].ID, parts[i].Pos[0], parts[i].Pos[1], parts[i].Pos[2], parts[i].Vel[0], parts[i].Vel[1], parts[i].Vel[2], parts[i].Mass, SphP[parts[i].PI].Density, SphP[parts[i].PI].MaxSignalVel, SphP[parts[i].PI].Entropy, SphP[parts[i].PI].DtEntropy);
                SphP[parts[i].PI].DhsmlEgyDensityFactor *= parts[i].Hsml/ (NUMDIMS * SphP[parts[i].PI].EgyWtDensity);
                SphP[parts[i].PI].DhsmlEgyDensityFactor *= - (*DhsmlDens);
                SphP[parts[i].PI].EgyWtDensity /= EntPred;
            }
            else
                SphP[parts[i].PI].DhsmlEgyDensityFactor = *DhsmlDens;

            MyFloat * Roti = Rot[PI];
            SphP[parts[i].PI].CurlVel = sqrt(Roti[0] * Roti[0] + Roti[1] * Roti[1] + Roti[2] * Roti[2]) / SphP[parts[i].PI].Density;

            SphP[parts[i].PI].DivVel /= SphP[parts[i].PI].Density;
            parts[i].DtHsml = (1.0 / NUMDIMS) * SphP[parts[i].PI].DivVel * parts[i].Hsml;
        }
        else if(parts[i].Type == 5)
        {
            BhP[parts[i].PI].DivVel /= BhP[parts[i].PI].Density;
            parts[i].DtHsml = (1.0 / NUMDIMS) * BhP[parts[i].PI].DivVel * parts[i].Hsml;
        }
        return done;
    }

    /* Returns 1 if we are done and do not need to loop. 0 if we need to repeat.*/
    MYCUDAFN int
    density_check_neighbours (const int i, const int verbose, struct particle_data * const parts, const DensityPriv * priv)
    {
        /* now check whether we had enough neighbours */
        double desnumngb = priv->DesNumNgb;

        if(priv->BlackHoleOn && parts[i].Type == 5)
            desnumngb = priv->DesNumNgbBH;

        if(verbose)
        {
             message(1, "i=%d ID=%lu Hsml=%g Left=%g Right=%g Ngbs=%g (des %g) Right-Left=%g pos=(%g|%g|%g)\n",
                 i, parts[i].ID, parts[i].Hsml, Left[i], Right[i],
                 NumNgb[i], desnumngb, Right[i] - Left[i], parts[i].Pos[0], parts[i].Pos[1], parts[i].Pos[2]);
        }

        if(NumNgb[i] < (desnumngb - MaxNumNgbDeviation) ||
                (NumNgb[i] > (desnumngb + MaxNumNgbDeviation)))
        {
            /* This condition is here to prevent the density code looping forever if it encounters
             * multiple particles at the same position. If this happens you likely have worse
             * problems anyway, so warn also. */
            if((Right[i] - Left[i]) < 1.0e-5 * Right[i])
            {
                /* If this happens probably the exchange is screwed up and all your particles have moved to (0,0,0)*/
                message(1, "Very tight Hsml bounds for i=%d ID=%lu Hsml=%g Left=%g Right=%g Ngbs=%g (des %g) Right-Left=%g pos=(%g|%g|%g) \n",
                 i, parts[i].ID, parts[i].Hsml, Left[i], Right[i], NumNgb[i], desnumngb, Right[i] - Left[i], parts[i].Pos[0], parts[i].Pos[1], parts[i].Pos[2]);
                parts[i].Hsml = Right[i];
                return 1;
            }

            /* If we need more neighbours, move the lower bound up. If we need fewer, move the upper bound down.*/
            if(NumNgb[i] < desnumngb) {
                    Left[i] = parts[i].Hsml;
            } else {
                    Right[i] = parts[i].Hsml;
            }

            /* Next step is geometric mean of previous. */
            if((Right[i] < priv->BoxSize && Left[i] > 0) || (parts[i].Hsml * 1.26 > 0.99 * priv->BoxSize))
                parts[i].Hsml = cbrt(0.5 * (pow(Left[i], 3) + pow(Right[i], 3)));
            else
            {
                if(!(Right[i] < priv->BoxSize) && Left[i] == 0)
                    endrun(8188, "Cannot occur. Check for memory corruption: i=%d L = %g R = %g N=%g. Type %d, Pos %g %g %g hsml %g Box %g\n", i, Left[i], Right[i], NumNgb[i], parts[i].Type, parts[i].Pos[0], parts[i].Pos[1], parts[i].Pos[2], parts[i].Hsml, priv->BoxSize);

                MyFloat DensFac = DhsmlDensityFactor[i];
                double fac = 1.26;
                if(NumNgb[i] > 0)
                    fac = 1 - (NumNgb[i] - desnumngb) / (NUMDIMS * NumNgb[i]) * DensFac;

                /* Find the initial bracket using the kernel gradients*/
                if(Right[i] > 0.99 * priv->BoxSize && Left[i] > 0)
                    if(DensFac <= 0 || fabs(NumNgb[i] - desnumngb) >= 0.5 * desnumngb || fac > 1.26)
                        fac = 1.26;

                if(Right[i] < 0.99*priv->BoxSize && Left[i] == 0)
                    if(DensFac <=0 || fac < 1./3)
                        fac = 1./3;

                parts[i].Hsml *= fac;
            }

            if(Right[i] < priv->MinGasHsml) {
                parts[i].Hsml = priv->MinGasHsml;
                return 1;
            }

            /* We need to repeat the particle! */
            return 0;
        }
        else {
            /* We might have got here by serendipity, without bounding.*/
            if(parts[i].Hsml < priv->MinGasHsml)
                parts[i].Hsml = priv->MinGasHsml;
            return 1;
        }
    }
};

class DensityQuery : public TreeWalkQueryBase<DensityPriv>
{
    public:
        double Vel[3];
        MyFloat Hsml;
        int Type;

        MYCUDAFN DensityQuery(const particle_data& particle, const int * const i_NodeList, const int firstnode, const DensityPriv& priv):
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
        };

        static MYCUDAFN bool haswork(const particle_data& particle)
        {
            /* Don't want a density for swallowed black hole particles*/
            if(!TreeWalkQueryBase::haswork(particle))
                return false;
            if(particle.Type == 0 || particle.Type == 5)
                return true;
            return false;
        };
};

class DensityResult : public TreeWalkResultBase<DensityQuery, DensityOutput> {
    /*These are only used for density independent SPH*/
    public:
        MyFloat EgyRho = 0;
        MyFloat DhsmlEgyDensity = 0;

        MyFloat Rho = 0;
        MyFloat DhsmlDensity = 0;
        MyFloat Ngb = 0;
        MyFloat Div = 0;
        MyFloat Rot[3] = {0};
        /*Only used if sfr_need_to_compute_sph_grad_rho is true*/
        MyFloat GradRho[3] = {0};
        MYCUDAFN DensityResult(DensityQuery& query): TreeWalkResultBase(query),
        EgyRho(0), DhsmlEgyDensity(0), Rho(0), DhsmlDensity(0), Ngb(0), Div(0)
        {}

        template<TreeWalkReduceMode mode>
        MYCUDAFN void reduce(int place, const DensityOutput * output, struct particle_data * const parts)
        {
            TreeWalkResultBase::reduce<mode>(place, output, parts);
            TREEWALK_REDUCE(output->NumNgb[place], Ngb);
            TREEWALK_REDUCE(output->DhsmlDensityFactor[place], DhsmlDensity);

            if(parts[place].Type == 0)
            {
                TREEWALK_REDUCE(SphP[parts[place].PI].Density, Rho);

                TREEWALK_REDUCE(SphP[parts[place].PI].DivVel, Div);
                int pi = parts[place].PI;
                TREEWALK_REDUCE(output->Rot[pi][0], Rot[0]);
                TREEWALK_REDUCE(output->Rot[pi][1], Rot[1]);
                TREEWALK_REDUCE(output->Rot[pi][2], Rot[2]);

                MyFloat * gradrho = output->GradRho;

                if(gradrho) {
                    TREEWALK_REDUCE(gradrho[3*pi], GradRho[0]);
                    TREEWALK_REDUCE(gradrho[3*pi+1], GradRho[1]);
                    TREEWALK_REDUCE(gradrho[3*pi+2], GradRho[2]);
                }

                /*Only used for density independent SPH*/
                //if(priv.DoEgyDensity) {
                    TREEWALK_REDUCE(SphP[parts[place].PI].EgyWtDensity, EgyRho);
                    TREEWALK_REDUCE(SphP[parts[place].PI].DhsmlEgyDensityFactor, DhsmlEgyDensity);
                //}
            }
            else if(parts[place].Type == 5)
            {
                TREEWALK_REDUCE(BhP[parts[place].PI].Density, Rho);
                TREEWALK_REDUCE(BhP[parts[place].PI].DivVel, Div);
            }
        }

};

/* Explicitly define the template specialisations and use the base constructors.
 * This is an asymmetric treewalk and defines the ngb iter function for the density.
 */
class DensityLocalTreeWalk: public LocalNgbTreeWalk<DensityLocalTreeWalk, DensityQuery, DensityResult, DensityPriv, NGB_TREEFIND_ASYMMETRIC, GASMASK>
{
    public:
        DensityKernel kernel;
        double kernel_volume;

        MYCUDAFN DensityLocalTreeWalk(const NODE * const Nodes, const DensityQuery& input): LocalNgbTreeWalk(Nodes, input)
        {
            density_kernel_init(&kernel, input.Hsml, DensityParams.DensityKernelType);
            kernel_volume = density_kernel_volume(&kernel);
            return;
        }
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
        MYCUDAFN void ngbiter(const DensityQuery& input, const int other, DensityResult * output, const DensityPriv& priv, const struct particle_data * const parts)
        {
            const particle_data& particle = parts[other];
            if(particle.Mass == 0) {
                endrun(12, "Density found zero mass particle %d type %d id %ld pos %g %g %g\n",
                    other, particle.Type, particle.ID, particle.Pos[0], particle.Pos[1], particle.Pos[2]);
            }

            /* We are too far away from the kernel */
            if(r2 >= kernel.HH)
                return;

            /* For the BH we wish to exclude wind particles from the density,
             * because they are excluded from the accretion treewalk.*/
            if(input.Type == 5 && winds_is_particle_decoupled(other))
                return;

            const double r = sqrt(r2);
            const double u = r * kernel.Hinv;
            const double wk = density_kernel_wk(&kernel, u);
            output->Ngb += wk * kernel_volume;

            const double dwk = density_kernel_dwk(&kernel, u);

            const double mass_j = particle.Mass;

            output->Rho += (mass_j * wk);

            /* Hinv is here because O->DhsmlDensity is drho / dH.
            * nothing to worry here */
            double density_dW = density_kernel_dW(&kernel, u, wk, dwk);
            output->DhsmlDensity += mass_j * density_dW;

            double EntVarPred;
            MyFloat VelPred[3];
            priv.kf.SPH_VelPred(particle, VelPred);

            if(priv.EntVarPred) {
                #pragma omp atomic read
                EntVarPred = priv.EntVarPred[particle.PI];
                /* Lazily compute the predicted quantities. We can do this
                * with minimal locking since nothing happens should we compute them twice.
                * Zero can be the special value since there should never be zero entropy.*/
                if(EntVarPred == 0) {
                    EntVarPred = SPH_EntVarPred(particle, &priv.times);
                    #pragma omp atomic write
                    priv.EntVarPred[particle.PI] = EntVarPred;
                }
            }
            else
                EntVarPred = SPH_EntVarPred(particle, &priv.times);

            if(priv.DoEgyDensity) {
                output->EgyRho += mass_j * EntVarPred * wk;
                output->DhsmlEgyDensity += mass_j * EntVarPred * density_dW;
            }

            if(r <= 0)
                return;

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
            for (d = 0; d < 3; d ++)
                output->GradRho[d] += fac * dist[d];
        }
};

class DensityTopTreeWalk: public TopTreeWalk<DensityQuery, DensityPriv, NGB_TREEFIND_ASYMMETRIC> { using TopTreeWalk::TopTreeWalk; };

#ifdef USE_CUDA
void density_cuda(const ActiveParticles * act, const ForceTree * tree, DensityPriv * priv, DensityOutput * output, particle_data * const parts, int update_hsml, MPI_Comm comm);
#endif

#endif
