#ifndef __BLACKHOLE_H
#define __BLACKHOLE_H
#include "utils/paramset.h"
#include "forcetree.h"
#include "forcetree2.h"
#include "density.h"
#include "timebinmgr.h"
#include "utils/system.h"

/* Parameters of the black hole model*/
struct BlackholeParams
{
    double BlackHoleAccretionFactor;	/*!< Fraction of BH bondi accretion rate */
    double BlackHoleFeedbackFactor;	/*!< Fraction of the black luminosity feed into thermal feedback */
    double BlackHoleEddingtonFactor;	/*! Factor above Eddington */

    int BlackHoleKineticOn; /*If 1, perform AGN kinetic feedback when the Eddington accretion rate is low */
    double BHKE_EddingtonThrFactor; /*Threshold of the Eddington rate for the kinetic feedback*/
    double BHKE_EddingtonMFactor; /* Factor for mbh-dependent Eddington threshold */
    double BHKE_EddingtonMPivot; /* Pivot MBH for mbh-dependent Eddington threshold */
    double BHKE_EddingtonMIndex; /* Powlaw index for mbh-dependent Eddington threshold */
    double BHKE_EffRhoFactor; /* Minimum kinetic feedback efficiency factor scales with BH density*/
    double BHKE_EffCap; /* Cap of the kinetic feedback efficiency factor */
    double BHKE_InjEnergyThr; /*Minimum injection of KineticFeedbackEnergy, controls the burstiness of kinetic feedback*/
    double BHKE_SfrCritOverDensity; /*for KE efficiency calculation, borrow from sfr.params */
    /**********************************************************************/
    int MergeGravBound; /*if 1, apply gravitational bound criteria for BH mergers */
    int BH_DRAG; /*Hydro drag force*/

    double SeedBHDynMass; /* The initial dynamic mass of BH particle */

    double SeedBlackHoleMass;	/*!< (minimum) Seed black hole mass */
    double MaxSeedBlackHoleMass; /* Maximum black hole seed mass*/
    double SeedBlackHoleMassIndex; /* Power law index for BH seed mass*/
    /************************************************************************/
};
extern struct BlackholeParams blackhole_params;

struct BHPriv {
    /* Temporary array to store the IDs of the swallowing black hole for gas.
     * We store ID + 1 so that SwallowID == 0 can correspond to the unswallowed case. */
    MyIDType * SPH_SwallowID;
    /* Similar for IDs of BH mergers*/
    MyIDType * BH_SwallowID;
    /* These are temporaries used in the accretion treewalk*/
    MyFloat * BH_Entropy;
    MyFloat (*BH_SurroundingGasVel)[3];

    /* These are temporaries used in the feedback treewalk.*/
    MyFloat * BH_accreted_Mass;
    MyFloat * BH_accreted_BHMass;
    MyFloat (*BH_accreted_momentum)[3];

    /* This is a temporary computed in the accretion treewalk and used
     * in the feedback treewalk*/
    MyFloat * BH_FeedbackWeightSum;

    /* temporary computed for kinetic feedback energy threshold*/
    MyFloat * NumDM;
    MyFloat * MgasEnc;
    /* mark the state of AGN kinetic feedback, 1 accumulate, 2 release */
    int * KEflag;

    /* Time factors*/
    double atime;
    double a3inv;
    double hubble;
    struct UnitSystem units;
    Cosmology * CP;
    RandTable * rnd;
};

/*Set the parameters of the star formation module*/
void set_blackhole_params(ParameterSet * ps);

/* Does the black hole feedback and accretion.
 * TimeNextSeedingCheck is the time of the BH next seeding check.
 * It will be compared to the current time and updated after seeding takes place.
 * gasTree is a tree containing gas and black holes, used for the accretion and feedback treewalks. Its hmax is updated if necessary.
 */
void blackhole(const ActiveParticles * act, double atime, Cosmology * CP, ForceTreeHmax * gasTree, DomainDecomp * ddecomp, DriftKickTimes * times, TimeBinMgr * timebinmgr, RandTable * rnd, const struct UnitSystem units, FILE * FdBlackHoles, FILE * FdBlackholeDetails, size_t *bhdetailswritten);

/* Make a black hole from the particle at index. Random number generator used for the initial mass drawn from a power law.*/
void blackhole_make_one(int index, const double atime, const RandTable * const rnd);

#endif
