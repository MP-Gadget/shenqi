/*Tests for the cooling rates module, ported from python.*/
#define BOOST_TEST_MODULE cooling_rates

#include "booststub.h"

#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <libgadget/physconst.h>
#include <libgadget/cooling_rates.h>
#include <libgadget/config.h>

//Stub
int
during_helium_reionization(double redshift)
{
    return 0;
}

double recomb_alphaHp(double temp);

#define NEXACT 20
/*For hydrogen recombination we have an exact answer from Ferland et al 1992 (http://adsabs.harvard.edu/abs/1992ApJ...387...95F).
This function returns as an array these rates, for testing purposes.*/
//case B recombination rates for hydrogen from Ferland 92, final column of Table 1. For n >= 2.
static const double f92g2[NEXACT] = {5.758e-11, 2.909e-11, 1.440e-11, 6.971e-12,3.282e-12, 1.489e-12, 6.43e-13, 2.588e-13, 9.456e-14, 3.069e-14, 8.793e-15, 2.245e-15, 5.190e-16, 1.107e-16, 2.221e-17, 4.267e-18, 7.960e-19, 1.457e-19,2.636e-20, 4.737e-21};
//case B recombination rates for hydrogen from Ferland 92, second column of Table 1. For n == 1.
static const double f92n1[NEXACT] = {9.258e-12, 5.206e-12, 2.927e-12, 1.646e-12, 9.246e-13, 5.184e-13, 2.890e-13, 1.582e-13, 8.255e-14, 3.882e-14, 1.545e-14, 5.058e-15, 1.383e-15, 3.276e-16, 7.006e-17, 1.398e-17, 2.665e-18, 4.940e-19, 9.001e-20, 1.623e-20};
static const double temps[NEXACT] = {3.16227766e+00, 1.0e+01, 3.16227766e+01, 1.0e+02, 3.16227766e+02, 1.00e+03, 3.16227766e+03, 1.e+04, 3.16227766e+04, 1.e+05, 3.16227766e+05, 1.e+06, 3.16227766e+06, 1.0e+07, 3.16227766e+07, 1.0e+08, 3.16227766e+08, 1.0e+09, 3.16227766e+09, 1.0e+10};

static struct cooling_params get_test_coolpar(void)
{
    static struct cooling_params coolpar;
    coolpar.CMBTemperature = 2.7255;
    coolpar.PhotoIonizeFactor = 1;
    coolpar.SelfShieldingOn = 1;
    coolpar.fBar = 0.17;
    coolpar.PhotoIonizationOn = 1;
    coolpar.recomb = Verner96;
    coolpar.cooling = Sherwood;
    coolpar.UVRedshiftThreshold = -1;
    coolpar.MinGasTemp = 100;
    coolpar.HeliumHeatOn = 0;
    coolpar.HydrogenHeatAmp = 0;
    return coolpar;
}

/*Test the recombination rates*/
BOOST_AUTO_TEST_CASE(test_recomb_rates)
{
    struct cooling_params coolpar = get_test_coolpar();
    int i;
    const char * TreeCool = GADGET_TESTDATA_ROOT "/examples/TREECOOL_ep_2018p";
    const char * MetalCool = "";

    Cosmology CP = {0};
    CP.OmegaCDM = 0.3;
    CP.OmegaBaryon = coolpar.fBar * CP.OmegaCDM;
    CP.HubbleParam = 0.7;

    set_coolpar(coolpar);
    init_cooling_rates(TreeCool,NULL,MetalCool,&CP);
    for(i=0; i< NEXACT; i++) {
        BOOST_TEST(recomb_alphaHp(temps[i]) == (f92g2[i]+f92n1[i]), tt::tolerance(0.01));
    }

    coolpar.recomb = Cen92;
    set_coolpar(coolpar);
    init_cooling_rates(TreeCool,NULL,MetalCool,&CP);
    /*Cen rates are not very accurate.*/
    for(i=4; i< 12; i++) {
        BOOST_TEST(recomb_alphaHp(temps[i]) == (f92g2[i]+f92n1[i]), tt::tolerance(0.5));
    }
}

/*Test that the UVBG loading code works*/
BOOST_AUTO_TEST_CASE(test_uvbg_loader)
{
    struct cooling_params coolpar = get_test_coolpar();
    coolpar.SelfShieldingOn = 1;
    const char * TreeCool = GADGET_TESTDATA_ROOT "/examples/TREECOOL_ep_2018p";
    const char * MetalCool = "";
    set_coolpar(coolpar);
    Cosmology CP = {0};
    CP.OmegaCDM = 0.3;
    CP.OmegaBaryon = coolpar.fBar * CP.OmegaCDM;
    CP.HubbleParam = 0.7;
    init_cooling_rates(TreeCool,NULL,MetalCool,&CP);
    /*Test sensible rates at high redshift*/
    struct UVBG uvbg = get_global_UVBG(16);
    BOOST_TEST(uvbg.epsH0 == 0);
    BOOST_TEST(uvbg.self_shield_dens > 1e8);
    BOOST_TEST(uvbg.gJH0 == 0);
    /*Test at zero redshift*/
    uvbg = get_global_UVBG(0);
    BOOST_TEST(uvbg.epsH0 == 3.65296e-25, tt::tolerance(1e-5));
    BOOST_TEST(uvbg.epsHe0 == 3.98942e-25, tt::tolerance(1e-5));
    BOOST_TEST(uvbg.epsHep == 3.33253e-26, tt::tolerance(1e-5));
    BOOST_TEST(uvbg.gJH0 == 6.06e-14, tt::tolerance(1e-5));
    BOOST_TEST(uvbg.gJHe0 == 3.03e-14, tt::tolerance(1e-5));
    BOOST_TEST(uvbg.gJHep == 1.1e-15, tt::tolerance(1e-5));
    BOOST_TEST(uvbg.self_shield_dens == 0.0010114161149989826, tt::tolerance(1e-5));

    // Test at intermediate redshift
    uvbg = get_global_UVBG(3.);
    //message(0, "uvbg %g %g %g %g %g %g %g\n", uvbg.gJH0, uvbg.gJHe0, uvbg.gJHep,  uvbg.epsH0, uvbg.epsHe0, uvbg.epsHep, uvbg.self_shield_dens);
    BOOST_TEST(uvbg.epsH0 == 5.96570906168362e-24, tt::tolerance(1e-5));
    BOOST_TEST(uvbg.epsHe0 == 4.466976578202419e-24, tt::tolerance(1e-5));
    BOOST_TEST(uvbg.epsHep == 2.758535690259892e-26, tt::tolerance(1e-5));
    BOOST_TEST(uvbg.gJH0 == 1.0549960730284017e-12, tt::tolerance(1e-5));
    BOOST_TEST(uvbg.gJHe0 == 4.759025257653999e-13, tt::tolerance(1e-5));
    BOOST_TEST(uvbg.gJHep == 2.270599708640625e-16, tt::tolerance(1e-5));
    BOOST_TEST(uvbg.self_shield_dens == 0.007691709693529007, tt::tolerance(1e-5));
}

/* Simple tests for the rate network */
BOOST_AUTO_TEST_CASE(test_rate_network)
{
    struct cooling_params coolpar = get_test_coolpar();
    const char * TreeCool = GADGET_TESTDATA_ROOT "/examples/TREECOOL_ep_2018p";
    const char * MetalCool = "";

    set_coolpar(coolpar);
    Cosmology CP = {0};
    CP.OmegaCDM = 0.3;
    CP.OmegaBaryon = coolpar.fBar * CP.OmegaCDM;
    CP.HubbleParam = 0.7;

    init_cooling_rates(TreeCool,NULL,MetalCool,&CP);

    struct UVBG uvbg = get_global_UVBG(2);

    //Complete ionisation at low density
    double logt;
    BOOST_TEST(get_equilib_ne(1e-6, 200.*1e10, 0.24, &logt, &uvbg, 1) / (1e-6*0.76)  == (1 + 2* 0.24/(1-0.24)/4), tt::tolerance(3e-5));
    BOOST_TEST(get_equilib_ne(1e-6, 200. * 1e10, 0.12, &logt, &uvbg, 1) / (1e-6 * 0.88) == (1 + 2 * 0.12 / (1 - 0.12) / 4), tt::tolerance(3e-5));
    BOOST_TEST(get_equilib_ne(1e-5, 200. * 1e10, 0.24, &logt, &uvbg, 1) / (1e-5 * 0.76) == (1 + 2 * 0.24 / (1 - 0.24) / 4), tt::tolerance(3e-4));
    BOOST_TEST(get_equilib_ne(1e-4, 200. * 1e10, 0.24, &logt, &uvbg, 1) / (1e-4 * 0.76) == (1 + 2 * 0.24 / (1 - 0.24) / 4), tt::tolerance(2e-3));

    double ne = 1.;
    double temp = get_temp(1e-4, 200.*1e10,0.24, &uvbg, &ne);
    BOOST_TEST(9500 < temp);
    BOOST_TEST(temp < 9510);
    // Roughly proportional to internal energy when ionised
    BOOST_TEST(get_temp(1e-4, 400. * 1e10, 0.24, &uvbg, &ne) == 2 * get_temp(1e-4, 200. * 1e10, 0.24, &uvbg, &ne), tt::tolerance(1e-3));
    BOOST_TEST(get_temp(1, 200. * 1e10, 0.24, &uvbg, &ne) == 14700, tt::tolerance(200.));

    // Neutral fraction proportional to density
    double dens[3] = {1e-5, 1e-6, 1e-7};
    for (int i = 0; i < 3; i++) {
        BOOST_TEST(get_neutral_fraction_phys_cgs(dens[i], 200. * 1e10, 0.24, &uvbg, &ne) == dens[i] * 0.3113, tt::tolerance(1e-3));
    }

    // Neutral (self-shielded) at high density:
    BOOST_TEST(get_neutral_fraction_phys_cgs(1, 100., 0.24, &uvbg, &ne) > 0.95);
    BOOST_TEST(0.75 > get_neutral_fraction_phys_cgs(0.1, 100. * 1e10, 0.24, &uvbg, &ne));
    BOOST_TEST(get_neutral_fraction_phys_cgs(0.1, 100. * 1e10, 0.24, &uvbg, &ne) > 0.735);

    //Check self-shielding is working.
    coolpar.SelfShieldingOn = 0;
    set_coolpar(coolpar);
    init_cooling_rates(TreeCool,NULL,MetalCool,&CP);

    BOOST_TEST( get_neutral_fraction_phys_cgs(1, 100.*1e10,0.24, &uvbg, &ne) < 0.25);
    BOOST_TEST( get_neutral_fraction_phys_cgs(0.1, 100.*1e10,0.24, &uvbg, &ne) <0.05);
}

/* This test checks that the heating and cooling rate is as expected.
 * In particular the physical density threshold is checked. */
BOOST_AUTO_TEST_CASE(test_heatingcooling_rate)
{
    struct cooling_params coolpar = get_test_coolpar();
    coolpar.recomb = Cen92;
    coolpar.cooling = KWH92;
    coolpar.SelfShieldingOn = 0;

    const char * TreeCool = GADGET_TESTDATA_ROOT "/examples/TREECOOL_ep_2018p";
    const char * MetalCool = "";

    /*unit system*/
    double HubbleParam = 0.697;
    double UnitDensity_in_cgs = 6.76991e-22;
    double UnitTime_in_s = 3.08568e+16;
    double UnitMass_in_g = 1.989e+43;
    double UnitLength_in_cm = 3.08568e+21;
    double UnitEnergy_in_cgs = UnitMass_in_g  * pow(UnitLength_in_cm, 2) / pow(UnitTime_in_s, 2);
    Cosmology CP = {0};
    CP.OmegaCDM = 0.3;
    CP.OmegaBaryon = coolpar.fBar * CP.OmegaCDM;
    CP.HubbleParam = HubbleParam;

    set_coolpar(coolpar);
    init_cooling_rates(TreeCool,NULL,MetalCool,&CP);

    struct cooling_units coolunits;
    coolunits.CoolingOn = 1;
    coolunits.density_in_phys_cgs = UnitDensity_in_cgs * HubbleParam * HubbleParam;
    coolunits.uu_in_cgs = UnitEnergy_in_cgs / UnitMass_in_g;
    coolunits.tt_in_s = UnitTime_in_s / HubbleParam;

    /*Default values from sfr_eff.c. Some dependence on HubbleParam, so don't change it.*/
    double egyhot = 2104.92;
    egyhot *= coolunits.uu_in_cgs;

    /* convert to physical cgs units */
    double dens = 0.027755;
    dens *= coolunits.density_in_phys_cgs/PROTONMASS;
    double ne = 1.0;

    struct UVBG uvbg = {0};
    /* XXX: We set the threshold without metal cooling
     * and with zero ionization at z=0.
     * It probably make sense to set the parameters with
     * a metalicity dependence. */
    double LambdaNet = get_heatingcooling_rate(dens, egyhot, 1 - HYDROGEN_MASSFRAC, 0, 0, &uvbg, &ne);

    double tcool = egyhot / (- LambdaNet);

    /*Convert back to internal units*/
    tcool /= coolunits.tt_in_s;

    //message(1, "tcool = %g LambdaNet = %g ne=%g\n", tcool, LambdaNet, ne);
    /* This differs by 0.13% from the old cooling code number,
     * apparently just because of rounding errors. The excitation cooling
     * numbers from Cen are not accurate to better than 1% anyway, so don't worry about it*/
    BOOST_TEST(tcool  == 4.68906e-06, tt::tolerance(1e-3));

    /*Now check that we get the desired cooling rate with a UVB*/
    uvbg = get_global_UVBG(0);

    BOOST_TEST(uvbg.epsHep > 0);
    BOOST_TEST(uvbg.gJHe0 > 0);

    dens /= 100;
    LambdaNet = get_heatingcooling_rate(dens, egyhot/10., 1 - HYDROGEN_MASSFRAC, 0, 0, &uvbg, &ne);
    //message(1, "LambdaNet = %g, uvbg=%g\n", LambdaNet, uvbg.epsHep);
    BOOST_TEST(LambdaNet == (-0.0410059), tt::tolerance(1e-3));

    LambdaNet = get_heatingcooling_rate(dens/2.5, egyhot/10., 1 - HYDROGEN_MASSFRAC, 0, 0, &uvbg, &ne);
    BOOST_TEST(LambdaNet > 0);
    /*Check self-shielding affects the cooling rates*/
    coolpar.SelfShieldingOn = 1;
    set_coolpar(coolpar);
    init_cooling_rates(TreeCool,NULL,MetalCool,&CP);
    LambdaNet = get_heatingcooling_rate(dens*1.5, egyhot/10., 1 - HYDROGEN_MASSFRAC, 0, 0, &uvbg, &ne);
    //message(1, "LambdaNet = %g, uvbg=%g\n", LambdaNet, uvbg.epsHep);
    BOOST_TEST(!(LambdaNet > 0));
    BOOST_TEST(LambdaNet == (-1.64834), tt::tolerance(1e-3));
}

#if 0
/* This test checks that the heating and cooling rate is as expected.
 * In particular the physical density threshold is checked. */
static void test_heatingcooling_rate_sherwood(void ** state)
{
    struct cooling_params coolpar = get_test_coolpar();
    coolpar.recomb = Verner96;
    coolpar.cooling = Sherwood;
//     coolpar.recomb = Cen92;
//     coolpar.cooling = KWH92;

    coolpar.SelfShieldingOn = 0;
    coolpar.MinGasTemp = 0;

    const char * TreeCool = GADGET_TESTDATA_ROOT "/examples/TREECOOL_ep_2018p";
    const char * MetalCool = "";

    /*unit system*/
    double HubbleParam = 0.679;
    Cosmology CP = {0};
    CP.OmegaCDM = 0.264;
    CP.OmegaBaryon = coolpar.fBar * CP.OmegaCDM;
    CP.HubbleParam = HubbleParam;

    set_coolpar(coolpar);
    init_cooling_rates(TreeCool,NULL,MetalCool,&CP);

    /* temp at mean cosmological density */
    double rhocb = CP.OmegaBaryon * 3.0 * pow(CP.HubbleParam*HUBBLE,2.0) /(8.0*M_PI*GRAVITY)/PROTONMASS;
    double ne = rhocb;

    /* Loop over redshift*/
    /* Now check that we get the desired cooling rate with a UVB*/
    struct UVBG uvbg = get_global_UVBG(2);
    double ienergy = 2.105e12;
    double temp = get_temp(rhocb, ienergy, 1 - HYDROGEN_MASSFRAC, &uvbg, &ne);
    int i;
    FILE * fd = fopen("cooling_rates_sherwood.txt", "w");
    fprintf(fd, "#density = %g temp = %g\n", rhocb, temp);
    fprintf(fd, "#zz LambdaNet Heat FF Collis Recomb Cmptn temp ne\n");
    FILE * fd2 = fopen("ion_state.txt", "w");
    fprintf(fd2, "#zz nhcgs nH0 nHe0 nHep nHepp\n");
    for(i = 0; i < 500; i++)
    {
        double zz = i * 6./ 500.;
        uvbg = get_global_UVBG(zz);
        double dens = rhocb * pow(1+zz,3);
        double LambdaNet = get_heatingcooling_rate(dens, ienergy, 1 - HYDROGEN_MASSFRAC, zz, 0, &uvbg, &ne);
        double LambdaCmptn = get_compton_cooling(dens, ienergy, 1 - HYDROGEN_MASSFRAC, zz, ne);
        double LambdaCollis = get_individual_cooling(COLLIS, dens, ienergy, 1 - HYDROGEN_MASSFRAC, &uvbg, &ne);
        double LambdaRecomb = get_individual_cooling(RECOMB, dens, ienergy, 1 - HYDROGEN_MASSFRAC, &uvbg, &ne);
        double LambdaFF = get_individual_cooling(FREEFREE, dens, ienergy, 1 - HYDROGEN_MASSFRAC, &uvbg, &ne);
        double Heat = get_individual_cooling(HEAT, dens, ienergy, 1 - HYDROGEN_MASSFRAC, &uvbg, &ne);
        double temp = get_temp(dens, ienergy, 1 - HYDROGEN_MASSFRAC, &uvbg, &ne);
        fprintf(fd, "%g %g %g %g %g %g %g %g %g\n",zz, LambdaNet, Heat, LambdaFF, LambdaCollis, LambdaRecomb, LambdaCmptn, temp, ne);
        double nH0 = get_neutral_fraction_phys_cgs(dens, ienergy, 1 - HYDROGEN_MASSFRAC, &uvbg, &ne);
        double He0 = get_helium_ion_phys_cgs(0, dens, ienergy, 1 - HYDROGEN_MASSFRAC, &uvbg, ne);
        double Hep = get_helium_ion_phys_cgs(1, dens, ienergy, 1 - HYDROGEN_MASSFRAC, &uvbg, ne);
        double Hepp = get_helium_ion_phys_cgs(2, dens, ienergy, 1 - HYDROGEN_MASSFRAC, &uvbg, ne);
        fprintf(fd2, "%g %g %g %g %g %g\n", zz, dens * HYDROGEN_MASSFRAC, nH0, He0, Hep, Hepp);
    }
    fclose(fd);
    fclose(fd2);
}
#endif
