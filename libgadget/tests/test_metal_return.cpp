/*Tests for the drift factor module.*/
#define BOOST_TEST_MODULE metal_return
#include "booststub.h"

#include <boost/math/quadrature/gauss_kronrod.hpp>

#include "libgadget/utils/endrun.h"
#include "libgadget/metal_return.h"
#include "libgadget/slotsmanager.h"
#include "libgadget/metal_tables.h"
#include "libgadget/cosmology.h"
#include "libgadget/physconst.h"

/* Set up a flat matter + Lambda cosmology, with radiation off and massless
 * neutrinos, so that the age has a simple closed form.*/
static void setup_cosmology(Cosmology * CP, double Omega0, double HubbleParam, const struct UnitSystem units)
{
    CP->CMBTemperature = 2.7255;
    CP->Omega0 = Omega0;
    CP->OmegaLambda = 1 - Omega0;
    CP->OmegaBaryon = 0.0483;
    CP->HubbleParam = HubbleParam;
    CP->RadiationOn = 0;
    CP->Omega_fld = 0;
    CP->w0_fld = -1;
    CP->wa_fld = 0;
    CP->Omega_ur = 0;
    CP->MNu[0] = CP->MNu[1] = CP->MNu[2] = 0;
    CP->HybridNeutrinosOn = 0;
    CP->use_class_radiation_convention = 0;
    init_cosmology(CP, 0.01, units);
}

/* Age of a flat matter + Lambda universe at scale factor a, in Myr:
 * t = 2/(3 sqrt(OL) H0) asinh(sqrt(OL/Om) a^(3/2)), with H0 = h * HUBBLE in 1/s.
 * Computed here without reference to the internal unit system.*/
static double lcdm_age_myr(double Omega0, double HubbleParam, double a)
{
    const double OmegaLambda = 1 - Omega0;
    const double hubtime_myr = 1 / (HubbleParam * HUBBLE * SEC_PER_MEGAYEAR);
    return 2. / (3 * sqrt(OmegaLambda)) * asinh(sqrt(OmegaLambda / Omega0) * pow(a, 1.5)) * hubtime_myr;
}

/* Check that the conversion from scale factor to Myr, in particular the
 * internal time unit of UnitTime_in_s / h, is right.*/
BOOST_AUTO_TEST_CASE(test_atime_to_myr)
{
    const double Omega0 = 0.2814;
    const double HubbleParam = 0.697;
    struct UnitSystem units = get_unitsystem(3.085678e21, 1.989e43, 1e5);
    Cosmology CP = {};
    setup_cosmology(&CP, Omega0, HubbleParam, units);

    /* The neutrino radiation density, which we cannot easily switch off, is a
     * 1e-4 relative correction to the expansion rate at z = 0 and less earlier.*/
    const double tol = 1e-3;

    /* Age of the Universe: about 13.7 Gyr.*/
    double age = atime_to_myr(&CP, 1e-3, 1);
    BOOST_TEST(age == lcdm_age_myr(Omega0, HubbleParam, 1) - lcdm_age_myr(Omega0, HubbleParam, 1e-3), tt::tolerance(tol));
    BOOST_TEST(age > 13000);
    BOOST_TEST(age < 14500);

    /* A few intervals spanning matter and Lambda domination.*/
    const double atimes[] = {0.05, 0.1, 0.25, 0.5, 0.75, 1};
    int i;
    for(i = 0; i < 5; i++) {
        double dt = atime_to_myr(&CP, atimes[i], atimes[i+1]);
        double expected = lcdm_age_myr(Omega0, HubbleParam, atimes[i+1]) - lcdm_age_myr(Omega0, HubbleParam, atimes[i]);
        message(0, "a %g -> %g: %g Myr (expected %g)\n", atimes[i], atimes[i+1], dt, expected);
        BOOST_TEST(dt == expected, tt::tolerance(tol));
    }

    /* Intervals add up.*/
    BOOST_TEST(atime_to_myr(&CP, 0.1, 0.5) + atime_to_myr(&CP, 0.5, 1) == atime_to_myr(&CP, 0.1, 1), tt::tolerance(1e-6));

    /* The age in Myr must not depend on the internal unit system.*/
    struct UnitSystem units2 = get_unitsystem(3.085678e24, 1.989e33, 3e7);
    Cosmology CP2 = {};
    setup_cosmology(&CP2, Omega0, HubbleParam, units2);
    BOOST_TEST(atime_to_myr(&CP2, 1e-3, 1) == age, tt::tolerance(1e-6));

    /* Halving h doubles the age: this is the h factor in the time unit.*/
    Cosmology CPh = {};
    setup_cosmology(&CPh, Omega0, HubbleParam / 2, units);
    BOOST_TEST(atime_to_myr(&CPh, 1e-3, 1) == 2 * age, tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(test_yields)
{
    set_metal_params(1.3e-3);

    struct interps interp;
    setup_metal_table_interp(&interp);
    /* Compute factor to normalise the total mass in the IMF to unity.*/
    double imf_norm = compute_imf_norm();
    BOOST_TEST(imf_norm == 0.936976167457, tt::tolerance(0.01));

    double agbyield = compute_agb_yield(&interp.agb_mass_interp, 0.01, 1, 40);
    double agbyield2 = compute_agb_yield(&interp.agb_mass_interp, 0.01, 1, SNAGBSWITCH);
    BOOST_TEST(agbyield == agbyield2, tt::tolerance(1e-3));
    /* Lifetime is about 200 Myr*/
    double agbyield3 = compute_agb_yield(&interp.agb_mass_interp, 0.01, 5, 40);

    /* Integrate the region of the IMF which contains SNII and AGB stars. The yields should never be larger than this
     * The Chabrier IMF used for computing SnII and AGB yields.
     * See 1305.2913 eq 3*/
    auto chabrier_mass = [](const double mass) {
        double imf;
        if(mass <= 1)
            imf = 0.852464 / mass * exp(- pow(log(mass / 0.079)/ 0.69, 2)/2);
        else
            imf = 0.237912 * pow(mass, -2.3);
        return mass * imf;
    };

    // Gauss-Kronrod integration for smooth functions. Boost uses by default the machine precision for accuracy and a max depth of 15.
    const double agbmax = boost::math::quadrature::gauss_kronrod<double, 61>::integrate(chabrier_mass, agb_total_mass[0], SNAGBSWITCH);
    const double sniimax = boost::math::quadrature::gauss_kronrod<double, 61>::integrate(chabrier_mass, SNAGBSWITCH, snii_masses[SNII_NMASS-1]);
    double sniiyield = compute_snii_yield(&interp.snii_mass_interp, 0.01, 1, 40);

    double sn1a = sn1a_number(0, 1500, 0.679)*sn1a_total_metals;
    BOOST_TEST(sn1a < 1.3e-3);

    message(0, "agbyield %g max %g (in 200 Myr: %g)\n", agbyield, agbmax, agbyield3);
    message(0, "sniiyield %g max %g sn1a %g\n", sniiyield, sniimax, sn1a);
    message(0, "Total fraction of mass returned %g\n", (sniiyield + sn1a + agbyield)/imf_norm);
    BOOST_TEST(agbyield < agbmax);
    BOOST_TEST(sniiyield < sniimax);
    BOOST_TEST((sniiyield + sn1a + agbyield)/imf_norm < 1.);

    double masslow1, masshigh1;
    double masslow2, masshigh2;
    double masslowsum, masshighsum;
    find_mass_bin_limits(&masslow1, &masshigh1, 0, 30, 0.02, &interp.lifetime_interp);
    find_mass_bin_limits(&masslow2, &masshigh2, 30, 60, 0.02, &interp.lifetime_interp);
    find_mass_bin_limits(&masslowsum, &masshighsum, 0, 60, 0.02, &interp.lifetime_interp);
    message(0, "0 - 30: %g %g 30 - 60 %g %g 0 - 60 %g %g\n", masslow1, masshigh1, masslow2, masshigh2, masslowsum, masshighsum);
    BOOST_TEST(masslow1 == masshigh2, tt::tolerance(0.01));
    BOOST_TEST(masslowsum == masslow2, tt::tolerance(0.01));
}
