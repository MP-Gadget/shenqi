/*Tests for the drift factor module.*/
#define BOOST_TEST_MODULE metal_return
#include "booststub.h"

#include <gsl/gsl_integration.h>
#include <boost/math/quadrature/gauss_kronrod.hpp>

#include "libgadget/utils/endrun.h"
#include "libgadget/metal_return.h"
#include "libgadget/slotsmanager.h"
#include "libgadget/metal_tables.h"

BOOST_AUTO_TEST_CASE(test_yields)
{
    gsl_integration_workspace * gsl_work = gsl_integration_workspace_alloc(GSL_WORKSPACE);
    set_metal_params(1.3e-3);

    struct interps interp;
    setup_metal_table_interp(&interp);
    /* Compute factor to normalise the total mass in the IMF to unity.*/
    double imf_norm = compute_imf_norm(gsl_work);
    BOOST_TEST(imf_norm == 0.624632, tt::tolerance(0.01));

    double agbyield = compute_agb_yield(interp.agb_mass_interp, agb_total_mass, 0.01, 1, 40, gsl_work);
    double agbyield2 = compute_agb_yield(interp.agb_mass_interp, agb_total_mass, 0.01, 1, SNAGBSWITCH, gsl_work);
    BOOST_TEST(agbyield == agbyield2, tt::tolerance(1e-3));
    /* Lifetime is about 200 Myr*/
    double agbyield3 = compute_agb_yield(interp.agb_mass_interp, agb_total_mass, 0.01, 5, 40, gsl_work);

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
    double sniiyield = compute_snii_yield(interp.snii_mass_interp, snii_total_mass, 0.01, 1, 40, gsl_work);

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
    find_mass_bin_limits(&masslow1, &masshigh1, 0, 30, 0.02, interp.lifetime_interp);
    find_mass_bin_limits(&masslow2, &masshigh2, 30, 60, 0.02, interp.lifetime_interp);
    find_mass_bin_limits(&masslowsum, &masshighsum, 0, 60, 0.02, interp.lifetime_interp);
    message(0, "0 - 30: %g %g 30 - 60 %g %g 0 - 60 %g %g\n", masslow1, masshigh1, masslow2, masshigh2, masslowsum, masshighsum);
    BOOST_TEST(masslow1 == masshigh2, tt::tolerance(0.01));
    BOOST_TEST(masslowsum == masslow2, tt::tolerance(0.01));
}
