/*Tests for the cosmology module, ported from N-GenIC.*/
#define BOOST_TEST_MODULE densitykernel

#include <boost/test/included/unit_test.hpp>
namespace tt = boost::test_tools;

#include <libgadget/densitykernel.hpp>

BOOST_AUTO_TEST_CASE(test_cubic_kernel)
{
    const double hh = 2.;
    CubicDensityKernel kern(hh);
    BOOST_TEST(33.510321638291124 == kern.desnumngb(1.0), tt::tolerance(1e-9));
    BOOST_TEST(4.0 / 3. * M_PI * pow(hh, 3) == kern.volume());
    BOOST_TEST(0.079577471545947673 == kern.wk(0.5));
    BOOST_TEST(-0.238732414637843 == kern.dwk(0.5));
}

BOOST_AUTO_TEST_CASE(test_quartic_kernel)
{
    const double hh = 2.;
    QuarticDensityKernel kern(hh);
    BOOST_TEST(65.449846949787357 == kern.desnumngb(1.0), tt::tolerance(1e-9));
    BOOST_TEST(4.0 / 3. * M_PI * pow(hh, 3) == kern.volume());
    BOOST_TEST(0.075283862851696096 == kern.wk(0.5));
    BOOST_TEST(-0.29142140458721072 == kern.dwk(0.5));
}

BOOST_AUTO_TEST_CASE(test_quintic_kernel)
{
    const double hh = 2.;
    QuinticDensityKernel kern(hh);
    BOOST_TEST(113.09733552923254 == kern.desnumngb(1.0), tt::tolerance(1e-9));
    BOOST_TEST(4.0 / 3. * M_PI * pow(hh, 3) == kern.volume());
    BOOST_TEST(0.066304197971682174 == kern.wk(0.5));
    BOOST_TEST(-0.3147351169541876 == kern.dwk(0.5));
}
