shenqi
=========

Massively Parallel Cosmological SPH Simulation Software on a GPU

An out of date source code browser may be found here:
`Source code browser <https://mp-gadget.github.io/MP-Gadget/classes.html>`_

Description
-----------

Shenqi is ultimately derived from P-Gadget / Gadget-2, with the gravity solver algorithm from Gadget-4, 
although there has been significant changes since then. Shenqi's predecessor, MP-Gadget, is the source code 
used to run the BlueTides and ASTRID simulations (http://bluetides-project.org).
Shenqi makes heavy use of boost and also requires GSL and a C++ compiler with C++20 and OpenMP 4.5 support.

The random number generator used in ShenqiIC is NOT THE SAME as the random number generator used in N-GenIC.
As a result the same random seed will produce different structure than N-GenIC. If this is important, use 
MP-GenIC to generate ICs, as the IC formats are compatible.

The infrastructure is heavily reworked. As a summary:

- A better PM solver for long range force with Pencil FFT.
- A better Tree solver with faster threading and less redundant code.
- Hierarchical gravity timestepping following Gadget-4.
- A better Domain decomposition that scales to half a million cores.
- A easier to use IO module with a Python binding.
- A more intuitive parameter file parser with schema and docstrings.
- A cleaner code base with less conditional compilation flags.

Physics models:

- Pressure Entropy SPH and Density Entropy SPH
- Radiation background in the expansion
- Massive neutrinos
- Dark energy
- ICs have species dependent density and velocity transfer functions
- Generic halo tracer particle seeding
- Various wind feedback and blackhole feedback models
- Various star formation criteria
- Primordial and metal cooling using updated recombination rates from the Sherwood simulation.
- Helium reionization
- Fluctuating UV background

Deprecated Features
-------------------

These features may be removed from shenqi if they get in the way (they remain in MP-Gadget if we need them):
- QuickLyaStarFormation
- SH03 winds (the ones that do not depend on the local velocity dispersion)
- Old-school SPH (ie, DensityIndependentSphOn = 0)
- Black hole repositioning
- The custom memory management stuff
- lightcone.c
- lenstools (depends on fftw3)
- The Gadget-3-style non-hierarchical gravitational timestepping (HierGravOn = 0)
- HeliumHeatOn model
- H2 star formation, get_sfr_factor_due_to_h2

Already removed:
- EXCUR_REION

Installation
------------

First time users:

.. code:: bash

    git clone https://github.com/MP-Gadget/shenqi.git
    cd shenqi
    make -j

The Makefile will automatically copy Options.mk.example to Options.mk. The default compile flags are appropriate for a linux using gcc, but may not be optimal.
We will need gsl. On HPC systems with the modules command,
usually it can be loaded with

.. code:: bash

    module load gsl

    env | grep GSL  # check if GSL path is reasonable

On a common PC/Linux system, refer to your package vendor how to
install gsl and gsl-devel.

If you wish to perform compile-time customisation (to, eg, change optimizations or use different compilers), you need an Options.mk file. The initial defaults are stored in Options.mk.example.

For other systems you should use the customised Options.mk file in the
platform-options directory. For example, for Stampede 2 you should do:

.. code:: bash

    cp platform-options/Options.mk.stampede2 Options.mk

For generic intel compiler based clusters, start with platform-options/Options.mk.icc

Compile-time options may be set in Options.mk. The remaining compile time options are generally only useful for development or debugging. All science options are set using a parameter file at runtime.

- DEBUG which enables various internal code consistency checks for debugging.
- NO_OPENMP_SPINLOCK uses the OpenMP default locking routines. These are often much slower than the default pthread spinlocks. However, they are necessary for Mac, which does not provide pthreads.
- USE_CFITSIO enables the output of lenstools compatible potential planes using cfitsio,

If compilation fails with errors related to the GSL, you may also need to set the GSL_INC or GSL_LIB variables in Options.mk to the filesystem path containing the GSL headers and libraries.

To run a N-Body sim, use IC files with no gas particles.

Now we are ready to build

.. code:: bash

    make -j

In the end, we will have 2 binaries:

.. code::

    ls gadget/shenqi genic/MP-GenIC

1. MP-Gadget is the main simulation program.

2. MP-GenIC is the initial condition generator.

Config Files
------------

Most options are configured at runtime with options in the config files.
The meaning of these options are documented in the params.c files in
the gadget/ and genic/ subdirectories.

Usage
-----

Find examples in examples/.

- dm-only : Dark Matter only
- lya : Lyman Alpha only
- hydro : hydro
- small : hydro with low resolution

Control number of threads with `OMP_NUM_THREADS`. Generally the code is faster with more threads per rank, up to hardware limits. On Frontera we run optimally with 28 threads, the number of cpus per hardware socket.

User Guide
----------

A longer user guide in LaTeX can be found here:
https://www.overleaf.com/read/kzksrgnzhtnh

IO Format
---------

The snapshot is in bigfile format. For data analysis in Python, use

.. code:: bash

   pip install bigfile

Refer to https://github.com/rainwoodman/bigfile for usage.

Otherwise directly open the blocks with Fortran or C, noting the data-type
information and attributes in header and attrs files (in plain text)


GLIBC 2.22
----------

Cray updated their GLIBC to 2.22+ recently.
A good move but it happens to be a buggy version of GLIBC:
https://sourceware.org/bugzilla/show_bug.cgi?id=19590
causing non-existing symbols like `_ZGVcN4v___log_finite`.
Adding `-lmvec -lmvec_nonshared` to GSL_LIBS works around the issue.

Bigfile
-------

Bigfile is incorporated using git-subtree, in the depends/bigfile prefix.
The command to update it (squash is currently mandatory) is:

.. code:: bash

    git subtree pull --prefix depends/bigfile "https://github.com/MP-Gadget/bigfile.git" master --squash

Contributors
------------

Gadget-2 was authored by Volker Springel.
The original P-GADGET3 was maintained by Volker Springel

MP-Gadget is maintained by Simeon Bird, Yu Feng and Yueying Ni.

Contributors to MP-Gadget include:

Yihao Zhou, Yanhui Yang. Nicholas Battaglia, Nianyi Chen, James Davies, Nishikanta Khandai, Karime Maamari, Chris Pederson, Phoebe Upton Sanderbeck, and Lauren Anderson.

Code review
-----------

Pull requests should ideally be reviewed. Here are some links on how to conduct review:

https://smartbear.com/learn/code-review/best-practices-for-peer-code-review/
http://web.mit.edu/6.005/www/fa15/classes/04-code-review/

Citation
--------

MP-Gadget was described most recently in https://arxiv.org/abs/2111.01160 and https://arxiv.org/abs/2110.14154 with various submodules having their own papers.

For usage of the code, here is a DOI for this repository that you can cite

.. image:: https://zenodo.org/badge/24486904.svg
   :target: https://zenodo.org/badge/latestdoi/24486904

Licence
-------

MP-Gadget is distributed under the terms of a 3-clause BSD license or the GNU General Public License v2 or later, at the option of the user. The use of PFFT and GSL libraries usually forces distribution under the terms of the GNU General Public License v3.

Status
------

master branch status:

.. image:: https://github.com/MP-Gadget/MP-Gadget/workflows/main/badge.svg
       :target: https://github.com/MP-Gadget/MP-Gadget/actions?query=workflow%3Amain
