shenqi
=========

Massively Parallel Cosmological SPH Simulation Software on a GPU.

An out of date source code browser may be found here:
`Source code browser <https://mp-gadget.github.io/MP-Gadget/classes.html>`_

Description
-----------

Shenqi is derived from MP-Gadget, ultimately derived from Gadget-2.
Shenqi's predecessor, MP-Gadget, is the source code used to run the BlueTides and ASTRID simulations (http://bluetides-project.org).
Shenqi requires boost, GSL and a C++ compiler with C++20 and OpenMP 5 support. At time of writing, no version of gcc fully supports OpenMP 5. Clang or nvcc are preferred.

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

GPU Porting
-----------

For the GPU (accelerator) support we use OpenMP 5.0 target.

A loop is enclosed within:

.. code:: C

    #pragma omp target teams distribute nowait

And the body of the loop is executed on the GPU. The CPU does not wait for the work to complete, but continues to the next loop. Current loops to be accelerated are ev_primary and ev_secondary. Ultimately we will also offload the tree build.

OpenMP also allows one to manually manage which memory is present on the GPU (note this is not necessary on TACC machines, which have unified memory). This is done with:

.. code:: C
    #pragma omp target enter data map(to: *this) nowait

which will map the memory pointed to by this (the current struct) to the GPU. The to map-type specifies that the data will only be read on the device, and thus performs essentially a read-only copy. The from map-type specifies that the data will only be written to on the device. Data which is mapped 'to' will be copied to the GPU but not copied back to the CPU after the kernel completes, whereas data mapped 'from' will not be copied to the GPU and will be copied back to the CPU at the end of the kernel. Copies are shallow. Pointers within the current struct must be moved separately.

nowait makes this happen asynchronously. One may think of OpenMP as really placing tasks (copies and kernels) within a queue for the GPU accelerator to execute. You can use `depend' clauses to express how each task depends on another. depend(in: ) means that the memory must be ready for reading, and depend(out: ) means ready for writing.
.. code:: C
    #pragma omp target enter data map(to: *this) nowait
    #pragma omp target enter data map(to: *this->array_data) nowait

Data which is already mapped to the GPU can be updated with new memory from the CPU using:

.. code:: C
    #pragma omp target update to(*this) nowait

Or GPU memory can be written back to the CPU with:

.. code:: C
    #pragma omp target update from(Accel[0:N]) nowait depend(in: Accel) depend(out: Accel)

Before memory is freed on the CPU (ie, in the destructor), you should call

.. code:: C
    #pragma omp target exit data map(delete:*this->array_data)
    #pragma omp target exit data map(delete:*this)

to remove the memory mappings. If you run target exit data without first running target enter data, the code may crash. To ensure that you do not delete data not present on the CPU, you should use:

.. code:: C
      if(omp_target_is_present(Accel, omp_get_default_device()))
          #pragma omp target exit data map(delete: Accel[0:N])

Memory which is solely for the use of the GPU, and never uses CPU memory at all, should be allocated with
.. code:: C
    omp_target_alloc(size_t size, omp_get_default_device());

and freed with omp_target_free(). This should be the default: any memory which is not transferred to other ranks should live on the GPU. Eventually the tree should be permanently GPU-resident.

Data which is created on the GPU and then moved to the CPU should be allocated on the CPU in the normal way and mapped to the GPU with a map(from:) clause:
.. code:: C
    #pragma omp target enter data map(from: *this) nowait
This allocates memory on the GPU shadowing the CPU memory, increments a reference count, but does not copy anything. The contents of the GPU memory is written back to the CPU only when the reference count reaches zero. The reference count is dereferenced at:
.. code:: C
    #pragma omp target exit data map(from: *this) depend(in: *this) nowait
which will cause an asynchronous memory writeback.

It is also possible to map memory when starting a kernel. If this is done, the memory will be written back to the CPU at the end of the kernel execution region. However, this will delay the start of the kernel until the memory copy to the GPU has completed, and should be avoided if possible. This can be done using:

.. code:: C
    #pragma omp target teams distribute nowait map(tofrom: data)

If the data is already mapped, no copy is performed, but no writeback is performed at the end of the region either. The data is not available until target exit data pragmas are run.

Note! If you forget to copy the memory to the device, but you still have a depend() clause, OpenMP will silently proceed as if the dependency were already satisfied. However, the memory will not necessarily be present on the GPU and the code will likely crash (unless it is a unified memory machine). There is no runtime check that catches a missing target enter data. The OpenMP standard leaves it as undefined behaviour. We therefore prefer the no-op map() clause on the kernel to a depends clause.

The map type on exit data controls writeback, not enter data:

.. code:: C
  #pragma omp target enter data map(to: Accel[0:N])     // ref count: 0→1
  #pragma omp target enter data map(to: Accel[0:N])     // ref count: 1→2

  #pragma omp target exit data map(release: Accel[0:N]) // ref count: 2→1, no writeback
  #pragma omp target exit data map(from: Accel[0:N])    // ref count: 1→0, writeback

The valid map types for target exit data are from, release, and delete. delete forces the count to zero immediately and skips writeback:

.. code:: C
  #pragma omp target exit data map(delete: Accel[0:N])  // ref count: forced to 0, deallocate, no writeback

Be extremely careful with these references counts: should they be mismatched there will be no error and very little debugging ability. target exit delete should be called in desctructors

Installation
------------

First time users:

.. code:: bash

    git clone https://github.com/MP-Gadget/shenqi.git
    cd shenqi
    make -j

The Makefile will automatically copy Options.mk.example to Options.mk. The default compiler is likely g++. To use clang you need:

.. code:: bash

    OMPI_CXX=clang++ OMPI_C=clang make

We will need boost and gsl. On HPC systems with the modules command:

.. code:: bash

    module load gsl boost

    env | grep GSL  # check if GSL path is reasonable
    env | grep BOOST  # check if BOOST path is reasonable

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
