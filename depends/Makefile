CONFIG = ../Options.mk
include $(CONFIG)

.PHONY: depends
.INTERMEDIATE: pfft
# MPICC ?= mpicc
MPICCDEP ?= mpicc
MPICXXDEP ?= mpic++
OPTIMIZE ?= -O2 -g -fopenmp -Wall
COPTIMIZE1 = $(filter-out -std=gnu++20,$(OPTIMIZE))
COPTIMIZE = $(filter-out -stdpar=gpu,$(COPTIMIZE1))
LIBRARIES=lib/libbigfile-mpi.a
FFTLIBRARIES=lib/libfftw3.a lib/libfftw3f.a lib/libfftw3_omp.a
HEFFTELIBRARY=lib/libheffte.a
depends: $(LIBRARIES) $(FFTLIBRARIES) $(HEFFTELIBRARY)
$(FFTLIBRARIES): pfft

lib/libbigfile-mpi.a: bigfile/src/bigfile-mpi.c
	mkdir -p lib; \
	mkdir -p include; \
	cd bigfile/src; \
	make install PREFIX=$(PWD) CC="$(MPICCDEP)" MPICC="$(MPICCDEP)" CFLAGS="$(COPTIMIZE)" AR="$(AR)"

# The pfft tarball bundles fftw3: we build it (in double and single
# precision) to get the fftw libraries used by the heffte FFTW backend.
pfft: install_pfft.sh
	mkdir -p lib; \
	mkdir -p include; \
	#Using -ipo causes icc to crash.
	MPICC="$(MPICCDEP)" CC="$(MPICCDEP)" CFLAGS="$(filter-out -stdpar=gpu,$(COPTIMIZE)) -I $(PWD)/include" LFLAGS="-L$(PWD)/lib" AR="$(AR)" RANLIB=$(RANLIB) \
        sh $(PWD)/install_pfft.sh $(PWD)/

$(HEFFTELIBRARY): $(FFTLIBRARIES) heffte/CMakeLists.txt
	cmake -S heffte -B heffte/build \
	    -DCMAKE_INSTALL_PREFIX=$(PWD) \
	    -DCMAKE_INSTALL_LIBDIR=lib \
	    -DCMAKE_BUILD_TYPE=Release \
	    -DBUILD_SHARED_LIBS=OFF \
	    -DCMAKE_CXX_COMPILER=$(MPICXXDEP) \
	    -DHeffte_ENABLE_FFTW=ON \
	    -DHeffte_ENABLE_TESTING=OFF \
	    -DFFTW_LIBRARIES="$(PWD)/lib/libfftw3_omp.a;$(PWD)/lib/libfftw3.a;$(PWD)/lib/libfftw3f_omp.a;$(PWD)/lib/libfftw3f.a" \
	    -DFFTW_INCLUDES=$(PWD)/include
	cmake --build heffte/build -j 8
	cmake --install heffte/build

clean: clean-fast clean-fft

clean-fast:
	rm -rf $(LIBRARIES)
	cd bigfile/src; make clean

clean-fft:
	rm -rf $(FFTLIBRARIES) $(HEFFTELIBRARY)
	rm -rf tmp-pfft-*/double
	rm -rf tmp-pfft-*/single
	rm -rf heffte/build
