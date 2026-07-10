CONFIG = ../Options.mk
include $(CONFIG)

.PHONY: depends
# MPICC ?= mpicc
MPICCDEP ?= mpicc
MPICXXDEP ?= mpic++
OPTIMIZE ?= -O2 -g -fopenmp -Wall
COPTIMIZE1 = $(filter-out -std=gnu++20,$(OPTIMIZE))
COPTIMIZE = $(filter-out -stdpar=gpu,$(COPTIMIZE1))
LIBRARIES=lib/libbigfile-mpi.a
HEFFTELIBRARY=lib/libheffte.a
depends: $(LIBRARIES) $(HEFFTELIBRARY)

lib/libbigfile-mpi.a: bigfile/src/bigfile-mpi.c
	mkdir -p lib; \
	mkdir -p include; \
	cd bigfile/src; \
	make install PREFIX=$(PWD) CC="$(MPICCDEP)" MPICC="$(MPICCDEP)" CFLAGS="$(COPTIMIZE)" AR="$(AR)"

# Enable the cufft backend in heffte when the code is built with CUDA.
ifneq ($(findstring -DUSE_CUDA, $(OPT)),)
    HEFFTE_CUDA = -DHeffte_ENABLE_CUDA=ON
endif

# heffte uses the system fftw3 (found via its cmake module: needs the
# double and single precision libraries, i.e. the fftw-devel package).
$(HEFFTELIBRARY): heffte/CMakeLists.txt
	cmake -S heffte -B heffte/build \
	    -DCMAKE_INSTALL_PREFIX=$(PWD) \
	    -DCMAKE_INSTALL_LIBDIR=lib \
	    -DCMAKE_BUILD_TYPE=Release \
	    -DBUILD_SHARED_LIBS=OFF \
	    -DCMAKE_CXX_COMPILER=$(MPICXXDEP) \
	    -DCMAKE_CXX_FLAGS=-fopenmp \
	    -DHeffte_ENABLE_FFTW=ON \
	    -DHeffte_ENABLE_TESTING=OFF $(HEFFTE_CUDA)
	cmake --build heffte/build -j 8
	cmake --install heffte/build

clean: clean-fast clean-fft

clean-fast:
	rm -rf $(LIBRARIES)
	cd bigfile/src; make clean

clean-fft:
	rm -rf $(HEFFTELIBRARY)
	rm -rf heffte/build
