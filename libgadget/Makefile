# vim: set ft=make:

# Customization; see Options.mk.example

CONFIG ?= ../Options.mk

include $(CONFIG)

UTILS_TESTED = peano memory interp
UTILS_MPI_TESTED = mpsort

TESTED = cooling \
	cooling_rates \
	cosmology \
	hci \
	slotsmanager \
	powerspectrum \
	forcetree \
	timebinmgr \
	neutrinos_lra \
	omega_nu_single \
	density \
	gravity \
	metal_return

MPI_TESTED = fof exchange

TESTBIN :=$(UTILS_TESTED:%=.objs/utils/test_%) $(UTILS_MPI_TESTED:%=.objs/utils/test_%) $(TESTED:%=.objs/test_%) $(MPI_TESTED:%=.objs/test_%)
MPISUITE = $(MPI_TESTED:%=test_%) $(UTILS_MPI_TESTED:%=utils/test_%)
SUITE = $(TESTED:%=test_%) $(UTILS_TESTED:%=utils/test_%) $(MPISUITE)
include ../Makefile.rules

FILES = $(shell git ls-files)

# define the version
# (easier to extract from other utils, e.g. python)
include ../Makefile.version

# Main Rules
#
# make V=1 to see the command line
#

GADGET_C_SRCS =  hci.c\
	 partmanager.c \
	 blackhole.c bhinfo.c bhdynfric.c \
	 drift.c stats.c \
	 init.c checkpoint.c \
	 cooling.c \
	 density.c treewalk.c \
	 gravshort-tree.c gravshort-pair.c hydra.c \
	 gravpm.c powerspectrum.c \
	 forcetree.c gravity.c \
	 densitykernel.c \
	 config.c uvbg.c \
	 lenstools.c

GADGET_CPP_SRCS = hydra2.cpp gravshort-tree2.cpp walltime.cpp \
	density2.cpp veldisp2.cpp \
cooling_rates.cpp cooling_uvfluc.cpp cooling_qso_lightup.cpp cosmology.cpp neutrinos_lra.cpp omega_nu_single.cpp timebinmgr.cpp lightcone.cpp \
fof.cpp fofpetaio.cpp petaio.cpp petapm.cpp \
slotsmanager.cpp domain.cpp timestep.cpp  plane.cpp winds.cpp \
exchange.cpp sfr_eff.cpp runtests.cpp run.cpp metal_return.cpp

GADGET_CUDA_SRCS = gravshortcuda.cu densitycuda.cu hydracuda.cu

ifneq ($(findstring -DUSE_CUDA, $(OPT)),)
GADGET_OBJS := $(addsuffix .o,$(basename $(GADGET_C_SRCS))) $(addsuffix .o,$(basename $(GADGET_CPP_SRCS))) $(addsuffix .o,$(basename $(GADGET_CUDA_SRCS)))
else
GADGET_OBJS := $(addsuffix .o,$(basename $(GADGET_C_SRCS))) $(addsuffix .o,$(basename $(GADGET_CPP_SRCS)))
endif

GADGET_UTILS_OBJS= \
utils/endrun.o \
utils/peano.o \
utils/memory.o \
utils/mymalloc.o \
utils/system.o \
utils/paramset.o \
utils/unitsystem.o \
utils/string.o \
utils/spinlocks.o

GADGET_OBJS := $(GADGET_OBJS:%=.objs/%)
GADGET_UTILS_OBJS := $(GADGET_UTILS_OBJS:%=.objs/%)

all: libgadget.a libgadget-utils.a

.PHONY: all test run-tests

.objs/utils/test_%: tests/test_%.cpp .objs/utils/%.o libgadget-utils.a
	$(MPICXX) $(TCFLAGS) -I../tests/ $^ $(LIBS) -o $@

.objs/test_%: tests/test_%.cpp .objs/%.o libgadget-utils.a
	$(MPICXX) $(TCFLAGS) -I../tests/ $^ $(LIBS) -o $@

#No .obj, header-only
.objs/utils/test_mpsort: tests/test_mpsort.cpp libgadget-utils.a
	$(MPICXX) $(TCFLAGS) -I../tests/ $^ $(LIBS) -o $@

.objs/utils/test_interp: tests/test_interp.cpp libgadget-utils.a
	$(MPICXX) $(TCFLAGS) -I../tests/ $^ $(LIBS) -o $@

.objs/test_cosmology: tests/test_cosmology.cpp .objs/cosmology.o libgadget-utils.a
	$(MPICXX) $(TCFLAGS) -I../tests/ $^ $(LIBS) -o $@

objs/test_densitykernel: tests/test_densitykernel.cpp densitykernel.hpp
	$(MPICXX) $(TCFLAGS) -I../tests/ $^ $(LIBS) -o $@

.objs/test_timebinmgr: tests/test_timebinmgr.cpp .objs/timebinmgr.o .objs/cosmology.o .objs/omega_nu_single.o libgadget-utils.a
	$(MPICC) $(TCFLAGS) -I../tests/ $^ $(LIBS) -o $@

.objs/test_neutrinos_lra: tests/test_neutrinos_lra.cpp .objs/neutrinos_lra.o .objs/cosmology.o .objs/omega_nu_single.o libgadget-utils.a
	$(MPICXX) $(TCFLAGS) -I../tests/ $^ $(LIBS) -o $@

.objs/test_cooling_rates: tests/test_cooling_rates.cpp .objs/cooling_rates.o .objs/cooling_uvfluc.o libgadget-utils.a
	$(MPICXX) $(TCFLAGS) -I../tests/ $^ $(LIBS) -o $@

.objs/test_exchange: tests/test_exchange.cpp .objs/exchange.o libgadget.a libgadget-utils.a
	$(MPICXX) $(TCFLAGS) -I../tests/ $^ $(LIBS) -o $@

.objs/test_density: tests/test_density.cpp .objs/density.o libgadget.a libgadget-utils.a
	$(MPICXX) $(TCFLAGS) -I../tests/ $^ $(LIBS) -o $@

.objs/test_metal_return: tests/test_metal_return.cpp .objs/metal_return.o libgadget.a libgadget-utils.a
	$(MPICXX) $(TCFLAGS) -I../tests/ $^ $(LIBS) -o $@

.objs/test_cooling: tests/test_cooling.cpp .objs/cooling.o .objs/cooling_rates.o .objs/cooling_uvfluc.o libgadget-utils.a
	$(MPICXX) $(TCFLAGS) -I../tests/ $^ $(LIBS) -o $@

.objs/test_gravity: tests/test_gravity.cpp libgadget.a libgadget-utils.a
	$(MPICXX) $(TCFLAGS) -I../tests/ $^ $(LIBS) -o $@

.objs/test_fof: tests/test_fof.cpp libgadget.a libgadget-utils.a
	$(MPICXX) $(TCFLAGS) -I../tests/ $^ $(LIBS) -o $@

.objs/test_forcetree: tests/test_forcetree.cpp libgadget.a libgadget-utils.a
	$(MPICXX) $(TCFLAGS) -I../tests/ $^ $(LIBS) -o $@

build-tests: $(TESTBIN)

test : build-tests
	trap 'err=1' ERR; for tt in $(SUITE) ; do \
		echo Running $$tt; \
		if [[ "$(MPISUITE)" =~ .*$$tt.* ]]; then \
            OMPI_MCA_rmaps_base_oversubscribe=1 \
			 mpirun -np 4 .objs/$$tt ;  \
		else \
			.objs/$$tt ; \
		fi ;  \
	done; exit $$err

libgadget-utils.a: $(GADGET_UTILS_OBJS)
	$(AR) r $@ $(GADGET_UTILS_OBJS)

libgadget.a: $(GADGET_OBJS)
	$(AR) r $@ $(GADGET_OBJS)

clean:
	rm -rf .objs libgadget.a libgadget-utils.a config.c

config.c: $(CONFIG)
	mkdir -p `dirname $@`
	MPICC="$(MPICC)" CFLAGS="$(CFLAGS)" OPT="$(OPT)" OPTIMIZE="$(OPTIMIZE)" VERSION="$(VERSION)" bash makeconfig.sh $@

GADGET_DEPS := $(GADGET_OBJS:.o=.d)
GADGET_UTILS_DEPS := $(GADGET_UTILS_OBJS:.o=.d)
-include $(GADGET_DEPS)
-include $(GADGET_UTILS_DEPS)


#This snippet works out the current git revision and the git revision in config.h.
#It checks whether they are the same.
#If they are not the same it makes config.h as a PHONY target to be rebuilt.
GIT=$(subst -,_,$(shell git describe --always --dirty --abbrev=10))
EXISTGIT=$(shell grep "GADGET_VERSION" config.c)
ifeq (,$(findstring $(GIT),$(EXISTGIT)))
.PHONY: config.c
endif

MPI_SHOWME := $(shell $(MPICC) -showme)

.objs/%.compdb_entry: %.c
	@echo "    {" > $@
	@echo "        \"command\": \"$(MPI_SHOWME) $(CFLAGS) -c $<\","   >> $@
	@echo "        \"directory\": \"$(CURDIR)\","               >> $@
	@echo "        \"file\": \"$<\""                    >> $@
	@echo "    },"                              >> $@

.objs/%.compdb_entry: %.cpp
	@echo "    {" > $@
	@echo "        \"command\": \"$(MPI_SHOWME) $(CFLAGS) -c $<\","   >> $@
	@echo "        \"directory\": \"$(CURDIR)\","               >> $@
	@echo "        \"file\": \"$<\""                    >> $@
	@echo "    },"                              >> $@

COMPDB_ENTRIES = $(addsuffix .compdb_entry, $(basename $(GADGET_OBJS)))

compile_commands.json: $(COMPDB_ENTRIES)
	@echo "[" > $@.tmp
	@cat $^ >> $@.tmp
	@sed '$$d' < $@.tmp > $@
	@echo "    }" >> $@
	@echo "]" >> $@
	@rm $@.tmp
