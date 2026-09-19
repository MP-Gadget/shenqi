
# Customization; see Options.mk.example
CONFIG ?= ../Options.mk

include $(CONFIG)

TESTED = power thermal
TESTBIN := $(TESTED:%=.objs/test_%) $(MPI_TESTED:%=.objs/test_%)
MPISUITE = $(MPI_TESTED:%=test_%)
SUITE = $(TESTED:%=test_%)

include ../Makefile.rules

GENIC_CPP_SRCS = power.cpp thermal.cpp zeldovich.cpp glass.cpp save.cpp

OBJS := $(addsuffix .o,$(basename $(GENIC_CPP_SRCS)))

OBJS := $(OBJS:%.o=.objs/%.o)

.PHONY: all test run-tests

all : libgenic.a

.objs/test_%: tests/test_%.cpp .objs/%.o ../libgadget/libgadget.a ../libgadget/libgadget-utils.a
	$(MPICC) $(TCFLAGS) -I../tests/ $^ $(LIBS) -o $@

build-tests: $(TESTBIN)

test : build-tests
	trap 'err=1' ERR; \
	for tt in $(SUITE) ; do \
		echo Running $$tt; \
		.objs/$$tt ; \
	done; \
	for tt in $(MPISUITE) ; do \
		echo Running $$tt "(4 ranks)"; \
		OMP_NUM_THREADS=2 OMPI_MCA_rmaps_base_oversubscribe=1 mpirun -np 4 --bind-to none .objs/$$tt ; \
	done; \
	exit $$err

libgenic.a: $(OBJS)
	$(AR) r $@ $(OBJS)

DEPS := $(OBJS:.o=.d)
-include $(DEPS)

clean:
	rm -rf .objs libgenic.a

