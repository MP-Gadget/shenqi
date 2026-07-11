# Standalone microbenchmarks. Needs the heffte library built in depends/.
CONFIG ?= ../Options.mk
include $(CONFIG)
include ../Makefile.rules

fftbench: fftbench.cpp
	$(MPICXX) $(OPTIMIZE) $(FFTW_INCL) -I../depends/include $< -o $@ \
	    -L../depends/lib -lheffte $(FFTW_LIBS)

clean:
	rm -f fftbench
