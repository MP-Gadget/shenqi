
# Customization; see Options.mk.example
CONFIG ?= ../Options.mk

include $(CONFIG)

include ../Makefile.rules

CPP_SRCS = main.cpp params.cpp
OBJS := $(addsuffix .o,$(basename $(CPP_SRCS)))
OBJS := $(OBJS:%.o=.objs/%.o)

all: MP-GenIC

MP-GenIC: $(OBJS) ../libgenic/libgenic.a ../libgadget/libgadget.a ../libgadget/libgadget-utils.a
	$(MPICXX) $(OPTIMIZE) $^ $(LIBS) -o $@

DEPS := $(OBJS:.o=.d)
-include $(DEPS)

clean:
	rm -rf $(OBJS) $(OBJS:.o=.d) MP-GenIC
