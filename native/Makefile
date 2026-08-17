CC ?= cc
BUILD_DIR ?= build
CFLAGS ?= -O3 -std=c11 -Wall -Wextra -Wpedantic
OMPFLAGS ?=

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
LIBRARY := $(BUILD_DIR)/libpvzwm_native.dylib
SHARED_FLAGS := -dynamiclib
else
LIBRARY := $(BUILD_DIR)/libpvzwm_native.so
SHARED_FLAGS := -shared
endif

.PHONY: all clean

all: $(LIBRARY)

$(LIBRARY): pvz_sim.c pvz_sim.h pvz_core.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $(OMPFLAGS) -fPIC $(SHARED_FLAGS) pvz_sim.c -o $@

clean:
	rm -f $(LIBRARY)
