# Model-check the TLA+ specification of M-of-N attestation and settlement.
#
#   make               check every configuration
#   make MC_N3M2       check one configuration
#   make jar           fetch tla2tools.jar without checking anything
#   make clean         remove the jar and the TLC output
#
# check.sh does the work and is the thing CI runs; this file exists so that
# `make -C spec` does the obvious thing.

.PHONY: all check jar clean

all: check

check:
	./check.sh

jar:
	./check.sh --jar-only

clean:
	rm -rf out tla2tools.jar

# Any other goal is taken as the name of a configuration to check, so
# `make MC_Replay_N3M2` works.
MC_%:
	./check.sh $@
