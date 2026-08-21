# Model-check the TLA+ specification of M-of-N attestation and settlement.
#
#   make               check every configuration
#   make MC_N3M2       check one configuration
#   make drift         check the modelled code has not changed underneath us
#   make accept-drift  re-record those hashes, after re-reading the model
#   make jar           fetch tla2tools.jar without checking anything
#   make clean         remove the jar and the TLC output
#
# check.sh does the work and is the thing CI runs; this file exists so that
# `make -C spec` does the obvious thing.

.PHONY: all check drift accept-drift jar clean

# Drift first: if the code moved, the model may be checking something it no
# longer describes, and forty-five seconds of TLC would say so in the most
# reassuring possible way.
all: drift check

check:
	./check.sh

drift:
	python3 drift.py --check

accept-drift:
	python3 drift.py --update

jar:
	./check.sh --jar-only

clean:
	rm -rf out tla2tools.jar

# Any other goal is taken as the name of a configuration to check, so
# `make MC_Replay_N3M2` works.
MC_%:
	./check.sh $@
