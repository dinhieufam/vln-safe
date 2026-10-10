# The documented interface to the benchmark. Every
# target is a thin wrapper over one script or module so a reader never has to
# reconstruct a command line. Variables: AGENT, BENCH, REGIME, EPISODES, OUT.

AGENT    ?= etpnav
BENCH    ?= r2r
REGIME   ?= block
EPISODES ?=
OUT      ?=
SHARDS   ?= 2

.PHONY: setup smoke arm merge numbers tables figures test lint

setup:              ## build the conda envs, wire data/ symlinks, pip install -e .
	environment/setup.sh

smoke:              ## 5 episodes, one agent, ~10 minutes (needs habitat)
	python -m vlnsafe smoke --agent $(AGENT) --benchmark $(BENCH)

arm:                ## one arm: python -m vlnsafe run --agent .. --benchmark .. --regime ..
	python -m vlnsafe run --agent $(AGENT) --benchmark $(BENCH) --regime $(REGIME) \
		$(if $(EPISODES),--episodes $(EPISODES),) $(if $(OUT),--out $(OUT),)

merge:              ## concatenate shards, dedupe, gzip the step log
	python scripts/merge_shards.py results/$(AGENT)/$(BENCH)/$(REGIME)

numbers:            ## logs -> results/summary/*.json -> outputs/{tables,figures}
	python analysis/collect.py
	$(MAKE) tables figures

tables:
	python analysis/make_tables.py
	python analysis/prose_numbers.py

figures:
	python analysis/make_figures.py

test:              ## pure tests need no habitat; add -m slow for the rollout test
	python -m pytest -m "not slow"

lint:
	python -m pyflakes vlnsafe analysis scripts || true
