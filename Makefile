# Interactive correction for manipulation
#
# Quick start:  make install && make assets && make test
# Full study:   make study
# GPU pipeline: make demos && make train && make eval

# MuJoCo publishes wheels for 3.10-3.13. Picking an interpreter in that range
# automatically avoids a source build that fails with an opaque error - the most
# common way a first install goes wrong.
PY      ?= $(shell for v in python3.12 python3.11 python3.13 python3.10 python3; do \
             command -v $$v >/dev/null 2>&1 && \
             $$v -c 'import sys; raise SystemExit(0 if (3,10) <= sys.version_info < (3,14) else 1)' \
             2>/dev/null && echo $$v && break; done)
VENV    ?= .venv
BIN     := $(VENV)/bin
RUNS    ?= runs
EPISODES?= 200
DEVICE  ?= auto

# The rendering backend is chosen at runtime by icm/__init__.py: EGL when a GPU
# is present, OSMesa on a headless Linux box, and the platform default on macOS
# and Windows. Set MUJOCO_GL here only to override that.

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

$(BIN)/python:
	@if [ -z "$(PY)" ]; then \
	  echo "No supported Python found (need 3.10-3.13; MuJoCo has no 3.14 wheel yet)."; \
	  echo "  macOS : brew install python@3.12  then  make install PY=python3.12"; \
	  echo "  Linux : sudo apt install python3.12-venv  then  make install PY=python3.12"; \
	  exit 1; \
	fi
	@echo "Using $(PY) ($$($(PY) --version))"
	$(PY) -m venv $(VENV)
	$(BIN)/python -m pip install --upgrade pip wheel

.PHONY: install
install: $(BIN)/python ## Minimal install: environment, study and tests
	$(BIN)/python -m pip install -e ./interventionkit
	$(BIN)/python -m pip install -e ".[dev]"
	@echo
	@echo "Installed. This is everything needed for 'make test' and 'make study'."
	@echo "Optional, only when you need them:"
	@echo "  make viz        figures and GIFs   (matplotlib, imageio)"
	@echo "  make torch-cpu  policy training    (CPU, small download)"
	@echo "  make torch-cuda policy training    (NVIDIA, ~2.5 GB - use wifi)"

.PHONY: viz
viz: ## Install plotting extras (only needed for figures and GIFs)
	$(BIN)/python -m pip install -e ".[viz]"

.PHONY: torch-cpu torch-cuda
torch-cpu: ## Install CPU-only PyTorch (small download)
	$(BIN)/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
torch-cuda: ## Install CUDA PyTorch (~2.5 GB; do this on wifi)
	$(BIN)/python -m pip install torch torchvision

.PHONY: assets
assets: ## Fetch the Franka Panda meshes at the pinned commit (~33 MB)
	$(BIN)/python -m icm.envs.assets.fetch

.PHONY: test
test: ## Run the full test suite
	$(BIN)/python -m pytest -q

.PHONY: lint
lint: ## Check formatting and lint
	$(BIN)/python -m ruff check src interventionkit tests
	$(BIN)/python -m ruff format --check src interventionkit tests

.PHONY: expert
expert: ## Measure the scripted expert baseline
	$(BIN)/python -m icm.cli.evaluate -n 100

.PHONY: study
study: ## Run the error-attribution study and write an HTML report
	$(BIN)/python -m icm.cli.study -o $(RUNS)/study -n 40 --controls 40 --report

.PHONY: sweep
sweep: ## Sweep supervisor tracing accuracy
	$(BIN)/python -m icm.cli.study -o $(RUNS)/sweep -n 25 --controls 10 --sweep 0.0 0.25 0.5 0.75 1.0

.PHONY: degradation
degradation: ## Measure what misattribution costs the trained policy
	$(BIN)/python -m icm.cli.dagger -o $(RUNS)/degradation --collect 150 --eval 150 --device $(DEVICE)

.PHONY: degradation-controlled
degradation-controlled: ## Same, with a shared demo pool so coverage is held fixed
	$(BIN)/python -m icm.cli.dagger -o $(RUNS)/degradation_controlled \
		--collect 200 --eval 250 --shared-demos 150 --device $(DEVICE)

.PHONY: demos
demos: ## Collect scripted demonstrations (add IMAGES=1 for camera data)
	$(BIN)/python -m icm.cli.collect -o $(RUNS)/demos -n $(EPISODES) $(if $(IMAGES),--images --depth,)

.PHONY: train
train: ## Train a behaviour-cloning policy on the demonstrations
	$(BIN)/python -m icm.cli.train $(RUNS)/demos -o $(RUNS)/bc --device $(DEVICE)

.PHONY: eval
eval: ## Evaluate the trained policy and write a GIF
	$(BIN)/python -m icm.cli.evaluate --checkpoint $(RUNS)/bc/checkpoint.pt -n 100 \
	  --state-key privileged --gif docs/media/rollout.gif

.PHONY: teleop
teleop: ## Drive the robot yourself (keyboard; needs pygame and a display)
	$(BIN)/python -m icm.cli.teleop -o $(RUNS)/teleop --device keyboard -n 5 --agent faulty

.PHONY: clean
clean: ## Remove run outputs and caches
	rm -rf $(RUNS) .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
