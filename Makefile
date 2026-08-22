# Interactive correction for manipulation
#
# Quick start:  make install && make assets && make test
# Full study:   make study
# GPU pipeline: make demos && make train && make eval

PY      ?= python3
VENV    ?= .venv
BIN     := $(VENV)/bin
RUNS    ?= runs
EPISODES?= 200
DEVICE  ?= auto

# Software GL so everything works on a headless box with no GPU. Override with
# MUJOCO_GL=egl on a machine with a GPU: it is roughly 50x faster to render.
export MUJOCO_GL ?= osmesa

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

$(BIN)/python:
	$(PY) -m venv $(VENV)
	$(BIN)/python -m pip install --upgrade pip wheel

.PHONY: install
install: $(BIN)/python ## Create a venv and install the project (CPU torch)
	$(BIN)/python -m pip install -e ./interventionkit
	$(BIN)/python -m pip install -e ".[dev,viz]"
	@echo
	@echo "Installed. PyTorch is NOT installed by default - it is ~2.5 GB."
	@echo "  CPU only : make torch-cpu"
	@echo "  NVIDIA   : make torch-cuda"

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
