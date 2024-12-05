.PHONY: *

VENV=venv
PYTHON=$(VENV)/bin/python3
DEVICE=gpu
DATASET_FOLDER := dataset


# ================== LOCAL WORKSPACE SETUP ==================
venv:
	python3 -m venv $(VENV)
	@echo 'Path to Python executable $(shell pwd)/$(PYTHON)'

pre_commit_install:
	@echo "=== Installing pre-commit ==="
	$(PYTHON) -m pre_commit install

install_all: venv
	@echo "=== Installing common dependencies ==="
	$(PYTHON) -m pip install -r requirements/requirements-$(DEVICE).txt

	make pre_commit_install

# ================== CONTINUOUS INTEGRATION =================

ci_static_code_analysis:
	$(PYTHON) -m pre_commit run --all-files