# Traffic Safety Clip Logger — developer convenience targets.
# Usage: `make <target>`. Run from Git Bash, WSL2, or a Linux shell.

.PHONY: help setup-dev setup-analyze setup-mini-pc test run-test lint \
        docker-cpu docker-gpu compose-up-capture compose-up-analyze clean

PYTHON ?= python
SAMPLE ?= samples/street-test.mp4
CONFIG_DEV ?= config/config.dev.yaml

help:
	@echo "Targets:"
	@echo "  setup-dev        Install package + dev tooling (editable)"
	@echo "  setup-analyze    Install package + CV analysis extras"
	@echo "  setup-mini-pc    Install package for the mini-PC appliance"
	@echo "  test             Run the pytest suite"
	@echo "  run-test         Run the offline stub on \$$SAMPLE"
	@echo "  docker-cpu       Build the CPU Docker image"
	@echo "  docker-gpu       Build the GPU Docker image"
	@echo "  clean            Remove caches and build artifacts"

setup-dev:
	$(PYTHON) -m pip install -e .[dev]

setup-analyze:
	$(PYTHON) -m pip install -e .[analyze,dev]

setup-mini-pc:
	$(PYTHON) -m pip install .

test:
	$(PYTHON) -m pytest -q

run-test:
	traffic-log test --source $(SAMPLE) --config $(CONFIG_DEV)

docker-cpu:
	docker compose build capture analyze

docker-gpu:
	docker compose build analyze-gpu

compose-up-capture:
	docker compose up -d capture

compose-up-analyze:
	docker compose up -d analyze

clean:
	rm -rf build dist *.egg-info src/*.egg-info .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
