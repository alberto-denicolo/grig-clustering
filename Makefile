CONFIG ?= configs/thesis.yaml

.PHONY: install test lint quick experiments figures all clean

install:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check .
	ruff format --check .

quick:
	grig run all --config configs/quick.yaml --out results/quick
	grig plot --results results/quick --figures figures/quick

experiments:
	grig run all --config $(CONFIG) --out results

figures:
	grig plot --results results --figures figures

all: experiments figures

clean:
	rm -rf results/quick figures/quick .pytest_cache .ruff_cache
