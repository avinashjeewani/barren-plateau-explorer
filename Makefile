.PHONY: install test quick reproduce

install:
	pip install -e ".[dev]"

test:
	pytest -q

quick:
	python -m bpx.experiments --config configs/quick.yaml

reproduce:
	python -m bpx.experiments --config configs/default.yaml
