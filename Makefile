.PHONY: install features train evaluate test clean

install:
	python -m pip install -e .

features:
	python scripts/extract_features.py

train:
	python scripts/train.py

evaluate:
	python scripts/evaluate.py

test:
	pytest

clean:
	rm -rf -- build dist .pytest_cache src/coffee_leaf.egg-info
	find app notebooks scripts src tests -type d -name __pycache__ -prune -exec rm -rf -- {} +
