#!/bin/sh

python3 -m pytest --doctest-modules epymetheus
python3 -m pytest --doctest-modules tests

python3 -m flake8 epymetheus
python3 -m black --check --quiet epymetheus || read -p "Run black? (y/n): " yn; [[ $yn = [yY] ]] && python3 -m black --quiet epymetheus
python3 -m isort --check --force-single-line-imports epymetheus || read -p "Run isort? (y/n): " yn; [[ $yn = [yY] ]] && python3 -m isort --force-single-line-imports --quiet epymetheus
python3 -m black --quiet tests
python3 -m isort --force-single-line-imports --quiet tests
