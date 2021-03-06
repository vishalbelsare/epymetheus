test:
	@pytest --doctest-modules epymetheus
	@pytest --doctest-modules tests

format:
	@python3 -m black --quiet .
	@python3 -m isort --force-single-line-imports --quiet .

publish:
	@gh workflow run publish.yml

