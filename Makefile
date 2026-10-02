PACKAGE := src/genplanner
TESTS := tests
FORMAT_PATHS := $(PACKAGE) $(TESTS) scripts
UV ?= uv
UV_RUN ?= $(UV) run
export BLACK_CACHE_DIR := .black-cache
export PYLINTHOME := .pylint-cache

.PHONY: install install-dev lock lint format format-check test coverage-xml docs build version version-next

install:
	python -m pip install .

install-dev:
	$(UV) sync --locked --all-groups

lock:
	$(UV) lock

lint:
	$(UV_RUN) --locked --no-default-groups --group lint python -m pylint --errors-only $(PACKAGE)
	$(UV_RUN) --locked --no-default-groups --group lint python -m pylint --fail-under=9.0 $(PACKAGE)

format:
	$(UV_RUN) --locked --no-default-groups --group lint python -m isort $(FORMAT_PATHS)
	$(UV_RUN) --locked --no-default-groups --group lint python -m black $(FORMAT_PATHS)

format-check:
	$(UV_RUN) --locked --no-default-groups --group lint python -m isort --check-only $(FORMAT_PATHS)
	$(UV_RUN) --locked --no-default-groups --group lint python -m black --check $(FORMAT_PATHS)

test:
	$(UV_RUN) --locked --no-default-groups --group test python -m pytest -q

coverage-xml:
	$(UV_RUN) --locked --no-default-groups --group test python -m pytest -q --cov=genplanner --cov-report=term-missing --cov-report=xml:coverage.xml

docs:
	$(UV_RUN) --locked --no-default-groups --group docs sphinx-build -n -W -b html -d docs/_build/doctrees docs/source docs/_build/html --keep-going

build:
	$(UV) build

version:
	python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])"

version-next:
	$(UV_RUN) --locked --no-default-groups --group release semantic-release --noop version --print
