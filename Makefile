CODE := app

build-and-publish: clean build publish

lint:
	poetry run pylint $(CODE)

format:
	poetry run isort $(CODE)
	poetry run black $(CODE)

install:
	pip install .

install-dev:
	poetry install --with dev

install-dev-pip:
	pip install -e . --config-settings editable_mode=strict

clean:
	rm -rf ./dist

build:
	poetry build

publish:
	poetry publish

update:
	poetry update

dev_install_rust:
	poetry run maturin develop

build_release_rust:
	poetry run maturin build --release

.PHONY: sync-version
sync-version:
	python scripts/sync_version.py

.PHONY: version-patch version-minor version-major
version-patch: # 0.0.v
	poetry version patch
	$(MAKE) sync-version

version-minor: # 0.v.0
	poetry version minor
	$(MAKE) sync-version

version-major: # v.0.0
	poetry version major
	$(MAKE) sync-version

VERSION := $(shell poetry version -s)

tag:
	git tag v$(VERSION)
	@echo "Created tag v$(VERSION)"

push-tag:
	git push origin v$(VERSION)

release: tag push-tag
	@echo "Tagged and pushed v$(VERSION)."

docs:
	poetry run sphinx-build -b html docs/source docs/build/html