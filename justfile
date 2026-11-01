# Justfile for analog-daddy

python_version := "3.14"

default:
    just --choose

alias b := build
alias c := clean
alias d := docs-serve
alias t := test
alias tc := type-check
alias m := marimo

# Launch marimo notebook
marimo notebook="src/analog_daddy/drafts.py":
    uv run marimo edit --headless --no-token --skip-update-check --no-sandbox {{ notebook }}

# Type check the project with ty
type-check:
    uv run --python={{ python_version }} ty check .

# Run all the formatting, linting, and testing commands
qa: (qa-path ".")

# Format, lint, type-check, and test a given path
qa-path path:
    uv run --python={{ python_version }} ruff check --fix {{ path }}
    uv run --python={{ python_version }} ruff format {{ path }}
    uv run --python={{ python_version }} format-docstring {{ path }}
    uv run --python={{ python_version }} --isolated --only-group lint pydoclint {{ path }}
    uv run --python={{ python_version }} ty check {{ path }}
    uv run --python={{ python_version }} pytest {{ path }}

# Run all the tests for all the supported Python versions
testall:
    uv run --python=3.12 pytest
    uv run --python=3.13 pytest
    uv run --python={{ python_version }} pytest

# Run all the tests, but allow for arguments to be passed
test *ARGS:
    @echo "Running with arg: {{ ARGS }}"
    uv run --python={{ python_version }} pytest {{ ARGS }}

# Serve docs locally with live reload
docs-serve:
    -lsof -ti :8000 | xargs kill
    uv run --group docs zensical serve

# Build docs (strict mode, fails on warnings)
docs-build:
    uv run --group docs zensical build --clean

# Build the project, useful for checking that packaging is correct
build:
    rm -rf build
    rm -rf dist
    uv build

# Tag, push, and create a GitHub release
release:
    uv run scripts/release.py

# Remove all build, test and Python artifacts
clean:
    just clean-build
    just clean-pyc
    just clean-test

# Remove build artifacts
clean-build:
    rm -fr build/
    rm -fr dist/
    rm -fr .eggs/
    find . -name '*.egg-info' -exec rm -fr {} +
    find . -name '*.egg' -exec rm -f {} +

# Remove Python file artifacts
clean-pyc:
    find . -name '*.pyc' -exec rm -f {} +
    find . -name '*.pyo' -exec rm -f {} +
    find . -name '*~' -exec rm -f {} +
    find . -name '__pycache__' -exec rm -fr {} +
    find . -name '.rumdl_cache' -exec rm -fr {} +
    find . -name '.ruff_cache' -exec rm -fr {} +

# Remove test artifacts
clean-test:
    rm -fr .pytest_cache

# Publish to PyPI (manual alternative to GitHub Actions)
publish:
    uv build
    uv publish
