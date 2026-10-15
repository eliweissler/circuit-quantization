"""Shared test setup: tests/circuits.py on the import path, and --runslow as in sircuitenum's conftest."""

import sys
from pathlib import Path

import pytest

# tests/circuits.py holds the notes' circuits; make it importable from subdirectories too.
sys.path.insert(0, str(Path(__file__).parent))


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test to run only with --runslow option.")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
