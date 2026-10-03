"""The optimizer tests need its optional dependencies; without them this directory is not collected."""

import importlib.util
import os
import shutil
import sys

MISSING = [name for name in ("z3", "func_timeout", "psutil") if importlib.util.find_spec(name) is None]
if not (shutil.which("Singular") or shutil.which("Singular", path=os.path.dirname(sys.executable))):
    MISSING.append("Singular binary")
if MISSING:
    collect_ignore_glob = ["test_*.py"]


def pytest_report_header(config):
    if MISSING:
        return f"tests/optimize skipped, optional optimizer dependencies missing: {', '.join(MISSING)}"
