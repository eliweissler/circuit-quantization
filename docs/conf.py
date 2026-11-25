"""Sphinx configuration for circuit-quantization."""

from importlib.metadata import version as _version

project = "circuit-quantization"
author = "Eli Weissler"
copyright = "2026, Eli Weissler"
release = _version("circuit-quantization")

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
]
myst_enable_extensions = ["dollarmath", "amsmath"]
myst_heading_anchors = 3

nb_execution_mode = "cache"
nb_execution_timeout = 300
nb_execution_raise_on_error = True

autodoc_member_order = "bysource"
autodoc_mock_imports = ["z3", "func_timeout"]

exclude_patterns = ["_build", "jupyter_execute", "superpowers", "**.ipynb_checkpoints"]
html_theme = "furo"
html_title = "circuit-quantization"
