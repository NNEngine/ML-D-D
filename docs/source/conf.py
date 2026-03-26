# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))  # docs/source/ → project root

# -- Project information -----------------------------------------------------
project = 'ml_D_D'
copyright = '2026, Shivam Sharma'
author = 'Shivam Sharma'
release = 'v1.0.4'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',        # Auto-generate docs from docstrings
    'sphinx.ext.viewcode',       # Add source code links
    'sphinx.ext.napoleon',       # Support Google/NumPy docstring styles
    'sphinx.ext.autosummary',    # Generate summary tables
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'  # ✅ Only defined once now
html_static_path = ['_static']
