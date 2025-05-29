# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'superleaf'
copyright = '2025, Erik Schomburg'
author = 'Erik Schomburg'
release = '0.2.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

import os
import sys

# assuming conf.py sits at repo_root/docs/conf.py
sys.path.insert(0, os.path.abspath('../src'))

extensions = [
    'sphinx.ext.autodoc',      # core autodoc support
    'sphinx.ext.napoleon',     # for Google/NumPy‑style docstrings
    'sphinx.ext.autosummary',  # generates stub pages automatically
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Tell Sphinx to write out stub .rst files for every module it discovers:
autosummary_generate = True
# (Optional) Overwrite existing stub files on each build:
autosummary_generate_overwrite = True

autodoc_default_options = {
    'members': True,            # include all methods & attributes
    'undoc-members': True,      # even if no docstring
    'inherited-members': True,  # show methods from base classes
    'show-inheritance': True,
}
# For classes, also merge class doc + __init__ doc:
autoclass_content = 'both'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
