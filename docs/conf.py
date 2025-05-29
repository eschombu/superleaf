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

import inspect
import os
import sys

# assuming conf.py sits at repo_root/docs/conf.py
sys.path.insert(0, os.path.abspath('../src'))

extensions = [
    'sphinx.ext.autodoc',      # core autodoc support
    'sphinx.ext.napoleon',     # for Google/NumPy‑style docstrings
    'sphinx.ext.autosummary',  # generates stub pages automatically
    'sphinx.ext.linkcode',     # adds link to source code
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Tell Sphinx to write out stub .rst files for every module it discovers:
autosummary_generate = True
# (Optional) Overwrite existing stub files on each build:
autosummary_generate_overwrite = True

autodoc_default_options = {
    'members': True,             # include all methods & attributes
    'undoc-members': True,       # even if no docstring
    'inherited-members': False,  # show methods from base classes
    'show-inheritance': True,    # insert base classes after the class signature
    'member-order': 'bysource',  # disable alphabetical sorting
}
# For classes, also merge class doc + __init__ doc:
autoclass_content = 'both'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_context = {
    "github_user": "eschombu",
    "github_repo": "superleaf",
    "github_version": "master",
    "doc_path": "docs",
}
html_theme_options = {
    "github_url": "https://github.com/eschombu/superleaf",
    "use_edit_page_button": True,  # Adds an "Edit on GitHub" button
}
html_static_path = ['_static']


def linkcode_resolve(domain, info):
    if domain != "py":
        return None
    if not info["module"]:
        return None

    try:
        module = __import__(info["module"], fromlist=[info["fullname"]])
        obj = module
        for part in info["fullname"].split("."):
            obj = getattr(obj, part)
        fn = inspect.getsourcefile(obj)
        source, lineno = inspect.getsourcelines(obj)
    except Exception:
        return None

    # Customize for your repo URL and branch
    repo_url = "https://github.com/eschombu/superleaf"
    branch = "master"

    # Calculate the relative path from your repository root
    rel_path = os.path.relpath(fn, start=os.path.abspath(".."))

    # Return URL pointing directly to the source lines on GitHub
    return f"{repo_url}/blob/{branch}/{rel_path}#L{lineno}-L{lineno + len(source) - 1}"
