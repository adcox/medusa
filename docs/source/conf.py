# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys
from importlib import metadata

rootpath = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, rootpath)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "medusa"
copyright = "2024, Andrew Cox"
author = "Andrew Cox"
release = metadata.version("medusa")

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    # plot_directive - Run a script that displays a plot
    "matplotlib.sphinxext.plot_directive",
    # math syntax
    "sphinx.ext.mathjax",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

autoclass_content = "both"
autodoc_member_order = "alphabetical"
autodoc_typehints = "description"
autosummary_generate = True

# matplotlib.sphinxext.plot_directive settings
plot_working_directory = os.path.abspath(os.path.join(rootpath, ".."))
plot_html_show_source_link = True

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

# Pygments (syntax highlighting) style
pygments_style = "monokai"
