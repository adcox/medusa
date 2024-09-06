# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys

rootpath = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, rootpath)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "medusa"
copyright = "2024, Andrew Cox"
author = "Andrew Cox"
release = "0.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinxcontrib.fulltoc",
    # plot_directive - Run a script that displays a plot
    "matplotlib.sphinxext.plot_directive",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

autoclass_content = "both"
autodoc_member_order = "alphabetical"
autosummary_generate = True

# matplotlib.sphinxext.plot_directive settings
plot_working_directory = os.path.abspath(os.path.join(rootpath, ".."))
plot_html_show_source_link = True

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "bizstyle"
html_static_path = ["_static"]

# Pygments (syntax highlighting) style
pygments_style = "monokai"
