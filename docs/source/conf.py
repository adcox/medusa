# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys
import functools
import inspect
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
autodoc_member_order = "bysource"
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

# -- Custom hooks ------------------------------------------------------------


def annotateOverride(app, what, name, obj, skip, lines):
    """
    Plug for the 'autodoc-process-docstring' hook that adds an annotation if
    a documented object overrides another object.

    Per `PEP 698`_ (python 3.12+), the ``@typing.override`` decorator will attempt
    to add an attribute ``__override__`` with value ``True`` at runtime to the
    decorated object. For older versions of python, the `overrides`_ package
    duplicates this behavior.

    This method checks a documented object for the ``__override__`` attribute
    and adds a line ``"Overrides: {base}"`` line to the docstring where ``{base}``
    is the object being overriden.

    Args:
        app (sphinx.application.Sphinx): the sphinx application object
        what (str): the type of the object which the docstring belongs to (one
            of `'module'`, `'class'`, `'exception'`, `'function'`, `'method'`,
            `'attribute'`)
        name (str): the fully qualified name of the object
        obj (object): the object itself
        skip (dict): the options given to the directive: an object with attributes
            `inherited-members`, `undoc-members`, `show-inheritance`, and `no-index`
            that are true if the flag option of the same name was given to the
            auto directive.
        lines (list[str]): the lines of the docstring. These can be edited in place
            to modify the documentation.

    See: https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#docstring-preprocessing

    .. _PEP 698: https://peps.python.org/pep-0698/
    """

    # Only proceed if show-inheritance is set to True
    if not getattr(skip, "show-inheritance", False):
        return

    # Figure out of the object overrides something
    overrider = None
    try:
        if what == "property":
            # Property objects obscure the actual function, check the getter/setter
            if getattr(obj.fget, "__override__", False):
                overrider = obj.fget
            elif getattr(obj.fst, "__override__", False):
                overrider = obj.fset
        else:
            if getattr(obj, "__override__", False):
                overrider = obj
    except Exception:
        pass

    # If the object is overriding something, try to figure out what it is overriding
    if overrider is not None:
        overrideStr = "Overrides: "

        try:
            # Get the class that defines the overrider object
            cls = _getDefiningParent(overrider)
            if cls is not None:
                # Find the first class in the MRO that also defines an attribute
                # with the same name as overrider
                mro = inspect.getmro(cls)
                for supercls in mro[1:]:
                    if hasattr(supercls, overrider.__name__):
                        ref = _objRef(what)
                        overrideStr += f":{ref}:`{supercls.__module__}.{supercls.__name__}.{overrider.__name__}`"
        except Exception:
            overrideStr += "*??*"  # could not determine object being overriden

        lines.insert(0, overrideStr)
        lines.insert(1, "")  # blank line after


def setup(app):
    app.connect("autodoc-process-docstring", annotateOverride)


def _getDefiningParent(meth):
    """
    Get the class object that defines the method

    Source:
        https://stackoverflow.com/questions/3589311/get-defining-class-of-unbound-method-object-in-python-3
    """
    if isinstance(meth, functools.partial):
        return _getDefiningParent(meth.func)
    if inspect.ismethod(meth) or (
        inspect.isbuiltin(meth)
        and getattr(meth, "__self__", None) is not None
        and getattr(meth.__self__, "__class__", None)
    ):
        for cls in inspect.getmro(meth.__self__.__class__):
            if meth.__name__ in cls.__dict__:
                return cls
        meth = getattr(meth, "__func__", meth)  # fallback to __qualname__ parsing
    if inspect.isfunction(meth):
        cls = getattr(
            inspect.getmodule(meth),
            meth.__qualname__.split(".<locals>", 1)[0].rsplit(".", 1)[0],
            None,
        )
        if isinstance(cls, type):
            return cls
    return getattr(meth, "__objclass__", None)  # handle special descriptor objects


def _objRef(what):
    """Convert object type (``what``) into sphinx reference language"""
    if what == "module":
        return "mod"
    if what == "class":
        return "class"
    if what == "function":
        return "func"
    if what == "method":
        return "meth"
    if what in ("property", "attribute"):
        return "attr"
    if what == "exception":
        return "exc"

    return "obj"
