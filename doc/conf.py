# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


import inspect
from importlib import import_module
import sys
import os

# here i added the relative path because sphinx was not able
# to locate FastSurferCNN module directly for autosummary
sys.path.append(os.path.dirname(__file__) + "/..")
sys.path.append(os.path.dirname(__file__) + "/../recon_surf")
sys.path.append(os.path.dirname(__file__) + "/sphinx_ext")

project = "FastSurfer"
author = "FastSurfer Developers"
copyright = f"2020, {author}"
gh_url = "https://github.com/deep-mi/FastSurfer"


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# If your documentation needs a minimal Sphinx version, state it here.
needs_sphinx = "5.0"

# The document name of the “root” document, that is, the document that contains
# the root toctree directive.
root_doc = "index"


# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named "sphinx.ext.*") or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "numpydoc",
    "sphinxcontrib.bibtex",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_issues",
    "nbsphinx",
    "IPython.sphinxext.ipython_console_highlighting",
    "myst_parser",
    "re_reference",
]

# Suppress myst.xref_missing warning and  i.e A target was
# not found for a cross-reference
# Reference: https://myst-parser.readthedocs.io/en/latest/configuration.html#build-warnings
suppress_warnings = [
    # "myst.xref_missing",
    "myst.duplicate_def",
]

myst_heading_anchors = 1

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
]


# Sphinx will warn about all references where the target cannot be found.
nitpicky = False
nitpick_ignore = []

# A list of ignored prefixes for module index sorting.
# modindex_common_prefix = [f"{package}."]

# The name of a reST role (builtin or Sphinx extension) to use as the default
# role, that is, for text marked up `like this`. This can be set to 'py:obj' to
# make `filter` a cross-reference to the Python function “filter”.
default_role = "py:obj"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_theme = "furo"
html_static_path = ["_static"]
html_title = project
html_show_sphinx = False

# Documentation to change footer icons:
# https://pradyunsg.me/furo/customisation/footer/#changing-footer-icons
html_theme_options = {
    "footer_icons": [
        {
            "name": "GitHub",
            "url": gh_url,
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            "class": "",
        },
    ],
}


# -- autosummary -------------------------------------------------------------
autosummary_generate = True

# -- autodoc -----------------------------------------------------------------
autodoc_typehints = "none"
autodoc_member_order = "groupwise"
autodoc_warningiserror = True
autoclass_content = "class"


# -- intersphinx -------------------------------------------------------------
intersphinx_mapping = {
    "matplotlib": ("https://matplotlib.org/stable", None),
    "mne": ("https://mne.tools/stable/", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "python": ("https://docs.python.org/3", None),
    # "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}
intersphinx_timeout = 5


# -- sphinx-issues -----------------------------------------------------------
issues_github_path = gh_url.split("https://github.com/")[-1]

# -- autosectionlabels -------------------------------------------------------
autosectionlabel_prefix_document = True

# -- numpydoc ----------------------------------------------------------------
numpydoc_class_members_toctree = False
numpydoc_attributes_as_param_list = False
# numpydoc_show_class_members = True


# x-ref
numpydoc_xref_param_type = True
numpydoc_xref_aliases = {
    # Matplotlib
    "Axes": "matplotlib.axes.Axes",
    "Figure": "matplotlib.figure.Figure",
    # Python
    "bool": ":class:`python:bool`",
    "Path": "pathlib.Path",
    "TextIO": "io.TextIOBase",
    # Scipy
    "csc_matrix": "scipy.sparse.csc_matrix",
}
# numpydoc_xref_ignore = {}

# validation
# https://numpydoc.readthedocs.io/en/latest/validation.html#validation-checks
error_ignores = {
    "GL01",  # docstring should start in the line immediately after the quotes
    "EX01",  # section 'Examples' not found
    "ES01",  # no extended summary found
    "SA01",  # section 'See Also' not found
    "RT02",  # The first line of the Returns section should contain only the type, unless multiple values are being returned  # noqa
    "PR01",  # Parameters {missing_params} not documented
    "GL08",  # The object does not have a docstring
    "SS05",  # Summary must start with infinitive verb, not third person
    "RT01",  # No Returns section found
    "SS06",  # Summary should fit in a single line
    "GL02",  # Closing quotes should be placed in the line after the last text
    "GL03",  # Double line break found; please use only one blank line to
    "SS03",  # Summary does not end with a period
    "YD01",  # No Yields section found
    "PR02",  # Unknown parameters {unknown_params}
    "SS01",  # Short summary in a single should be present at the beginning of the docstring.
}
numpydoc_validate = True
numpydoc_validation_checks = {"all"} | set(error_ignores)
numpydoc_validation_exclude = {  # regex to ignore during docstring check
    r"\.__getitem__",
    r"\.__contains__",
    r"\.__hash__",
    r"\.__mul__",
    r"\.__sub__",
    r"\.__add__",
    r"\.__iter__",
    r"\.__div__",
    r"\.__neg__",
}

# -- sphinxcontrib-bibtex ----------------------------------------------------
bibtex_bibfiles = ["./references.bib"]

# -- sphinx.ext.linkcode -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/linkcode.html

#  Alternative method for linking to code by Osama, not sure which one is better
from urllib.parse import quote



# https://github.com/python-websockets/websockets/blob/e217458ef8b692e45ca6f66c5aeb7fad0aee97ee/docs/conf.py#L102-L134
def linkcode_resolve(domain, info):
    # Check if the domain is Python, if not return None
    if domain != "py":
        return None
    if not info["module"]:
        return None

    # Import the module using the module information
    mod = import_module(info["module"])

    # Check if the fullname contains a ".", indicating it's a method or attribute of
    # a class
    if "." in info["fullname"]:
        objname, attrname = info["fullname"].split(".")
        # Get the object from the module
        obj = getattr(mod, objname)
        try:
            # Try to get the attribute from the object
            obj = getattr(obj, attrname)
        except AttributeError:
            # If the attribute doesn't exist, return None
            return None
    else:
        # If the fullname doesn't contain a ".", get the object directly from the module
        obj = getattr(mod, info["fullname"])

    try:
        # Try to get the source file and line numbers of the object
        lines, first_line = inspect.getsourcelines(obj)
    except TypeError:
        # If the object is not a Python object that has a source file, return None
        return None

    # Replace "." with "/" in the module name to construct the file path
    filename = quote(info["module"].replace(".", "/"))
    # If the filename doesn't start with "tests", add a "/" at the beginning
    if not filename.startswith("tests"):
        filename = "/" + filename

    # Construct the URL that points to the source code of the object on GitHub
    return f"{gh_url}/blob/dev{filename}.py#L{first_line}-L{first_line + len(lines) - 1}"

# Which domains to search in to create links in markdown texts
myst_ref_domains = ["myst", "std", "py"]


# re-reftarget=(regex) => used in missing-reference
# re-refuri/refid=(regex) => used in doctree-
re_reference = {
    "re-refid=^((../)*)(Singularity|Docker)\\/README\\.md#": "/overview/\\3.md#",
    "re-reftarget=^\\/overview\\/intro\\.md#": "/index.rst#",
    "re-refid=^README.md#requirements-": "/index.rst#requirements-",
    "re-refid=^../README.md": "/index.rst",
}
