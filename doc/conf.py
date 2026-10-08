# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


import importlib
import io
import os
import re
import subprocess
import sys
from pathlib import Path

import tomllib

# relative path so sphinx can locate the different modules directly for autosummary
sys.path.append(str(Path(__file__).parents[1]))
sys.path.append(str(Path(__file__).parents[1] / "recon_surf"))
sys.path.append(str(Path(__file__).parent / "sphinx_ext"))

from resolve_links import LinkCodeResolver
from FastSurferCNN.gpu_support import LEGACY_BUILD, MIN_DRIVER
from FastSurferCNN.version import main as _version_info, parse_build_file

project = "FastSurfer"
author = "FastSurfer Developers"
copyright = f"2020-2026, {author}"
gh_url = "https://github.com/Deep-MI/FastSurfer"

# run the version script and save in build dir
_streambuf = io.StringIO()
_version_info(file=_streambuf)
_version_dict = parse_build_file(_streambuf)

# the commit, not the branch: `git_branch` needs a non-empty `sections` to be filled in at all, and
# actions/checkout leaves a detached HEAD where `git branch --show-current` is empty anyway. The
# hash is optional in the version line, so fall back to a ref that exists rather than to nothing.
commit = _version_dict["git_hash"] or "dev"
version = _version_dict["version"]

# doc.yml publishes each build to gh-pages under the ref it was built from, so that ref is what
# says whether this tree documents a release. It is read from the environment rather than from git,
# because actions/checkout leaves a detached HEAD and `git branch --show-current` is empty there.
# The tag pattern matches the release tags this project actually uses, all of them X.Y.Z. A tag
# with a suffix falls through to the development wording, which is the safe way round. doc.yml
# publishes no tags today, so only "stable" reaches this in practice.
publish_ref = os.environ.get("GITHUB_REF_NAME", "")
documents_a_release = publish_ref == "stable" or re.fullmatch(r"v\d+\.\d+\.\d+", publish_ref) is not None


def _latest_release() -> str:
    """Return the version of the newest release tag (vX.Y.Z) of the repository."""
    tags = subprocess.run(
        ["git", "tag", "--list", "v*"], cwd=Path(__file__).parents[1], capture_output=True, text=True, check=True,
    ).stdout.split()
    releases = [tuple(map(int, m.groups())) for m in map(re.compile(r"v(\d+)\.(\d+)\.(\d+)").fullmatch, tags) if m]
    if not releases:
        raise RuntimeError(
            "The documentation of a development version refers to the newest release, but the repository has no "
            "release tags (vX.Y.Z), fetch them with `git fetch --tags`."
        )
    return ".".join(map(str, max(releases)))


# Official Docker images only exist for releases, so commands in the documentation (e.g. docker image tags) use the
# version from pyproject.toml if this tree documents a release, and the newest release otherwise.
image_version = version if documents_a_release else _latest_release()


def _read_file_gitref(path: str, ref: str | None) -> str:
    """Return the text of path (relative to the repository root), in the working tree or at ref."""
    root = Path(__file__).parents[1]
    if ref is None:
        return (root / path).read_text()
    return subprocess.run(
        ["git", "show", f"{ref}:{path}"], cwd=root, capture_output=True, text=True, check=True,
    ).stdout


# tool.<key> of releases whose pyproject.toml predates the key (python.version: ARG PYTHON_VERSION of their
# tools/Docker/Dockerfile, cuda.version and docker.runtime_base: DEFAULTS.CUDA_VERSION, DEFAULTS.ROCM_VERSION,
# DEFAULTS.BUILD_BASE_IMAGE and DEFAULTS.RUNTIME_BASE_IMAGE of their tools/Docker/build.py); remove an
# entry once the newest release defines the keys
_TOOL_VALUES_FALLBACK = {
    "v2.5.4": {
        "python.version": "3.12",
        "cuda.version": "12.8",
        "rocm.version": "6.3",
        "docker.runtime_base": "ubuntu:24.04",
        "docker.build_base": "ubuntu:24.04",
    },
}


def _tool_values_gitref(ref: str | None, *keys: str) -> tuple[str, ...]:
    """Return tool.<key> of pyproject.toml for each of keys (e.g. python.version), in the working tree or at ref."""
    tool = tomllib.loads(_read_file_gitref("pyproject.toml", ref)).get("tool", {})
    values = dict(_TOOL_VALUES_FALLBACK.get(ref, {}))
    for key in keys:
        section, _, name = key.partition(".")
        if name in tool.get(section, {}):
            values[key] = tool[section][name]
    if missing := [key for key in keys if key not in values]:
        raise RuntimeError(
            f"pyproject.toml ({ref or 'working tree'}) does not define "
            f"{', '.join(f'tool.{key}' for key in missing)}."
        )
    return tuple(values[key] for key in keys)


# the tree the images named by image_version were built from, which is also what the native installation clones
# (--branch stable), so the versions of the software in both come from there as well
_image_ref = None if documents_a_release else f"v{image_version}"
# the CUDA and ROCm versions and the base image of the images named by image_version, the CUDA version is the one of
# the `latest` image
version_python, version_freesurfer, version_cuda, version_rocm, _runtime_base_image = _tool_values_gitref(
    _image_ref, "python.version", "freesurfer.version", "cuda.version", "rocm.version", "docker.runtime_base",
)
if not _runtime_base_image.startswith("ubuntu:"):
    raise RuntimeError(f"UBUNTU_VERSION needs an ubuntu image, tool.docker.runtime_base is {_runtime_base_image}.")
version_ubuntu = _runtime_base_image.removeprefix("ubuntu:")
# the PyTorch backend and device name of that CUDA version (13.2 -> cu132)
image_cuda = "cu" + version_cuda.replace(".", "")
image_rocm = "rocm" + version_rocm
# the CUDA images this tree builds, their GPUs and oldest drivers, for the table of which image fits which GPU; from
# this tree, not image_version, so the table matches pyproject.toml and the message FastSurfer prints for a GPU it
# cannot use
(version_cuda_default,) = _tool_values_gitref(None, "cuda.version")
image_cuda_default = "cu" + version_cuda_default.replace(".", "")
(_legacy_major, _legacy_minor), image_cuda_legacy = LEGACY_BUILD
version_cuda_legacy = f"{_legacy_major}.{_legacy_minor}"
driver_cuda = MIN_DRIVER[int(version_cuda_default.split(".")[0])]
driver_cuda_legacy = MIN_DRIVER[_legacy_major]

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
    "sphinxcontrib.programoutput",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_issues",
    # sphinx.ext.autosectionlabel and nbsphinx together with sphinxarg.ext causes a
    # duplicate label warning: https://github.com/spatialaudio/nbsphinx/issues/787
    # nbsphinx is currently not 'needed' as we do not include ipynb files.
    # "nbsphinx",
    "IPython.sphinxext.ipython_console_highlighting",
    "myst_parser",
    "sphinxarg.ext",
    "fix_links",
]

# Suppress myst.xref_missing warning and  i.e A target was
# not found for a cross-reference
# Reference: https://myst-parser.readthedocs.io/en/latest/configuration.html#build-warnings
suppress_warnings = [
    # "myst.xref_missing",
    "myst.duplicate_def",
    "autosectionlabel",
]

# create anchors for which headings?
myst_heading_anchors = 7

# myst extensions
myst_enable_extensions = {
    "substitution",
}

# configure substitutions, fix_links also replaces string substitutions inside code, which MyST does not
myst_substitutions = {
    "FASTSURFER_VERSION": image_version,
    # the same version for the folder of the macOS package, without the note on docker images
    "PACKAGE_VERSION": image_version,
    "CUDA_STRING": image_cuda,
    "CUDA_VERSION": version_cuda,
    "CUDA_DEFAULT_STRING": image_cuda_default,
    "CUDA_DEFAULT_VERSION": version_cuda_default,
    "CUDA_DRIVER": str(driver_cuda),
    "CUDA_LEGACY_STRING": image_cuda_legacy,
    "CUDA_LEGACY_VERSION": version_cuda_legacy,
    "CUDA_LEGACY_DRIVER": str(driver_cuda_legacy),
    "ROCM_STRING": image_rocm,
    "ROCM_VERSION": version_rocm,
    "PYTHON_VERSION": version_python,
    "UBUNTU_VERSION": version_ubuntu,
    "FREESURFER_VERSION": version_freesurfer,
}

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
    # instructions for writing the documentation, not part of it
    "AGENTS.md",
    "CONVENTIONS.md",
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
html_js_files = ["doc-version-link.js"]  # points the announcement bar at the sibling doc tree
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

# doc.yml publishes each build to gh-pages under the ref it was built from, so that ref is what
# says whether this tree documents a release. It is read from the environment rather than from git,
# because actions/checkout leaves a detached HEAD and `git branch --show-current` is empty there.
# The tag pattern matches the release tags this project actually uses, all of them X.Y.Z. A tag
# with a suffix falls through to the development wording, which is the safe way round. doc.yml
# publishes no tags today, so only "stable" reaches this in practice.
publish_ref = os.environ.get("GITHUB_REF_NAME", "")
documents_a_release = publish_ref == "stable" or re.fullmatch(r"v\d+\.\d+\.\d+", publish_ref) is not None

# The announcement bar is the only site-wide notice furo offers, so it also carries the link to the
# other published tree. The href here is a fallback that is only correct at the tree root;
# doc-version-link.js rewrites it for the page it actually ends up on.
_other_tree = "dev" if documents_a_release else "stable"
_other_link = (
    f'<a href="../{_other_tree}/" data-doc-tree="{_other_tree}">{_other_tree} documentation</a>'
)

if documents_a_release:
    html_theme_options["announcement"] = (
        f"This documents the latest release. The {_other_link} covers changes that are not "
        "released yet."
    )
else:
    # Every other build, the dev tree included, describes code ahead of the newest release.
    html_theme_options["announcement"] = (
        "You are reading the documentation of the development version. It may describe features "
        f"and options that are not part of a release yet. See the {_other_link} for the latest "
        "release."
    )


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

# -- sphinx-copybutton -------------------------------------------------------
# ```text fences are explanations with placeholders (see CONVENTIONS.md), so only other code gets a copy button
copybutton_selector = "div:not(.highlight-text) > div.highlight > pre"

# -- sphinxcontrib-programoutput ---------------------------------------------
# command-output shows the command above its output; output blocks have no `$` prompt (see CONVENTIONS.md)
programoutput_prompt_template = "{command}\n{output}"

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
    r'\.WarmupCosineLR\.step$',  # Exclude due to error in inherited step
}

# -- sphinxcontrib-bibtex ----------------------------------------------------
bibtex_bibfiles = ["./references.bib"]

# -- sphinx.ext.linkcode -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/linkcode.html

def import_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# linking at the commit rather than at a branch keeps the line numbers in each link matching the
# code that was documented, however far the branch moves afterwards
linkcode_resolve = LinkCodeResolver(gh_url, commit)

_re_script_dirs = "fastsurfercnn|cerebnet|recon_surf|hypvinn|corpuscallosum"
_up = "^/\\.\\./"
_end = "(\\.md)?(#.*)?$"

# -- sphinx_ext.fix_links -----------------------------------------------------
# re_reference_target=(regex) => used in missing-reference
fix_links_target = {
    # all regexpr are ignorecase, individual replacements are applied until no further
    # change occurs, but different (different repl str) replacements are not combined
    # "^\\/overview\\/intro\\.md#": "/overview/index.rst#",
    "^/?(.*)#(.*)ubuntu-(\\d{2})(\\d{2})": ("/\\1#\\2ubuntu-\\3-\\4",),
    f"{_up}readme{_end}": ("/index.rst\\1", "/overview/intro.rst\\1"),
    "^/overview/intro(#.*)?$": ("/overview/index.rst\\2",),
    f"{_up}/tools/docker/readme{_end}": ("/overview/docker.rst\\2",),
    f"{_up}({_re_script_dirs})/readme{_end}": ("/scripts/\\1.rst\\2",),
    f"{_up}license": ("/overview/license.rst",),
}
fix_links_alternative_targets = {
    "/overview/intro": ("/index.rst", "/overview/index.rst"),
}
fix_links_project_root = Path("..")
# set of substitution names => text (one MyST markdown paragraph) of the note fix_links renders on top of each fenced
# code block that uses exactly these of the names used here as `{{ name }}`, sets without an entry get no note
# the commands use the official images, which may not match this tree's version (FastSurfer, default CUDA version)
_docker_hub = f"[Docker Hub](https://hub.docker.com/r/deepmi/fastsurfer/tags?name=v{image_version})"
_torch_docs_url = f"(https://pytorch.org/docs)"
if documents_a_release:
    fix_links_substitution_banners = {
        frozenset({"CUDA_STRING"}): (
            f"The commands below use {image_cuda}, which references the default CUDA version ({version_cuda}) of "
            f"FastSurfer {image_version}. Other CUDA versions are supported by [PyTorch]({_torch_docs_url}), but "
            f"depend on the PyTorch version. If you use `uv`, then `--torch-backend auto` automatically lets `uv` "
            f"decide."
        ),
        frozenset({"FASTSURFER_VERSION", "CUDA_STRING"}): (
            f"The commands below use the tagged FastSurfer image `:{image_cuda}-v{image_version}`. CUDA {version_cuda} "
            f"is the default CUDA version bundled in both `:latest` and that image. Images of {image_version} for "
            f"other CUDA versions, ROCm and CPU are available on {_docker_hub}."
        ),
    }
else:
    _latest_release_str = (
        f"This documents the development version {version}. Official Docker images only exist for releases, so the "
        f"commands below use the latest release, {image_version}"
    )
    _build_image = "To run the development version, {doc}`build your own image </overview/docker>`."
    fix_links_substitution_banners = {
        frozenset({"FASTSURFER_VERSION"}): f"{_latest_release_str}. {_build_image}",
        frozenset({"CUDA_STRING"}): (
            f"{_latest_release_str}. The commands below use {image_cuda}, which references the default CUDA version "
            f"({version_cuda}) of FastSurfer {image_version}. The default PyTorch and CUDA versions might be different "
            f"for this development version (see supported [PyTorch's documentation]({_torch_docs_url}). If you use "
            f"`uv`, then `--torch-backend auto` automatically lets `uv` decide."
        ),
        frozenset({"FASTSURFER_VERSION", "CUDA_STRING"}): (
            f"{_latest_release_str}, for its default CUDA version, {version_cuda}. Images of {image_version} for other "
            f"CUDA versions, ROCm and CPU are available on {_docker_hub}. {_build_image}"
        ),
    }

