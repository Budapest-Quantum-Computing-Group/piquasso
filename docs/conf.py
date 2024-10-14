# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath(".."))


# -- Project information -----------------------------------------------------

project = "Piquasso"
copyright = "2021-2024, Budapest Quantum Computing Group"
author = "Budapest Quantum Computing Group"


# -- General configuration ---------------------------------------------------

add_module_names = False
autodoc_member_order = "bysource"
autoclass_content = "both"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "nbsphinx",
    "sphinx_design",
    "IPython.sphinxext.ipython_console_highlighting",
]


# Intersphinx setup
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "blackbird": ("https://quantum-blackbird.readthedocs.io/en/latest/", None),
    "scipy": ("http://docs.scipy.org/doc/scipy/reference/", None),
    "tensorflow": (
        "https://www.tensorflow.org/api_docs/python",
        "https://github.com/GPflow/tensorflow-intersphinx/raw/master/tf2_py_objects.inv",  # noqa: E501
    ),
}


# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"

html_static_path = ["_static"]
html_favicon = "_static/favicon.png"

pq_color = {
    "pq-grey": "#656565",
    "pq-smokey-white": "#f4f4f4",
    "pq-blue-shade1": "#1e2844",
    "pq-blue-shade2": "#273250",
    "pq-blue-shade3": "#2f3b5a",
    "pq-blue-shade4": "#364469",
    "pq-font-color-shade1": "#ccd1da",
    "pq-font-color-shade2": "#b2bac8",
    "pq-font-color-shade3": "#8995a8",
    "pq-font-color-shade4": "#717c8e",
    "pq-color-red": "#e46363",
    "pq-color-border": "#232d48",
    "pq-color-border-dark": "#1a243e",
}

html_theme_options = {
    "sidebar_hide_name": True,
    "light_logo": "logo_light.svg",
    "dark_logo": "logo_main.svg",
    "dark_css_variables": {
        "color-background-primary": pq_color["pq-blue-shade1"],
        "color-background-secondary": pq_color["pq-blue-shade2"],
        "color-code-background": pq_color["pq-blue-shade3"],
        "color-foreground-primary": pq_color["pq-font-color-shade1"],
        "color-foreground-secondary": pq_color["pq-font-color-shade2"],
        "color-sidebar-link-text--top-level": pq_color["pq-font-color-shade1"],
        "color-brand-primary": pq_color["pq-font-color-shade4"],
        "color-brand-content": pq_color["pq-color-red"],
        "color-admonition-title--note": pq_color["pq-color-red"],
        "color-admonition-background": pq_color["pq-blue-shade3"],
        "color-admonition-title-background--note": pq_color["pq-blue-shade2"],
        "color-background-hover": pq_color["pq-blue-shade3"],
    },
}

html_css_files = ["custom.css"]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".


# NBSPHINX

nbsphinx_prompt_width = "0"

nbsphinx_prolog = r"""
{% set docname = env.doc2path(env.docname, base=None) %}

.. note::
    .. raw:: html

        <div>
            You can download this notebook
            <a href="/{{ docname }}" download>here</a>.
        </div>
"""  # TODO: get base path.
