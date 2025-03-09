# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to documentation with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'MetaDrive'
copyright = 'MetaDriverse'
author = 'MetaDriverse'

# The full version, including alpha/beta/rc tags
release = '0.1.1'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    # 'nbsphinx',
    "myst_nb",
    "sphinx.ext.autosectionlabel",
    "sphinx_rtd_theme",
    "sphinx.builders.linkcheck",
    "sphinx_copybutton"
]

autosectionlabel_prefix_document = True

nbsphinx_execute = 'never'
nb_execution_mode = 'off'
nb_execution_timeout = 300

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'collapse_navigation': False,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['custom.css']

html_css_files = ['custom.css']

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]
myst_url_schemes = ("http", "https", "mailto")

# This will add the colab execution link automatically!
import os
import nbformat


def add_hyperlink_to_notebook(notebook_path):
    """
    Add the colab link to notebook!
    """
    nb = nbformat.read(notebook_path, as_version=4)
    link_text = "https://colab.research.google.com/github/metadriverse/metadrive/blob/main/documentation/source/{}".format(
        os.path.basename(notebook_path))
    link_text = "[![Click and Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({})".format(
        link_text)
    # Assuming you want to add the link below the first heading
    cell = nb.cells[0]
    assert cell.cell_type == 'markdown', cell
    if "colab" not in cell.source:
        p = cell.source.find("\n")
        cell.source = cell.source[:p] + f"\n\n{link_text}\n" + cell.source[p:]
        # print(cell.source)
        nbformat.write(nb, notebook_path)


def process_notebooks(directory):
    """
    Add to all notebook.
    """
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".ipynb"):
                add_hyperlink_to_notebook(os.path.join(root, file))


process_notebooks(os.path.dirname(__file__))
