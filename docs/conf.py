import os
import sys
import django

sys.path.insert(0, os.path.abspath('../src'))
os.environ['DJANGO_SETTINGS_MODULE'] = 'gae.settings'
django.setup()


# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'GAE-vehicle-safety'
copyright = '2023, Anahita Pakiman'
author = 'Anahita Pakiman'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',  # to autogenerate .rst files
    'sphinx.ext.napoleon',  # to parse google stye python docstrings
    'sphinx.ext.mathjax',  # to include math expressions in the .rst files
    'recommonmark',  # to include markdown files in sphinx documentation
    'nbsphinx'  # to include jupyter notebooks
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
