# Sphinx documentation

## Overview

This project is a parallel project intent to create a proper documentation along with code. The documentation can be markdown or pdf, code should ahve comments. This was not able to move forward since the gitlab of fraunhofer couldn't publish it in CI/CD. If the fraunhofer made it available this project can be used further to produce a proper documentation along with the source code.

<!-- Instructions to build documentation from this repository -->

## Instructions to build documentation from this repository

1. Clone [this](https://gitlab.scai.fraunhofer.de/ndv/research/automotive/cae_nlp/-/edit/ganesh/Sphinx%20Documentation) repository using
   ```
   git clone https://gitlab.scai.fraunhofer.de/ndv/research/automotive/cae_nlp/-/edit/ganesh/Sphinx%20Documentation.git
   ```
2. Install the following packages using `pip`.
   ```
   pip install sphinx
   pip install sphinx_rtd_theme
   pip install recommonmark
   pip install nbsphinx
   ```
3. `pandoc` is needed to generate documentation from Jupyter Notebooks using Sphinx. Installation instructions can be found [here](https://pandoc.org/installing.html).
4. Navigate to `Simple python code/docs/` directory and use the `make` command to generate HTML and LaTeX documentation.
   ```
   cd sphinx-doc-tutorial/docs/
   make html
   make latexpdf
   ```
5. The generated documentation can be found in the [`Simple python code/docs/_build/html/`]()
6. For more details refer the official website of [Sphinx](https://www.sphinx-doc.org/en/master/)

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- Instructions to generate Python documentation using Sphinx -->

### Instructions to generate Python documentation using Sphinx

To create your own documentation from scratch like the simple python code project, follow the instructions below.

<!-- Create documented code -->

### Step 1: Create documented code

- Create scripts and modules in Python.
- For automatic docstring generation in VSCode, use [Python Docstring Generator](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring) plugin with appropriate convention.
- Example docstrings following the sphinx convention is shown below:

  ```python
  def add( a, b ):
      """
      This function computes the sum of the two arguments.

      :param num1: first argument
      :type num1: float
      :param num2: second argument
      :type num2: float
      :return: sum of the two arguments
      :rtype: int or float

      .. note:: This function can accept :class:`int` parameters too.

      Example::
          result = add(a,b)
      """
      assert type(a) == type(b)
      return a + b
  ```

- **Note**: For other docstring conventions such as the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html), additional extensions
  such as [`sphinx.ext.napoleon`](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html) are necessary.
- Different docstring conventions can be used in the same project and [Sphinx](https://www.sphinx-doc.org/en/master/) (and its extensions) will parse them to generate documentation with a uniform convention.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- Setup sphinx project -->

### Step 2: Setup sphinx project

- Install `sphinx` package using pip, if not already installed.
- From your code's parent directory, create a `docs/` sub-directory to build the documentation files.
- In the `docs/` directory, initiate the sphinx project using `sphinx-quickstart` with the default options for the prompts as shown below.
- Separating source and build directories keeps the `docs/` directory structured.

  ```
  $ sphinx-quickstart
  Welcome to the Sphinx 4.0.3 quickstart utility.

  Please enter values for the following settings (just press Enter to
  accept a default value, if one is given in brackets).

  Selected root path: .

  You have two options for placing the build directory for Sphinx output.
  Either, you use a directory "_build" within the root path, or you separate
  "source" and "build" directories within the root path.
  > Separate source and build directories (y/n) [n]: y

  The project name will occur in several places in the built documentation.
  > Project name: Sphinx Doc Tutorial
  > Author name(s): Dinesh Krishna Natarajan
  > Project release []: 0.0.1

  If the documents are to be written in a language other than English,
  you can select a language here by its language code. Sphinx will then
  translate text that it generates into that language.

  For a list of supported codes, see
  https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-language.
  > Project language [en]: en

  Creating file sphinx-doc-tutorial/docs/source/conf.py.
  Creating file sphinx-doc-tutorial/docs/source/index.rst.
  Creating file sphinx-doc-tutorial/docs/Makefile.
  Creating file sphinx-doc-tutorial/docs/make.bat.

  Finished: An initial directory structure has been created.

  You should now populate your master file sphinx-doc-tutorial/docs/source/index.rst and create other documentation
  source files. Use the Makefile to build the docs, like so:
  make builder
  where "builder" is one of the supported builders, e.g. html, latex or linkcheck.



  ```

<p align="right">(<a href="#top">back to top</a>)</p>
   
<!-- Configure sphinx documentation builder -->   
### Step 3: Configure sphinx documentation builder
 * In the `docs/source/conf.py` file, uncomment the path setup comments shown below and edit the absolute path to the modules directory of the project.

    ```python
    # -- Path setup --------------------------------------------------------------

    # If extensions (or modules to document with autodoc) are in another directory,
    # add these directories to sys.path here. If the directory is relative to the
    # documentation root, use os.path.abspath to make it absolute, like shown here.
    #
    import os
    import sys
    sys.path.insert(0, os.path.abspath('../../utils'))
    ```

- In the `docs/source/conf.py` file, add 'sphinx.ext.autodoc' and other necessary extensions to the extensions list.

  ```python
  extensions = [ 'sphinx.ext.autodoc',  # to autogenerate .rst files
  'sphinx.ext.napoleon', # to parse google stye python docstrings
  'sphinx.ext.mathjax', # to include math expressions in the .rst files
  'recommonmark', # to include markdown files in sphinx documentation
  'nbsphinx' # to include jupyter notebooks
  ]
  ```

- Django connection: In your project folder, find /docs/conf.py and inside it, somewhere near the top of the file, find “#import os”. Just below it, write this:

  ```
  import os
  import sys
  import django
  sys.path.insert(0, os.path.abspath('..'))
  os.environ['DJANGO_SETTINGS_MODULE'] = 'Your_project_name.settings'
  django.setup()
  ```

- Sphinx-apidoc: This is the simpler method where you just need to navigate to your /docs folder and execute:

  ```
  sphinx-apidoc -o . ..
  ```

- In the same file, change the html theme if necessary. Note: The default installed theme is 'alabaster'. Any other theme might have to be installed via `pip install theme_name`. Available sphinx themes can be found [here](https://sphinx-themes.org/#themes).
  ```python
  html_theme = 'sphinx_rtd_theme'
  ```

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- Configure the `.rst` files -->

### Step 4: Configure the `.rst` files

- The 'reStructured Text' files indicate the contents of the documentation. [Here](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html) is a primer for the _reStructuredText_ markup language.
- The `index.rst` acts as the main file (equivalent to a Latex main file). The contents of the documentation can be added directly to the `index.rst` file or individual `.rst` files can be created and then referenced in the `index.rst` file. Each `.rst` file gets its own webpage.
- Documentation for the modules present in the `utils/arithmetic.py` and `utils/operations.py` can be automatically generated using the [`sphinx-apidoc`](https://www.sphinx-doc.org/en/master/man/sphinx-apidoc.html) command. From the `docs/` directory, the following command can be used to automatically generate the `.rst` files for the modules in `utils/` directory. The `-o` argument points to the output directory `source/`.
- To recursively generate the `.rst` files for the submodules inside the `utils/` folder, simply create an `__init__.py` file inside the submodule directories. The `-M` option lists the modules before the submodules.
  ```
  $ sphinx-apidoc -M -o docs/source/ utils/
  Creating file docs/source/arithmetic.rst.
  Creating file docs/source/operations.rst.
  Creating file docs/source/submodule.rst.
  Creating file docs/source/modules.rst.
  ```
- The `modules.rst` file includes a Table of Contents which lists all the python scripts in the `utils/` directory. Each python script leads to an individual `.rst` file for that script.

  ```
  utils
  =====

  .. toctree::
     :maxdepth: 4

     arithmetic
     operations
     submodule

  ```

- The generated `arithmetic.rst` file contains:
  ```
  .. automodule:: arithmetic
     :members:
     :undoc-members:
     :show-inheritance:
  ```
  For customization of the various `autodoc` commands, refer to its [documentation](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html). The `__init__` function is excluded by default, but can be added using the automodule option:
  ```
    :special-members: __init__
  ```
- In the `.rst` files, additional descriptive text can be directly added without any commands to give more context to the documentation. Images can also be added using the `image` command. Additional options for the command can be found [here](https://docutils.sourceforge.io/0.4/docs/ref/rst/directives.html#image).
  ```
  .. image:: images/image.png
     :alt: some useful image
     :width: 300px
  ```
- The `modules.rst` is then included in the parent `index.rst` using the `include` command.
  ```
  .. include:: modules.rst
  ```
- An alternative to using the `include` command is to add the names
  of the `.rst` files (without extension) to the toctree in `index.rst`. This method also has the benefit that each `.rst` file has an entry (with the title mentioned in its `.rst` file) in the Table of Contents with a hyperlink to the module's individual webpage.
  ```
  .. toctree::
      modules
     :maxdepth: 2
     :caption: Contents
  ```
- Including math expressions in the `.rst` files is handled by the extension `sphinx.ext.mathjax`. It has to be added to the extensions list in `conf.py`.
  - Syntax for inline math:
    ```
    :math:`x^2 + y^2 = z^2`
    ```
  - Symntax for math equations:
    ```
    .. math::
            x^2 + y^2 = z^2 \\
            x^2 + y^2 = z^2
    ```
- In order to include markdown files into the documentation, an extension named [`recommonmark`](https://recommonmark.readthedocs.io/en/latest/) is required. It has to be added to the extensions list in `conf.py`.
  ```
  pip install recommonmark
  ```
  Now, any markdown files from the `source/` directory can be added to the documentation by including the file in the toctree of `index.rst`.
  ```
  .. toctree::
     Instructions <../README.md>
     modules
     :maxdepth: 2
     :caption: Contents
  ```
- Jupyter Notebooks can be added using a Sphinx extension called [`nbsphinx`](https://nbsphinx.readthedocs.io/en/0.8.6/). An `.rst` file for the notebook can be created to import the contents and also add any descriptive text.

  Notebooks from the project directory / module directory could not be added using relative path. A solution has to be found. The easiest solution was to place the notebooks were placed inside the `docs/source/` directory.

  In the `.rst` file for the notebook, add the following to create a table of contents linking to the contents of the Jupyter notebook.

  ```
  .. toctree::
     :caption: Contents
     :maxdepth: 1

     notebooks/demo
  ```

  <p align="right">(<a href="#top">back to top</a>)</p>
  <!-- Build the documentation in HTML and/or Latex -->

### Step 5: Build the documentation in HTML and/or Latex

- From the `docs/` directory, the documentation can be built using the `make builder` command where the builder is either `html` or `latex` or `latexpdf`.
- For HTML, the options are automatically generated in `docs/source/conf.py` by the `sphinx-quickstart` command.
  ```
  make html
  ```
- LaTeX documentation can be automatically generated without any additions to the `conf.py` file. If customization is necessary, the Latex options can be added to the `conf.py` file. Example for LaTeX options for customizations can be found [here](https://www.sphinx-doc.org/en/master/latex.html).
  ```
  make latexpdf
  ```
- The generated HTML or Latex files can be found in `docs/build/html` or `docs/build/latex` respectively.
- The HTML documentation can be viewed locally via the `docs/build/html/index.html` file. The individual webpage can also be found as `.html` files.
- The Latex documentation can be viewed via the `docs/build/latex/[project_name].pdf` file.

 <p align="right">(<a href="#top">back to top</a>)</p>

#### For detailed explanation refer the [document](https://gitlab.scai.fraunhofer.de/ndv/research/automotive/cae_nlp/-/blob/ganesh/Sphinx%20Documentation/Sphinx_Documentation.pdf)
