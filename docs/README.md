# Building NeoRadium Documentation

0. See the DEMO DOCUMENTATION section in here: https://sphinx-rtd-theme.readthedocs.io/en/stable/demo/structure.html

1. Create a virtual environment in a location just outside the git folder:
```
python3 -m venv dve
source dve/bin/activate
pip install --upgrade pip setuptools
```
or use a conda environment.

2. Install sphinx and other dependencies:
```
pip install -U Sphinx sphinx_rtd_theme nbsphinx numpy scipy pyyaml matplotlib
```
"pandoc" also needs to be installed:
- brew install pandoc, or
- pip install pandoc, or
- conda install pandoc      (Use this if you are using a conda environment)

3. Then build the html documentation:
```
make clean html
```

The output html is generated in the `_build/html` folder. Open `_build/html/index.html` in your browser to view the locally generated documentation.
