# Developer Guide

## SPHINX reporting

local debugging, run the build,
```
sphinx-build -b html docs docs/_build
```
to recover all old builds:
```
sphinx-build -b html -E -a docs docs/_build
```

run the server to view the files:
```
python -m http.server -d docs/_build
```
on push the same process is done with gg-pages that is defined in githubworkflows: `workflows/documentation.yaml`.