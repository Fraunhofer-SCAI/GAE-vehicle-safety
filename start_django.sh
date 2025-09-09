#!/bin/bash

cd ..

# Use the installed python
pypath=$(pwd)/"py-3.8"
export PATH="$pypath/bin":$PATH

cd GAE-vehicle-safety/src/
python manage.py runserver
