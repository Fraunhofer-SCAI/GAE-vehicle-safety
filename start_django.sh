#!/bin/bash

cd ..

# Activate installed python
eval "$("py-3.8/bin/conda" shell.bash hook)"

cd GAE-vehicle-safety/src/
python manage.py runserver
