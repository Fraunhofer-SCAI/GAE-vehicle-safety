  cd /home/apakiman/Projects/pyLODE

  moved to:
  /var/www/caewebvis/cae_web_fe/build/GAE

  
  182  git clone https://github.com/RDFLib/pyLODE.git
  183  ll
  184  cd pyLODE/
  185  ll
  186  python setup.py
  187  cd pylode/
  188  ll
  189  python cli.py ~/Projects/GAE-vehicle-safety/ontology/GAE-vehicle-safety-v1.0.owl
  190   pip install dominate
  191  cd ..
  192  python setup.py
  193  ll
  194  pip install -r requirements.txt
  195  pip install pytest-cov=2.0
  196  pip install pytest-cov
  197  pip install pytest-filter-subpackage
  198  python setup.py
  199  pip install -r requirements.txt
  200  python setup.py
  201  ll
  202  cd pylode/
  203  ll
  204  history
  205  python cli.py ~/Projects/GAE-vehicle-safety/ontology/GAE-vehicle-safety-v1.0.owl
