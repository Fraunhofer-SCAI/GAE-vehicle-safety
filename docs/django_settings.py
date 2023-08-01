

# INSTALLED_APPS with these apps is necessary for Sphinx to build
# without warnings & errors
# Depending on your package, the list of apps may be different

INSTALLED_APPS = [
    # internal added app
    'ld_data',
    # general
    "django.contrib.auth",
    "django.contrib.contenttypes",
]
