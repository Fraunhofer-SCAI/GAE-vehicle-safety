from django.urls import path
from . import views

app_name = 'ld_data'

urlpatterns = [
    path('load/', views.LoadView, name='load'), # Load data to the database
    path('ontology-report/', views.ontology_report_view, name='ontology_report'),  # Ontology documentatin
]

