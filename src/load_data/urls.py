from django.urls import path
from . import views

app_name = 'load_data'

urlpatterns = [
    path(
        '',
        views.IndexView.as_view(), name='index'
        )]
