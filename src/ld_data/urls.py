from django.urls import path
from . import views

app_name = 'ld_data'

urlpatterns = [
    path(
        '',
        views.IndexView.as_view(), name='index'
        )]
