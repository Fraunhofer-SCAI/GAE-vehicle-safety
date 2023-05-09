from django.urls import path, include

urlpatterns = [
    path('gae/', include('load_data.urls'))
]
