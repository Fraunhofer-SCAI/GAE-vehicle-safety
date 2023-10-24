# from django.urls import path, include

# urlpatterns = [
#     path('gae/load', include('ld_data.urls')),
#     path('gae/', include('gae.README.md')),
# ]


from django.contrib import admin
# from django.conf.urls import url
from django.urls import path, include, re_path

from django.views.generic import TemplateView

from django.conf import settings
from django.conf.urls.static import static

# Loading plotly Dash apps script
import nrg_fts.dash_app_code
import sim_rnk.dash_app_code
import ld_pth_detn.dash_app_code
import gae.views

#from django_plotly_dash.views import add_to_session

urlpatterns = [
    path('admin/', admin.site.urls),

    path('', include('django.contrib.auth.urls')),

    re_path('^nrg_fts_dash$', gae.views.dash_update, name="nrg_fts_dash"), # Energy features dashboard
    re_path('^sim_rnk_dash$', gae.views.dash_update, name="sim_rnk_dash"),  # SimRankTargeted dashboard
    re_path('^ld_pth_dtn$', gae.views.dash_update, name="ld_pth_dash"), # Load path detection dashboard
    re_path('^django_plotly_dash/', include('django_plotly_dash.urls')), 
    path('<str:dash>/', gae.views.dash_update, name='dash_update'),
    path('ld_data/', include('ld_data.urls')), 

    path('', TemplateView.as_view(template_name='home.html'), name='home'),
]
