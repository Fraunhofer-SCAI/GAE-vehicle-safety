from django.views import generic
from .models import Sim


# Create your views here.
class IndexView(generic.ListView):
    template_name = 'general/index.html'
    context_object_name = 'run_list'

    def get_queryset(self, **kwargs):
        """
        Return N number of runs for load case a,b,c which are deined as
        /a,b,c/10. EX: fo5,fpq,for/10
        """
        # no = self.kwargs.get('no')
        # testcode = self.kwargs.get('testcode').split(',')

        return Sim.nodes.all()[:2]
