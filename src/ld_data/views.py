from django.views import generic
from .models import Sim
from django.shortcuts import render
from .ontology_parser import parse_ontology


# Create your views here.
class IndexView(generic.ListView):
    template_name = 'general/index.html'
    cntxt_object_name = 'run_list'

    def get_queryset(self, **kwargs):
        """
        Return N number of runs for load case a,b,c which are deined as
        /a,b,c/10. EX: fo5,fpq,for/10
        """
        # no = self.kwargs.get('no')
        # testcode = self.kwargs.get('testcode').split(',')

        return Sim.nodes.all()[:2]

def ontology_report_view_manual(request):
    file_path = 'static/data/GAE-vehicle-safety-v1.0.rdf' 
    base_uri = "https://caewebvis.scai.fraunhofer.de/GAE//#"
    template_name = 'ld_data/ontology_GAE.html'

    cntxt = parse_ontology(file_path, base_uri)
    cntxt['classes_name'] = [x.split(base_uri)[-1] for x in cntxt['classes']]
    print(cntxt['classes_name'] )

    return render(request, template_name, cntxt)

def ontology_report_view(request):
    template_name = 'ontology/index.html'
    ontlogy_path = 'static/data/gae-ontlogy.html'

    return render(request, template_name)
