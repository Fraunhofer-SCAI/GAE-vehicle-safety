from django.views import generic
from .models import Sim
from django.shortcuts import render
from .ontology_parser import parse_ontology
from django.contrib import messages
import oems
import glob 
from .functions import CaeSim, PopulateSim
from gae.constants import OEM_NAME, IE_UNIT, TIME_UNIT, DATA_PATH, SIM_PATTERN

# Create your views here.
def LoadView(request):
    template_name = 'ld_data/load.html'

    oem = oems.oems(OEM_NAME, IE_UNIT, TIME_UNIT, DATA_PATH, SIM_PATTERN)
    oem.backend_server()
    sims = glob.glob(oem.data_path)


    ''' load or update all CAE simulations'''
    msg = []
    for s in sims:
        # break
        msg.append(s.split(DATA_PATH)[-1])
        sim = CaeSim(OEM_NAME)
        sim.dataOEM(s)
        PS = PopulateSim(sim)
        PS.populate_sim()
        print(s)
        # break
        # update_part(sim, pids_box, rng_box, cog_box)

    # connect nodes
    PS.cnct_nodes('IE_ti_tn', lc='')

    msg = ', '.join(msg)

    messages.success(request, 'Loaded Simulations:')
    messages.success(request, msg)
    return render(request, template_name)


def ontology_report_view(request):
    template_name = 'ontology/index.html'
    ontlogy_path = 'static/data/gae-ontlogy.html'

    return render(request, template_name)
