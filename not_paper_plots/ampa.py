from dbbs_models import GolgiCell
from dbbs_models.test import quick_test
import numpy as np
import pickle
from patch import p
import plotly.graph_objs as go

def generate_synaptic_response():
    gc = GolgiCell()
    syn = gc.create_synapse(gc.dendrites[0], "AMPA_MF")
    syn.stimulate(start=90, number=1, interval=10)
    r = syn.record()
    p.time
    quick_test(gc, duration=500)
    with open("synapse.pickle", "wb") as f:
        pickle.dump([np.array(list(p.time)), np.array(list(r))], f)
    fig = go.Figure(go.Scatter(x=list(p.time), y=list(r)))
    return fig

def plot():
    with open("synapse.pickle", "rb") as f:
        obj = pickle.load(f)
    
