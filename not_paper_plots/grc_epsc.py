from dbbs_models import GranuleCell
from dbbs_models.test import quick_test
from patch import p

def dendritic_synapse(cell, type, i=0):
    syn = cell.create_synapse(cell.dendrites[i], type)
    syn.stimulate(start=100, number=3, interval=2)
    syn.stimulate(start=120, number=2, interval=15)
    return syn.record()

def create_recording(GABA=False):
    gc = GranuleCell()
    recorders = {"Vm": gc.record_soma()}
    recorders["AMPA"] = dendritic_synapse(gc, "AMPA")
    recorders["NMDA"] = dendritic_synapse(gc, "NMDA")
    if GABA:
        recorders["GABA"] = dendritic_synapse(gc, "GABA")
    return (gc, recorders)

def plot():
    from plotly import graph_objs as go

    gc, r1 = create_recording()
    gc_gaba, r2 = create_recording(GABA=True)
    t = p.time
    quick_test(gc, gc_gaba, duration=300)
    figs = {
        "noGABA": go.Figure([go.Scatter(x=list(t), y=list(r), name=k) for k, r in r1.items()]),
        "GABA": go.Figure([go.Scatter(x=list(t), y=list(r), name=k) for k, r in r2.items()]),
    }
    return figs
