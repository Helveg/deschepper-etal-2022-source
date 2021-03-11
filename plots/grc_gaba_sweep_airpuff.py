from dbbs_models import GranuleCell
from dbbs_models.test import quick_test
import numpy as np
from patch import p as patch
import plotly.graph_objs as go

def plot():
    gcs = []
    for i in range(20):
        gc = GranuleCell()
        hz = i * 5
        for j in range(1):
            r = np.random.randn(5)
            p = [r[0] + 103, r[1] * 2 + 106, r[2] * 2 + 109, r[3] * 10 + 125, r[4] * 10 + 145]
            syn = gc.create_synapse(gc.dendrites[j], "AMPA")
            for t in p:
                syn.stimulate(start=t, number=1, interval=1)
            syn = gc.create_synapse(gc.dendrites[j], "NMDA")
            for t in p:
                syn.stimulate(start=t, number=1, interval=1)
        for k in range(3):
            syn = gc.create_synapse(gc.dendrites[k], "GABA")
            syn.stimulate(start=0, number=1000 if hz else 0, interval=1000/max(hz, 0.001), noise=True)
        gcs.append(gc)
    patch.time
    quick_test(*gcs)
    for i in range(20):
        go.Figure(go.Scatter(x=list(patch.time), y=list(gcs[i].Vm), name=f"{i*5}Hz")).show()
