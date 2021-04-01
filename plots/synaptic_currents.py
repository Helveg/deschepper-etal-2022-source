import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from bsb.output import MorphologyRepository

import numpy as np
from bsb.plotting import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import h5py

import collections
from collections import defaultdict


from ._paths import *
from glob import glob
import selection

def plot(path=None):
    if path is None:
        path = glob(results_path("single_impulse", "*"))[0]
    duration = 18000
    cutoff = 8000
    ID = 126
    with h5py.File(path, "r") as f:
        synapses = {}
        for n, g in f[f"/recorders/synapses/{ID}/current"].items():
            if g.attrs["type"] not in synapses:
                synapses[g.attrs["type"]] = []
            synapses[g.attrs["type"]].append(np.array(g[()]))
        Vm = np.array(f[f"/recorders/soma_voltages/{ID}"])

    fig = make_subplots(rows=2, cols=1)

    fig.add_scatter(y=Vm, name='Vm', row=1, col=1)
    fig.update_xaxes(range=[cutoff, duration])

    for k, v in synapses.items():
        fig.add_trace(go.Scatter(y=sum(v), name=f"{len(v)} x {k}"), row=2, col=1)

    fig.update_xaxes(range=[cutoff, duration])

    return fig
