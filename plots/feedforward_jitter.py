from bsb.core import from_hdf5
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import selection, grc_cloud
from colour import Color
import numpy as np, h5py, pickle
from glob import glob
from bsb.plotting import plot_morphology
from ._paths import *
import os

import plotly
colors = plotly.colors.DEFAULT_PLOTLY_COLORS

frozen = False

def distr(cont, sp):
    print("M", sp)
    print("C", sp[0:1], sp[1:2], sp[2:3], sp[3:])
    cont[0].extend(sp[0:1])
    cont[1].extend(sp[1:2])
    cont[2].extend(sp[2:3])
    cont["3+"].extend(sp[3:])

def plot(net_path=None, wt_path=None, ko_path=None):
    if net_path is None:
        net_path = network_path(selection.network)
    if wt_path is None:
        wt_path = results_path("single_impulse", "sensory_burst", "single_impulse_all.hdf5")
    if ko_path is None:
        ko_path = results_path("single_impulse", "sensory_burst", "single_impulse_all_nomli.hdf5")
    network = from_hdf5(net_path)
    cells = selection.onbeam["purkinje_cell"]
    fig = make_subplots(rows=2, cols=2, subplot_titles=("All WT", "Onbeam WT", "All NoMLI", "Onbeam NoMLI"))
    for row, (lbl, fp) in enumerate((("wt", wt_path), ("ko", ko_path))):
        row += 1
        spikez = {0: [], 1: [], 2: [], "3+": []}
        beamspikez = {0: [], 1: [], 2: [], "3+": []}
        with h5py.File(fp, "r") as f:
            for k, v in f[f"recorders/soma_spikes/"].items():
                id = int(k)
                if v.attrs["label"] == "purkinje_cell":
                    spikes = v[:, 1]
                    spikes = spikes[(spikes > 6000) & (spikes < 6200)]
                    distr(spikez, spikes)
                    if id in cells:
                        distr(beamspikez, spikes)
        for i, (k, v) in enumerate(spikez.items()):
            fig.add_trace(
                go.Histogram(x=v, name=f"{k} spikes", nbinsx=200, marker_line_width=0, marker_color=colors[i], showlegend=row == 1),
                row=row,
                col=1
            )
        for i, (k, v) in enumerate(beamspikez.items()):
            fig.add_trace(
                go.Histogram(x=v, name=f"{k} spikes", nbinsx=200, marker_line_width=0, marker_color=colors[i], showlegend=False),
                row=row,
                col=2
            )
    fig.update_layout(barmode="overlay", bargap=0, bargroupgap=0)
    fig.update_traces(opacity=0.75)
    fig.update_xaxes(range=[6000, 6200])
    return fig
