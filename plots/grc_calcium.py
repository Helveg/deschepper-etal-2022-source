from bsb.core import from_hdf5
import os, sys, h5py, numpy as np
import selection, plotly.graph_objs as go, scipy.stats
from ._paths import *
from glob import glob
import selection, random

frozen = False

def crop(data, min, max, indices=False):
    c = data[:, 1]
    if indices:
        return np.where((c > min) & (c < max))[0]
    return c[(c > min) & (c < max)]

def plot(path=None, net_path=None, stim_start=5500, stim_end=6500):
    if path is None:
        path = results_path("results_grc_calcium.hdf5")
    if net_path is None:
        net_path = network_path(selection.network)
    network = from_hdf5(net_path)
    ids = network.get_placement_set("granule_cell").identifiers

    with h5py.File(path, "r") as f:
        fig = go.Figure(
            [
                go.Scatter(
                    mode="lines",
                    y=f[f"recorders/ions/ca/{id}/concentration/1"][()]
                )
                for id in random.choices(ids, k=300) if str(id) in f[f"recorders/ions/ca/"]
            ] + [
                go.Scatter(
                    mode="lines",
                    y=f[f"recorders/ions/ca/10001/concentration/1"][()]
                )
            ]
        )
    return fig
