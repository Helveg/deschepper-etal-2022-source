import os, plotly.graph_objects as go
from bsb.core import from_hdf5
from bsb.plotting import hdf5_plot_psth, hdf5_plot_spike_raster
import numpy as np, h5py
from scipy import stats
import selection
from ._paths import *
from glob import glob

def select_groups(kv):
    name, group = kv
    return group.attrs.get("label", None) != "granule_cell" or np.random.rand() < 0.1

def plot(path=None):
    if path is None:
        path = glob(results_path("sensory_burst", "*.hdf5"))[1]
    with h5py.File(path, "r") as f:
        groups = {k: v for k, v in filter(select_groups, f["/recorders/soma_spikes"].items())}
        fig = hdf5_plot_spike_raster(groups, show=False, cutoff=5500, cell_type_sort=lambda x, y: list(reversed(["granule_cell", "golgi_cell", "purkinje_cell", "basket_cell", "stellate_cell"])))
        fig.update_layout(xaxis_range= [0, 1000]) #, yaxis_range=[0, 40])
    return fig
