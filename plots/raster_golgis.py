import os, plotly.graph_objects as go
from bsb.core import from_hdf5
from bsb.plotting import hdf5_plot_psth, hdf5_plot_spike_raster
import numpy as np, h5py
from scipy import stats

def select_golgis(kv):
    name, group = kv
    return group.attrs.get("label", None) == "granule_cell"

from ._paths import *
from glob import glob
import selection

def plot(path=None, net_path=None):
    if path is None:
        path = glob(results_path("sensory_burst", "*"))[0]
    if net_path is None:
        net_path = network_path(selection.network)
    network = from_hdf5(net_path)
    figs = {}
    with h5py.File(path, "r") as f:
        groups = {k: v for k, v in filter(select_golgis, f["/recorders/soma_spikes"].items())}
        figs["poiss"] = hdf5_plot_spike_raster(groups, show=False, cutoff=0)
    return figs
