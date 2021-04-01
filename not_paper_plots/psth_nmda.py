import os, plotly.graph_objects as go, itertools
from bsb.core import from_hdf5
from bsb.plotting import hdf5_plot_psth, hdf5_plot_spike_raster
import numpy as np, h5py
from scipy import stats
from glob import glob


class valueify:
    def __init__(self, iter):
        self.iter = iter

    def values(self):
        return self.iter


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
        figs["grc_NMDA"] = hdf5_plot_psth(network, {k: v for k, v in f["/recorders/soma_spikes"].items() if v.attrs["label"] == "granule_cell"}, show=False, cutoff=300, duration=2)
        figs["grc_NMDA"].update_xaxes(range=[375, 500])
        figs["grc_NMDA"].update_yaxes(range=[0, 75])
    with h5py.File(results_path("results_stim_on_MFs_4syncImp_noNMDAglomgrc.hdf5"), "a") as f:
        figs["grc_no_NMDA"] = hdf5_plot_psth(network, {k: v for k, v in f["/all"].items() if v.attrs["label"] == "granule_cell"}, show=False, cutoff=300, duration=2)
        figs["grc_no_NMDA"].update_xaxes(range=[375, 500])
        figs["grc_no_NMDA"].update_yaxes(range=[0, 75])
    with h5py.File(results_path("results_stim_on_MFs_4syncImp.hdf5"), "a") as f:
        figs["goc_NMDA"] = hdf5_plot_psth(network, {k: v for k, v in f["/all"].items() if v.attrs["label"] == "golgi_cell"}, show=False, cutoff=300, duration=2)
        figs["goc_NMDA"].update_xaxes(range=[375, 500])
        figs["goc_NMDA"].update_yaxes(range=[0, 130])
    with h5py.File(results_path("results_stim_on_MFs_4syncImp_noNMDAglomgoc_noNMDAaagoc.hdf5"), "a") as f:
        figs["goc_no_NMDA"] = hdf5_plot_psth(network, {k: v for k, v in f["/all"].items() if v.attrs["label"] == "golgi_cell"}, show=False, cutoff=300, duration=2)
        figs["goc_no_NMDA"].update_xaxes(range=[375, 500])
        figs["goc_no_NMDA"].update_yaxes(range=[0, 130])
    return figs

def meta(key):
    return {"width": 1920 / 4, "height": 1080 / 3}
