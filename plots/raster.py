import os, plotly.graph_objects as go
from bsb.core import from_hdf5
from bsb.plotting import hdf5_plot_psth, hdf5_plot_spike_raster
import numpy as np, h5py
from scipy import stats

network_path = os.path.join(
    os.path.dirname(__file__), "..", "networks", "results.hdf5"
)
def results_path(*args):
    return os.path.join(
        os.path.dirname(__file__), "..", "results", *args
    )

def select_groups(kv):
    name, group = kv
    return group.attrs.get("label", None) != "granule_cell" or np.random.rand() < 0.05

def plot():
    figs = {}
    with h5py.File(results_path("results_stim_on_MFs_Poiss.hdf5"), "r") as f:
        groups = {k: v for k, v in filter(select_groups, f["/recorders/soma_spikes"].items())}
        figs["poiss"] = hdf5_plot_spike_raster(groups, show=False, cutoff=300)
    with h5py.File(results_path("results_stim_on_MFs_4syncImp.hdf5"), "r") as f:
        groups = {k: v for k, v in filter(select_groups, f["/recorders/soma_spikes"].items())}
        figs["sync"] = hdf5_plot_spike_raster(groups, show=False, cutoff=300)
    return figs
