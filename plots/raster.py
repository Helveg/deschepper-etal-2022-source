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
    cell_id = []
    return group.attrs.get("label", None) == "granule_cell" or name == cell_id

def plot():
    with h5py.File(results_path("results_NEST_stim_on_MFs_PoissFinal.hdf5"), "r") as f:
        print("items ",f["/recorders/soma_spikes"].items())
        groups = {k: v for k, v in filter(select_groups, f["/recorders/soma_spikes"].items())}
        print("groups ", groups)
        fig = hdf5_plot_spike_raster(groups, show=False, cutoff=300)
    return fig
