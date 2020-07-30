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


def plot():
    with h5py.File(results_path("150Hz/stacked_results_150.hdf5"), "r") as f:
        fig = hdf5_plot_psth(f["stacked_psth"], show=False, cutoff=400, duration=10)
        # fig = hdf5_plot_spike_raster(f["/recorders/soma_spikes"], show=False)
    return fig
