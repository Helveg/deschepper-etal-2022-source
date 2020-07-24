import os, plotly.graph_objects as go
from scaffold.core import from_hdf5
from scaffold.plotting import hdf5_plot_spike_raster
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
    with h5py.File(results_path("first_run_300_200.hdf5"), "r") as f:
        fig = hdf5_plot_spike_raster(f["/purkinjes"], show=False)
    return fig
