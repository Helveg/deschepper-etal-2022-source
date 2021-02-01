    # filename = '/home/claudia/deschepper-etal-2020/networks/300x_200z.hdf5'
    # scaffoldInstance = from_hdf5(filename)
    # config = scaffoldInstance.configuration
import os, plotly.graph_objects as go, itertools
from bsb.core import from_hdf5
from bsb.plotting import hdf5_plot_psth, hdf5_plot_spike_raster
import numpy as np, h5py
from scipy import stats
from glob import glob


network_path = os.path.join(
    os.path.dirname(__file__), "..", "networks", "300x_200z.hdf5"
)
def results_path(*args):
    return os.path.join(
        os.path.dirname(__file__), "..", "results", *args
    )


class valueify:
    def __init__(self, iter):
        self.iter = iter

    def values(self):
        return self.iter

network = from_hdf5(network_path)
config = network.configuration

def plot():
    figs = {}
    order=dict(mossy_fiber=0, granule_cell=1, golgi_cell=2, purkinje_cell=3, stellate_cell=4, basket_cell=5)
    color=dict(
        mossy_fiber=config.cell_types["glomerulus"].plotting.color,
        granule_cell=config.cell_types["granule_cell"].plotting.color,
        golgi_cell=config.cell_types["golgi_cell"].plotting.color,
        purkinje_cell=config.cell_types["purkinje_cell"].plotting.color,
        stellate_cell=config.cell_types["stellate_cell"].plotting.color,
        basket_cell=config.cell_types["basket_cell"].plotting.color
    )
    with h5py.File(results_path("results_365b0_poiss.hdf5"), "a") as f:
        for g in f["/recorders/soma_spikes"].values():
        # for g in f["/all"].values():
            if g.attrs["label"] not in order:
                print("Not sorting", g.name, "no order found")
            g.attrs["order"] = order.get(g.attrs["label"], 0)
            g.attrs['color'] = color.get(g.attrs["label"], 0)
        figs["poiss"] = hdf5_plot_psth(network, f["/recorders/soma_spikes"], show=False, cutoff=0, duration=5)


    with h5py.File(results_path("results_365b0_sync.hdf5"), "a") as f:
        for g in f["/recorders/soma_spikes"].values():
        # for g in f["/all"].values():
            if g.attrs["label"] not in order:
                print("Not sorting", g.name, "no order found")
            g.attrs["order"] = order.get(g.attrs["label"], 0)
            g.attrs['color'] = color.get(g.attrs["label"], 0)
        figs["sync"] = hdf5_plot_psth(network, f["/recorders/soma_spikes"], show=False, cutoff=0, duration=5)
    return figs
