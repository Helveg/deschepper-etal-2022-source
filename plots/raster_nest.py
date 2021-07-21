import os, plotly.graph_objects as go
from bsb.core import from_hdf5
from bsb.plotting import hdf5_plot_psth, hdf5_plot_spike_raster
import numpy as np, h5py
from scipy import stats
import random

cell_map = {
    "record_mossy_spikes": "mossy_fibers",
    "record_glomerulus_spikes": "glomerulus",
    "record_grc_spikes": "granule_cell",
    "record_basket_spikes": "basket_cell",
    "record_stellate_spikes": "stellate_cell",
    "record_golgi_spikes": "golgi_cell",
    "record_pc_spikes": "purkinje_cell",
}

label_map = {
    "record_mossy_spikes": "MF",
    "record_glomerulus_spikes": "Glom",
    "record_grc_spikes": "Grc",
    "record_basket_spikes": "BC",
    "record_stellate_spikes": "SC",
    "record_golgi_spikes": "GoC",
    "record_pc_spikes": "PC",
}

excluded = set(["record_glomerulus_spikes", "record_mossy_spikes"])

subsampler = {
    "record_grc_spikes": 0.1,
    "record_glomerulus_spikes": 0.2,
}

order = list(reversed(["record_mossy_spikes", "record_glomerulus_spikes", "record_grc_spikes", "record_golgi_spikes", "record_pc_spikes", "record_basket_spikes", "record_stellate_spikes"]))

class FakeDataset:
    def __init__(self, network, h5set):
        sub = subsampler.get(h5set.attrs["label"])
        if sub is not None:
            data = h5set[()]
            ids = data[:,0]
            subsample = (np.max(ids) - np.min(ids)) * sub + np.min(ids)
            self.arr = data[ids < subsample,:]
        else:
            self.arr = np.array(h5set[()])
        times = self.arr[:, 1]
        self.arr = self.arr[(times > 5500) & (times < 6500)] - 5500
        self.attrs = dict(h5set.attrs)
        self.attrs["color"] = network.configuration.cell_types[cell_map[self.attrs["label"]]].plotting.color
        self.attrs["label"] = label_map[self.attrs["label"]]

    def __getitem__(self, key):
        return self.arr[key]

    @property
    def shape(self):
        return self.arr.shape

from ._paths import *
from glob import glob
import selection

def plot(path=None, net_path=None):
    if path is None:
        path = glob(results_path("nest", "*.hdf5"))[0]
    if net_path is None:
        net_path = network_path(selection.network)
    network = from_hdf5(net_path)
    with h5py.File(path, "r") as f:
        groups = {k: FakeDataset(network, v) for k, v in sorted(f["/recorders/soma_spikes"].items(),key=lambda kv: order.index(kv[0])) if v.attrs["label"] not in excluded}
        fig = hdf5_plot_spike_raster(groups, show=False, cell_type_sort=lambda x, y: x)

    return fig

def meta():
    return {"width": 800, "height": 450}
