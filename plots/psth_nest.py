import os, plotly.graph_objects as go
from bsb.core import from_hdf5
from bsb.plotting import hdf5_plot_psth, hdf5_plot_spike_raster
import numpy as np, h5py
from scipy import stats
import plotly.io as pio
from glob import glob
import itertools
from ._paths import *
import selection

order = ["record_mossy_spikes", "record_glomerulus_spikes", "record_grc_spikes", "record_golgi_spikes", "record_pc_spikes", "record_basket_spikes", "record_stellate_spikes"]

class valueify:
    def __init__(self, iter):
        self.iter = iter

    def values(self):
        return self.iter

class FakeDataset:
    def __init__(self, h5set):
        self.arr = np.array(h5set[()])
        times = self.arr[:, 1]
        self.arr = self.arr[(times > 5500) & (times < 6500)] - 5500
        self.attrs = dict(h5set.attrs)

    def __getitem__(self, key):
        return self.arr[key]

    @property
    def shape(self):
        return self.arr.shape

def plot(path=None, net_path=None):
    if path is None:
        path = results_path("nest")
    if net_path is None:
        net_path = network_path(selection.network)
    network = from_hdf5(net_path)
    handles = [h5py.File(f, "a") for f in glob(os.path.join(path, "*.hdf5"))]
    try:
        for run_id, handle in enumerate(handles):
            for g in handle["/recorders/soma_spikes"].values():
                g.attrs["order"] = order.index(g.attrs["name"])
                g.attrs["run_id"] = run_id

        ranges = [[0, 15], [0, 15], [0, 10], [0, 65], [0, 120], [0, 155], [0, 172]]
        fig = hdf5_plot_psth(
            network,
            valueify(itertools.chain(*((FakeDataset(v) for v in handle["/recorders/soma_spikes"].values()) for handle in handles))),
            show=False,
            duration=5,
            gaps=False
        )
        for i in range(len(ranges)):
            fig.update_yaxes(range=ranges[i], row=i + 1, col=1)
        return fig
    finally:
        for handle in handles:
            handle.close()


def meta():
    return {"width": 800, "height": 450}
