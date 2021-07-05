import os, plotly.graph_objects as go, itertools
from bsb.core import from_hdf5
from bsb.plotting import hdf5_plot_psth, hdf5_plot_spike_raster
import numpy as np, h5py
from scipy import stats
from ._paths import *
from glob import glob
import selection


force_run_ids = False
order_datasets = True


class valueify:
    def __init__(self, iter):
        self.iter = iter

    def values(self):
        return self.iter


def plot(path=None, net_path=None):
    if path is None:
        path = results_path("balanced_sensory")
    if net_path is None:
        net_path = network_path(selection.network)
    network = from_hdf5(net_path)
    handles = [h5py.File(f, "a") for f in glob(os.path.join(path, "*.hdf5"))]
    order = dict(
        glomerulus=0,
        granule_cell=1,
        golgi_cell=2,
        purkinje_cell=3,
        stellate_cell=4,
        basket_cell=5,
    )
    try:
        for run_id, handle in enumerate(handles):
            if order_datasets:
                for ds in handle["/recorders/soma_spikes"].values():
                    ds.attrs["order"] = order.get(ds.attrs["label"], 6)
            for ds in handle["/recorders/soma_spikes"].values():
                if ds.attrs.get("run_id", None) is not None:
                    if not force_run_ids:
                        break
                ds.attrs["run_id"] = run_id
            else:
                continue
            print("Run ids already detected in:", handle.filename)
            print("Skipping run id insertion")
            break
        fig = hdf5_plot_psth(network, valueify(itertools.chain(*(handle["/recorders/soma_spikes"].values() for handle in handles))), show=False, cutoff=0, duration=5, gaps=False)
        fig.update_xaxes(range=[5500, 6500], tickmode="array", tickvals=list(i * 200 for i in range(40)), ticktext=list(str(i * 200 - 5500) for i in range(40)))
        ranges = [[0, 13], [0, 100], [0, 100], [0, 75], [0, 90]]
        for i, r in zip(range(1, 6), ranges):
            fig.update_yaxes(range=r, row=i, col=1)
        return fig
    finally:
        for handle in handles:
            handle.close()

def meta():
    return {"width": 1920 / 4 * 0.8702 * 1.0359 * 1.1645, "height": 1080 * 0.6054 * 1.0303 * 1.1509}
