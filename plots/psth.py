import os, plotly.graph_objects as go, itertools
from bsb.core import from_hdf5
from bsb.plotting import hdf5_plot_psth, hdf5_plot_spike_raster
import numpy as np, h5py
from scipy import stats
from ._paths import *
from glob import glob
import selection


force_run_ids = False


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
    try:
        for run_id, handle in enumerate(handles):
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
        # ranges = [[0, 10], [0, 10], [0, 65], [0, 75], [0, 60], [0, 60]]
        return fig
    finally:
        for handle in handles:
            handle.close()

def meta():
    return {"width": 1920 / 4}
