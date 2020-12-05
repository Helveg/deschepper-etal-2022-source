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


def plot():
    poisson_path = results_path("5poiss")
    network = from_hdf5(network_path)
    handles = [h5py.File(f, "a") for f in glob(os.path.join(poisson_path, "*.hdf5"))]
    try:
        for run_id, handle in enumerate(handles):
            for ds in handle["/all"].values():
                if ds.attrs.get("run_id", None) is not None:
                    break
                ds.attrs["run_id"] = run_id
            else:
                continue
            print("Run ids already detected in:", handle.filename)
            print("Skipping run id insertion")
            break
        figs = {}
        figs["poiss"] = hdf5_plot_psth(network, valueify(itertools.chain(*(handle["/all"].values() for handle in handles))), show=False, cutoff=400, duration=5)
        with h5py.File(results_path("results_stim_on_MFs_4syncImp.hdf5"), "r") as f:
            figs["sync"] = hdf5_plot_psth(network, f["recorders/soma_spikes"], show=False, cutoff=400, duration=5)
        return figs
    finally:
        for handle in handles:
            handle.close()
