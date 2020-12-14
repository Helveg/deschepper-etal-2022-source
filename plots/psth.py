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
    figs = {}
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
        figs["poiss"] = hdf5_plot_psth(network, valueify(itertools.chain(*(handle["/all"].values() for handle in handles))), show=False, cutoff=300, duration=5)
        ranges = [[0, 10], [0, 10], [0, 65], [0, 75], [0, 60], [0, 60]]
        for i in range(len(ranges)):
            figs["poiss"].update_yaxes(range=ranges[i], row=i + 1, col=1)
        with h5py.File(results_path("results_stim_on_MFs_4syncImp.hdf5"), "a") as f:
            reorder = {"granule_cell": 0, "mossy_fiber": -1}
            for g in f["/all"].values():
                label = g.attrs.get("label", None)
                if label in reorder:
                    if g.attrs.get("order", None) == reorder[label]:
                        print("Already reordered, continuing")
                        break
                    g.attrs["order"] = reorder[label]
            figs["sync"] = hdf5_plot_psth(network, f["/all"], show=False, cutoff=300, duration=5)
            ranges = [[0, 30], [0, 27], [0, 125], [0, 130], [0, 120], [0, 120]]
            for i in range(len(ranges)):
                figs["sync"].update_yaxes(range=ranges[i], row=i + 1, col=1)
        return figs
    finally:
        for handle in handles:
            handle.close()

def meta(key):
    return {"width": 1920 / 2}
