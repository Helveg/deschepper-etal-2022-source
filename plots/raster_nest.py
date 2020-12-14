import os, plotly.graph_objects as go
from bsb.core import from_hdf5
from bsb.plotting import hdf5_plot_psth, hdf5_plot_spike_raster
import numpy as np, h5py
from scipy import stats
import random

network_path = os.path.join(
    os.path.dirname(__file__), "..", "networks", "results.hdf5"
)
def results_path(*args):
    return os.path.join(
        os.path.dirname(__file__), "..", "results", *args
    )

# def select_groups(kv):
#     name, group = kv
#     cell_id = []
#     print("attrssssssssssssssssssss    ",group.attrs.get("label", None))
#     return group.attrs.get("label", None) == "record_sc_spikes" or \
#     group.attrs.get("label", None) == "record_bc_spikes" or \
#     group.attrs.get("label", None) == "record_glomerulus_spikes" or \
#     group.attrs.get("label", None) == "record_pc_spikes" or \
#     group.attrs.get("label", None) == "record_golgi_spikes" or name == cell_id

def select_ids(kv):
    sel_labels = ["record_granules_spikes", "record_glomerulus_spikes"]
    cell_ids = {"record_granules_spikes":  [16011, 3083, 15265, 3075, 11800, 3068, 9623, 3069, 31681, 3070],
    #[3070, 31681, 3074, 9163, 3083, 11399, 3764, 15288, 5987, 17372],\    Poiss
    #[3070, 31681, 3069, 9623, 3068, 11800, 3075, 15265, 3083, 16011]        4sync
    #[16011, 3083, 15265, 3075, 11800, 3068, 9623, 3069, 31681, 3070]   4sync flipped
    "record_glomerulus_spikes": random.sample(range(732,2336), int((2336-732)*0.05))}

    name, group = kv
    group = FakeDataset(group)
    # if ids within current label dataset
    for label in sel_labels:
        if group.attrs.get("label", None) == label:
            group.arr = group.arr[np.isin(group.arr[:, 0], cell_ids[label])]

    return name, group

class FakeDataset:
    def __init__(self, h5set):
        self.arr = np.array(h5set[()])
        self.attrs = dict(h5set.attrs)

    def __getitem__(self, key):
        return self.arr[key]

    @property
    def shape(self):
        return self.arr.shape

def plot():
    with h5py.File(results_path("results_stim_on_MFs_10Hzdrive.hdf5"), "r") as f:
        print("items ",f["/recorders/soma_spikes"].items())
        #groups = {k: v for k, v in filter(select_groups, f["/recorders/soma_spikes"].items())}
        groups = {k: v for k, v in map(select_ids, f["/recorders/soma_spikes"].items())}
        #fig = hdf5_plot_spike_raster(groups, show=False)
        fig = hdf5_plot_spike_raster(groups, show=False, cutoff=300, \
        sorted_labels=[ "record_golgi_spikes", "record_granules_spikes"], sorted_ids={"record_granules_spikes": [16011, 3083, 15265, 3075, 11800, 3068, 9623, 3069, 31681, 3070]})
        #sorted_labels=["record_bc_spikes", "record_sc_spikes", "record_pc_spikes", "record_golgi_spikes", "record_granules_spikes"], sorted_ids={"record_granules_spikes": [16011, 3083, 15265, 3075, 11800, 3068, 9623, 3069, 31681, 3070]})

    return fig
