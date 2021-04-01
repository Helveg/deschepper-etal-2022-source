import os, plotly.graph_objects as go
from bsb.core import from_hdf5
from bsb.plotting import hdf5_plot_psth, hdf5_plot_spike_raster
import numpy as np, h5py
from scipy import stats
import random

# def select_groups(kv):
#     name, group = kv
#     cell_id = []
#     print("attrssssssssssssssssssss    ",group.attrs.get("label", None))
#     return group.attrs.get("label", None) == "record_sc_spikes" or \
#     group.attrs.get("label", None) == "record_bc_spikes" or \
#     group.attrs.get("label", None) == "record_glomerulus_spikes" or \
#     group.attrs.get("label", None) == "record_pc_spikes" or \
#     group.attrs.get("label", None) == "record_golgi_spikes" or name == cell_id

ordered_labels = ["record_bc_spikes","record_sc_spikes","record_pc_spikes","record_golgi_spikes", "record_granules_spikes"]
cell_ids = {
    "record_granules_spikes": [16011, 3083, 15265, 3075, 11800, 3068, 9623, 3069, 31681, 3070],
    # Poiss flipped
    # [17372, 5987, 15288, 3764, 11399, 3083, 9163, 3074, 31681, 3070]
    # Poiss
    # [3070, 31681, 3074, 9163, 3083, 11399, 3764, 15288, 5987, 17372]
    # 4 sync
    # [3070, 31681, 3069, 9623, 3068, 11800, 3075, 15265, 3083, 16011]
    # 4 sync flipped
    # [16011, 3083, 15265, 3075, 11800, 3068, 9623, 3069, 31681, 3070]
    "record_glomerulus_spikes": random.sample(range(732,2336), int((2336-732)*0.05))
}

def cell_type_sorter(x, y):
    return ordered_labels

def cell_sorter(label, ids):
    if label in cell_ids:
        ids = cell_ids[label]
        return dict(zip(ids, range(len(ids))))

def select_data(kv):
    # Unpack the `.items()` key value pairs
    name, group = kv
    # Create a fake dataset that we can filter and pass on into the raster plot function.
    group = FakeDataset(group)
    label = group.attrs.get("label", None)
    # If a cell id filter exists for this label filter the fake dataset's spikes
    if label in cell_ids:
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

from ._paths import *
from glob import glob
import selection

def plot(path=None):
    return go.Figure(layout=dict(title_text="Alice has to provide the NEST result files"))
    if path is None:
        path = glob(results_path("sensory_burst", "*"))[0]
    with h5py.File(results_path("nest_PoissFinal2/results_NEST_stim_on_MFs_PoissFinal2_finetuning0-02.hdf5"), "r") as f:
        groups = {k: v for k, v in map(select_ids, f["/recorders/soma_spikes"].items(), generate(sel_labels), generate(cell_ids))}
        fig = hdf5_plot_spike_raster(groups, show=False, cutoff=300, sorted_labels=sel_labels, sorted_ids=cell_ids)
    with h5py.File(results_path("results_NEST_stim_on_MFs_4syncFinal2_finetuning0-02.hdf5"), "r") as f:
        print("items ",f["/recorders/soma_spikes"].items())
        #groups = {k: v for k, v in filter(select_groups, f["/recorders/soma_spikes"].items())}
        print("groups:", *(f["/recorders/soma_spikes"].items()))
        groups = {k: v for k, v in map(select_ids, f["/recorders/soma_spikes"].items(), generate(sel_labels), generate(cell_ids))}
    #    print("groups sel", groups[""])
        #fig = hdf5_plot_spike_raster(groups, show=False)
        fig = hdf5_plot_spike_raster(groups, show=False, cutoff=300, \
        sorted_labels=sel_labels, sorted_ids=cell_ids)
        #fig.update_yaxes(range=[-5, 30])

    return fig


def meta():
    return {"width": 1920 / 2}
