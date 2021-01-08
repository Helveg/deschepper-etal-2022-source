import os, plotly.graph_objects as go
from bsb.core import from_hdf5
from bsb.plotting import hdf5_plot_psth, hdf5_plot_spike_raster
import numpy as np, h5py
from scipy import stats
import random

network_path = os.path.join(
    os.path.dirname(__file__), ".", "networks", "results.hdf5"
)
def results_path(*args):
    return os.path.join(
        os.path.dirname(__file__), ".", "results", *args
    )

cutoff = 300
window_burst = [400, 475]
# def select_groups(kv):
#     name, group = kv
#     cell_id = []
#     print("attrssssssssssssssssssss    ",group.attrs.get("label", None))
#     return group.attrs.get("label", None) == "record_sc_spikes" or \
#     group.attrs.get("label", None) == "record_bc_spikes" or \
#     group.attrs.get("label", None) == "record_glomerulus_spikes" or \
#     group.attrs.get("label", None) == "record_pc_spikes" or \
#     group.attrs.get("label", None) == "record_golgi_spikes" or name == cell_id

sel_labels = ["record_bc_spikes","record_sc_spikes","record_pc_spikes","record_golgi_spikes", "record_granules_spikes"]
#sel_labels = ["record_granules_spikes"]

cell_ids = {"record_granules_spikes": [16011, 3083, 15265, 3075, 11800, 3068, 9623, 3069, 31681, 3070],
#[17372, 5987, 15288, 3764, 11399, 3083, 9163, 3074, 31681, 3070],    # Poiss flipped
#[3070, 31681, 3074, 9163, 3083, 11399, 3764, 15288, 5987, 17372],\    Poiss
#[3070, 31681, 3069, 9623, 3068, 11800, 3075, 15265, 3083, 16011]        4sync
#[16011, 3083, 15265, 3075, 11800, 3068, 9623, 3069, 31681, 3070]   4sync flipped
"record_glomerulus_spikes": random.sample(range(732,2336), int((2336-732)*0.05))}

def generate(arg):
    def generator():
        while True:
            yield arg

    return generator()

def select_ids(kv, sel_labels, cell_ids):

    name, group = kv
    group = FakeDataset(group)
    for label in sel_labels:
        print(name, group.attrs.get("label"))
        if label not in cell_ids:
            continue
        elif group.attrs.get("label", None) == label:
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


with h5py.File(results_path("results_NEST_stim_on_MFs_4syncFinal2_finetuning0-02_noMLIPC.hdf5"), "r") as f:
    print("items ",f["/recorders/soma_spikes"].items())
    #groups = {k: v for k, v in filter(select_groups, f["/recorders/soma_spikes"].items())}
    print("groups:", *(f["/recorders/soma_spikes"].items()))
    groups = {k: v for k, v in map(select_ids, f["/recorders/soma_spikes"].items(), generate(sel_labels), generate(cell_ids))}
#    print("groups sel", groups[""])
    #fig = hdf5_plot_spike_raster(groups, show=False)
    x_labelled = {}
    y_labelled = {}
    colors = {}
    ids = {}
    for cell_id, dataset in groups.items():
        attrs = dict(dataset.attrs)
        if len(dataset.shape) == 1 or dataset.shape[1] == 1:
            times = dataset[()] - cutoff
            set_ids = np.ones(len(times)) * int(
                attrs.get("cell_id", attrs.get("cell", cell_id))
            )
        else:
            times = dataset[:, 1] - cutoff
            set_ids = dataset[:, 0]
        label = attrs.get("label", "unlabelled")
        if not label in x_labelled:
            x_labelled[label] = []
        if not label in y_labelled:
            y_labelled[label] = []
        if not label in colors:
            colors[label] = attrs.get("color", "black")
        if not label in ids:
            ids[label] = 0
        ids[label] += 1
        # Add the spike timings on the X axis.
        x_labelled[label].extend(times)
        # Set the cell id for the Y axis of each added spike timing.
        y_labelled[label].extend(set_ids)
    times = x_labelled
    neurons = y_labelled
    pc_times = times['record_pc_spikes']
    pc_ids = np.unique(neurons['record_pc_spikes'])
    print("pc spikes",pc_times,pc_ids)
    spike_burst = {}
    pause = {}
    for p in pc_ids:        # For each PC we compute number of spikes in window burst and pause duration
        current_pc_spikes_ids = np.where(neurons['record_pc_spikes'] == p)
        current_pc_spikes = [pc_times[i] for i in list(current_pc_spikes_ids[0])]
        spike_burst[p] = sum(map(lambda x : x>window_burst[0] and x<window_burst[1], current_pc_spikes))
        print("p", p, " spikes ",current_pc_spikes_ids,current_pc_spikes, " in burst window ", spike_burst[p])
        ind = np.where(np.array(current_pc_spikes)>window_burst[1])
        print("p pause ind",ind, " first ", ind[0][0])
        pause[p] = current_pc_spikes[ind[0][0]] - window_burst[1]
        print("pause ",pause[p])

    fig = dict({
        "data": [{"type": "bar",
                  "x": pause,
                  "y": spike_burst}],
        "layout": {"title": {"text": "A Figure Specified By Python Dictionary"}}
    })

    # To display the figure defined by this dict, use the low-level plotly.io.show function
    import plotly.io as pio

    pio.show(fig)
    # fig = hdf5_plot_spike_raster(groups, show=False, cutoff=300, \
    # sorted_labels=sel_labels, sorted_ids=cell_ids)
    #fig.update_yaxes(range=[-5, 30])
