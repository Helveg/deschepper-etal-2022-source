import os, plotly.graph_objects as go
from plotly.subplots import make_subplots
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
window_burst = [400, 500]           # Time window where we analyse the burst (number of spikes or mean ISI)

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


with h5py.File(results_path("results_NEST_stim_on_MFs_1syncFinal2_finetuning0-02.hdf5"), "r") as f:
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
    pc_times = times['record_golgi_spikes']
    pc_ids = np.unique(neurons['record_golgi_spikes'])
    spike_burst = {}
    count_burst = {}
    isi_burst = {}
    pause = {}
    pause_ratio = {}
    spike_baseline = {}
    isi_baseline = []
    for p in pc_ids:        # For each PC we compute number of spikes in window burst and pause duration
        current_pc_spikes_ids = np.where(neurons['record_golgi_spikes'] == p)
        current_pc_spikes = np.array([pc_times[i] for i in list(current_pc_spikes_ids[0])])
        # Spikes in baseline (time window before the window_burst)
        spike_baseline[p]  = current_pc_spikes[(current_pc_spikes > 0) & (current_pc_spikes < window_burst[0])]
        if len(spike_baseline[p])>1:
            isi_baseline.extend(list(np.diff(spike_baseline[p])))

        # Total number of spikes in the window_burst
        spike_burst[p]  = current_pc_spikes[(current_pc_spikes > window_burst[0]) & (current_pc_spikes < window_burst[1])]
        #spike_burst[p] = sum(map(lambda x : x>window_burst[0] and x<window_burst[1], current_pc_spikes))
        count_burst[p] = len(spike_burst[p])
        if len(spike_burst[p])>1:
            isi_burst[p] = np.mean(np.diff(spike_burst[p]))
        else:
            isi_burst[p] = -1
        ind = np.where(np.array(current_pc_spikes)>window_burst[1])
        # pause - ISI between first spike after window_burst and last spike in window_burst
        pause[p] = current_pc_spikes[ind[0][0]] - current_pc_spikes[ind[0][0]-1]  #window_burst[1]
        # Ratio between pause duration and baseline ISI for the current PC
        pause_ratio[p] = pause[p]/(np.mean(np.diff(spike_baseline[p])))

    import plotly.express as px
    lists = (pause.items())
    x, pause_values = zip(*lists)
    lists = (pause_ratio.items())
    x, pause_ratio_values = zip(*lists)
    lists = (count_burst.items())
    x, spike_burst_count = zip(*lists)
    lists = (isi_burst.items())
    x, isi_burst = zip(*lists)

    fig = make_subplots(rows=2, cols=2)

    # Add burst spike count and pause
    fig.add_trace(
        go.Scatter(x=spike_burst_count, y=pause_values, mode="markers", showlegend=False, marker=dict(color="#1f77b4")),
        row=1, col=1
    )

    # standard deviation area baseline ISI baseline ISI
    x=[min(spike_burst_count), max(spike_burst_count)]
    y1_upper = [np.mean(np.array(isi_baseline))+np.std(np.array(isi_baseline)), np.mean(np.array(isi_baseline))+np.std(np.array(isi_baseline))]
    y1_lower = [np.mean(np.array(isi_baseline))-np.std(np.array(isi_baseline)), np.mean(np.array(isi_baseline))-np.std(np.array(isi_baseline))]
    y1_lower = y1_lower[::-1]
    fig.add_trace(go.Scatter(x=x+x[::-1],
                                y=y1_upper+y1_lower,
                                fill='tozerox',
                                line=dict(color='rgba(255,255,255,0)'),
                                showlegend=False
                                ), row=1, col=1)

    # line trace baseline ISI
    fig.add_trace(go.Scatter(x=x+x[::-1],
                              y=[np.mean(np.array(isi_baseline)), np.mean(np.array(isi_baseline))],
                              line=dict(width=2.5,color="#d62728"),
                              mode='lines',name='avg baseline ISI'), row=1, col=1
                                )
    # line trace 1
    fig.add_trace(go.Scatter(x=x+x[::-1],
                              y=[1, 1],
                              line=dict(width=2.5,color="#7f7f7f"),  showlegend=False,
                              mode='lines',name='avg baseline ISI'), row=2, col=1
                                )

    # Add burst ISI and pause
    fig.add_trace(
        go.Scatter(x=isi_burst, y=pause_values, mode="markers", showlegend=False, marker=dict(color="#2ca02c")),
        row=1, col=2
    )

    # standard deviation area baseline ISI baseline ISI
    x=[min(isi_burst), max(isi_burst)]
    y1_upper = [np.mean(np.array(isi_baseline))+np.std(np.array(isi_baseline)), np.mean(np.array(isi_baseline))+np.std(np.array(isi_baseline))]
    y1_lower = [np.mean(np.array(isi_baseline))-np.std(np.array(isi_baseline)), np.mean(np.array(isi_baseline))-np.std(np.array(isi_baseline))]
    y1_lower = y1_lower[::-1]
    fig.add_trace(go.Scatter(x=x+x[::-1],
                                y=y1_upper+y1_lower,
                                fill='tozerox',
                                #fillcolor=new_col,
                                line=dict(color='rgba(255,255,255,0)'),
                                showlegend=False
                                ), row=1, col=2)

    # line trace baseline ISI
    fig.add_trace(go.Scatter(x=x+x[::-1],
                              y=[np.mean(np.array(isi_baseline)), np.mean(np.array(isi_baseline))],
                              line=dict(width=2.5,color="#d62728"),
                              mode='lines', name='avg baseline ISI', showlegend=False), row=1, col=2
                                )
    # line trace 1
    fig.add_trace(go.Scatter(x=x+x[::-1],
                              y=[1, 1],
                              line=dict(width=2.5,color="#7f7f7f"),  showlegend=False,
                              mode='lines',name='avg baseline ISI'), row=2, col=2
                                )


    # Add burst spike count and pause ratio
    fig.add_trace(
        go.Scatter(x=spike_burst_count, y=pause_ratio_values, mode="markers", showlegend=False, marker=dict(color="#1f77b4")),
        row=2, col=1
    )

    # Add burst ISI and pause ratio
    fig.add_trace(
        go.Scatter(x=isi_burst, y=pause_ratio_values, mode="markers", showlegend=False, marker=dict(color="#2ca02c")),
        row=2, col=2
    )


    # Layout
    # Pause
    fig.update_xaxes(
        title='# spikes in burst window',
        tickmode = 'array',
        tickvals = [0, 1, 2, 3, 4],
        ticktext = ['0', '1', '2', '3', '4']
        , row=1, col=1
    )
    fig.update_xaxes(
        title='ISI in burst window [ms]',
        tickmode = 'array',
        tickvals = [-1, 5, 10, 15, 20, 30, 40, 50, 60],
        ticktext = ['no burst', '5', '10', '15', '20','30','40','50','60'],
        row=1, col=2
    )
    fig.update_yaxes(
        title='pause duration [ms]',
        tickmode = 'linear',
        tick0 = 5,
        dtick = 50, row=1, col=1
    )
    fig.update_yaxes(
        title='pause duration [ms]',
        tickmode = 'linear',
        tick0 = 5,
        dtick = 50, row=1, col=2
    )

    # Pause ratio
    fig.update_xaxes(
        title='# spikes in burst window',
        tickmode = 'array',
        tickvals = [1, 2, 3, 4],
        ticktext = ['1', '2', '3', '4']
        , row=2, col=1
    )
    fig.update_xaxes(
        title='ISI in burst window [ms]',
        tickmode = 'array',
        tickvals =  [-1, 5, 10, 15, 20, 30, 40, 50, 60],
        ticktext = ['no burst', '5', '10', '15', '20','30','40','50','60'],
        row=2, col=2
    )
    fig.update_yaxes(
        title='pause ratio',
        tickmode = 'linear',
        tick0 = 0,
        dtick = 0.5, row=2, col=1
    )
    fig.update_yaxes(
        title='pause ratio',
        tickmode = 'linear',
        tick0 = 0,
        dtick = 0.5, row=2, col=2
    )
    fig.update_layout(title='Burst-pause analysis Purkinje cells')
    fig.show()


    # fig = hdf5_plot_spike_raster(groups, show=False, cutoff=300, \
    # sorted_labels=sel_labels, sorted_ids=cell_ids)
    #fig.update_yaxes(range=[-5, 30])
