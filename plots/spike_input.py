from bsb.core import from_hdf5
import selection, h5py, numpy as np
from scipy import sparse
from plotly import graph_objs as go

network = from_hdf5("networks/300x_200z.hdf5")

def crop(data, min, max, indices=False):
    if len(data.shape) > 1:
        c = data[:, 1]
    else:
        c = data
    if indices:
        return np.where((c > min) & (c < max))[0]
    return c[(c > min) & (c < max)]

def calc_spike_input(activity, conn_sets, id):
    multiplier_map = {}
    for conn_set in conn_sets:
        data = conn_set.get_dataset().astype(int)
        print(data.shape)
        roi = data[data[:, 1] == id]
        inputs, multiplicity = np.unique(roi, axis=0, return_counts=True)
        senders = inputs[:, 0]
        for sender, mult in zip(senders, multiplicity):
            multiplier_map[sender] = multiplier_map.get(sender, 0) + mult
    input_spikes = []
    for sender_id, multiplier in multiplier_map.items():
        # Get the spike vector of the transmitting cell
        act_vector = activity.get(str(sender_id), np.empty((0, 2)))[:, 1]
        # Add spike vector multiple times depending on multiplicity between the cell pair.
        for _ in range(multiplier):
            input_spikes.extend(act_vector)
    return input_spikes

def sliding_average(series, window_width):
    return np.convolve(series, np.ones(window_width) / window_width, mode="valid")

def prep_figure(name, h5file, cell, bin_width, window_widths, conn_sets):
    with h5py.File(h5file, "r") as f:
        activity = f["recorders/soma_spikes"]
        input_spikes = np.array(calc_spike_input(activity, conn_sets, cell))
        print(input_spikes.shape)
        bins = np.bincount(np.floor(input_spikes / bin_width).astype(int))
        fig = go.Figure(
            go.Bar(name=f"Input spikes per {bin_width}ms", y=bins, x=np.arange(0, len(bins) * bin_width, bin_width)),
            layout=dict(title_text=f"Input spike analysis of cell {cell} during {name} stimulation"),
        )
        for window_width in window_widths:
            sliding_avg = sliding_average(bins, window_width)
            offset = window_width / 2 * bin_width
            fig.add_scatter(y=sliding_avg, x=np.arange(offset, len(bins) * bin_width - offset, bin_width), name=f"Sliding average ({window_width})")
    return fig

def plot():

    cell = 561
    bin_width = 5
    window_widths = (5, 7, 11, 20)
    conn_sets = [network.get_connectivity_set(x) for x in ("parallel_fiber_to_stellate",)]
    figs = {}
    figs["sync"] = prep_figure("sync", "results/results_365b0_sync.hdf5", cell, bin_width, window_widths, conn_sets)
    figs["sensory"] = prep_figure("sensory burst", "results/results_365b0.hdf5", cell, bin_width, window_widths, conn_sets)

    return figs
