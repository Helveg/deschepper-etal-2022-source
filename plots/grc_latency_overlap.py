from bsb.core import from_hdf5
import os, sys, h5py, numpy as np
sys.path.insert(0, os.path.join("..", "plots"))
import selection, plotly.graph_objs as go, scipy.stats
import pickle
from ._paths import *
from glob import glob
import selection

MFs = selection.stimulated_mf_poiss
# Re-use previous results?
frozen = False

def plot():
    fig = plot2(glob(results_path("sensory_burst", "*"))[0], color='red', shift=0)
    fig2 = plot2(results_path("sensory_burst_gabazine.hdf5"), color='grey', shift=0.2)
    fig.add_traces(fig2.data)
    return fig

def latency(data, min, max):
    c = data[:, 1]
    return np.min(c[(c > min) & (c < max)], initial=float("+inf"))

def plot2(path=None, net_path=None, stim_start=6000, stim_end=6020, color='red', shift=0):
    if path is None:
        path = glob(results_path("sensory_burst", "*"))[0]
    if net_path is None:
        net_path = network_path(selection.network)
    network = from_hdf5(net_path)
    ids = network.get_placement_set("granule_cell").identifiers
    if not frozen:
        with h5py.File(path, "r") as f:
            latencies = {id: v - stim_start for id in ids if (v := latency(f["recorders/soma_spikes/" + str(id)], stim_start, stim_end)) != float("+inf")}
        with open("grc_lat.pickle", "wb") as f:
            pickle.dump(latencies, f)
    else:
        with open("grc_lat.pickle", "rb") as f:
            latencies = pickle.load(f)


    if not frozen:
        mf_glom = network.get_connectivity_set("mossy_to_glomerulus").get_dataset()
        glom_grc = network.get_connectivity_set("glomerulus_to_granule").get_dataset()
        active_glom = mf_glom[np.isin(mf_glom[:, 0], MFs), 1]
        active_dendrites = glom_grc[np.isin(glom_grc[:, 0], active_glom), 1]
        active_grc_ids, active_dend_count = np.unique(active_dendrites, return_counts=True)
        act_grc_list = list(active_grc_ids)
        ids = np.array(list(latencies.keys()))
        x = [active_dend_count[act_grc_list.index(id)] if id in act_grc_list else 0 for id in ids]
        y = [latencies[id] for id in ids]
        with open("grc_lat2.pickle", "wb") as f:
            pickle.dump((ids, active_glom, active_dendrites, active_grc_ids, active_dend_count, x, y), f)
    else:
        with open("grc_lat2.pickle", "rb") as f:
            ids, active_glom, active_dendrites, active_grc_ids, active_dend_count, x, y = pickle.load(f)
    m = np.column_stack((x, y))
    combos, counts = np.unique(m, return_counts=True, axis=0)

    x = np.array(x)
    print("prefilter", len(x))
    y = np.array(y)[x != 0]
    x = x[x != 0]
    print("postfilter", len(x))
    print("Tiniest possible value on this machine:", np.finfo(float).tiny)
    r, p = scipy.stats.pearsonr(x, y)
    print("r=", r, " p=", max(p, np.finfo(float).tiny))
    fig = go.Figure([go.Box(y=y[x == i], name=f"{i} active dendrites", marker_color =color) for i in range(1, 5)])
    fig.update_layout(xaxis_title="Granule cells", yaxis_title="latencies of first spike [ms]")
    fig.update_layout(
        xaxis = dict(
            tickmode = 'array',
            tickvals = [1, 2, 3, 4],
            #ticktext = [f"n={len(y[x == i])}" for i in range(1, 5)]
        )
    )
    return fig

def meta():
    return {"width": 800, "height": 800}