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

def latency(data, min):
    c = data[:, 1]
    return np.min(c[c > min])

def plot(path=None, net_path=None, stim_start=6000, stim_end=6020):
    if path is None:
        path = glob(results_path("sensory_burst", "*"))[0]
    if net_path is None:
        net_path = network_path(selection.network)
    network = from_hdf5(net_path)
    ids = network.get_placement_set("granule_cell").identifiers
    if not frozen:
        with h5py.File(path, "r") as f:
            latencies = {id: latency(f["recorders/soma_spikes/" + str(id)], stim_start) for id in ids}
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
        x = [active_dend_count[act_grc_list.index(id)] if id in act_grc_list else 0 for id in ids]
        y = [latencies[id] for id in ids]
        with open("grc_lat2.pickle", "wb") as f:
            pickle.dump((active_glom, active_dendrites, active_grc_ids, active_dend_count, x, y), f)
    else:
        with open("grc_lat2.pickle", "rb") as f:
            active_glom, active_dendrites, active_grc_ids, active_dend_count, x, y = pickle.load(f)
    m = np.column_stack((x, y))
    combos, counts = np.unique(m, return_counts=True, axis=0)

    def combo_iter():
        for i in range(np.max(combos[:, 0]) + 1):
            ind = combos[:, 0] == i
            xs = combos[ind, 1]
            ns = counts[ind]
            if ns.size:
                yield np.concatenate(tuple(np.repeat(x, n) for x, n in zip(xs, ns)))

    x = np.array(x)
    y = np.array(y)
    print("Tiniest possible value on this machine:", np.finfo(float).tiny)
    r, p = scipy.stats.pearsonr(x[x != 0], y[x != 0])
    print("r=", r, " p=", max(p, np.finfo(float).tiny))
    fig = go.Figure([go.Box(y=y, name=f"{d} active dendrites") for d, y in enumerate(combo_iter())])
    fig.update_layout(xaxis_title="Granule cells", yaxis_title="latencies of first spike [ms]")
    for dends, spikes, count in map(lambda y: (y[1][0], y[1][1], counts[y[0]]), enumerate(combos)):
        if (dends + spikes == 0) or (dends == 1 and spikes < 3) or (dends == 2 and spikes < 5) or (dends == 3 and np.abs(spikes - 5.5) <= 1.5) or dends == 4:
            continue
        fig.add_annotation(text=f"n={count}", x=dends + 0.12, y=spikes, showarrow=False)
    fig.update_layout(
        xaxis = dict(
            tickmode = 'array',
            tickvals = [0, 1, 2, 3, 4],
            ticktext = [f"n={len(y)}" for y in combo_iter()]
        )
    )
    return fig

def meta():
    return {"width": 800, "height": 800}
