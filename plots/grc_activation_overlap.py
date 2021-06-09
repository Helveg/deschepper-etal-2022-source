from bsb.core import from_hdf5
import os, sys, h5py, numpy as np
sys.path.insert(0, os.path.join("..", "plots"))
import selection, plotly.graph_objs as go, scipy.stats
import pickle
from ._paths import *
from glob import glob
import selection
import hashlib

def hash(s):
    return hashlib.sha256(str(s).encode()).hexdigest()

MFs = selection.stimulated_mf_poiss
# Re-use previous results?
frozen = True

def plot():
    trc_control = plot2(title="Control", path=results_path("sensory_gabazine", "sensory_burst_control.hdf5"), color='red', shift=0)
    trc_gaba = plot2(title="Gabazine", path=results_path("sensory_gabazine", "sensory_burst_gabazine.hdf5"), color='grey', shift=0.2)
    fig = go.Figure(trc_control + trc_gaba)
    fig.update_layout(title_text="Influence of gabazine on granule cell activity", xaxis_title="Granule cells", yaxis_title="Number of spikes", boxmode="group", xaxis_type="category", xaxis_range=[-0.5, 3.5])
    return fig

def crop(data, min, max, indices=False):
    c = data[:, 1]
    if indices:
        return np.where((c > min) & (c < max))[0]
    return c[(c > min) & (c < max)]

def plot2(path=None, title=None, net_path=None, stim_start=6000, stim_end=6040, color='red', shift=0):
    if path is None:
        path = glob(results_path("sensory_burst", "*"))[0]
    if net_path is None:
        net_path = network_path(selection.network)
    network = from_hdf5(net_path)
    ids = network.get_placement_set("granule_cell").identifiers
    if not frozen:
        with h5py.File(path, "r") as f:
            activity = {id: len(crop(f["recorders/soma_spikes/" + str(id)], stim_start, stim_end)) for id in ids}
            response= {id: crop(f["recorders/soma_spikes/" + str(id)], stim_start, stim_end) for id in ids}
            for id in ids:
                #print(response[id])
                if len(response[id])>0:
                    firstSpikeLat = {id: (response[id][0]-stim_start)}

        with open(f"grc_act_{hash(path)}.pickle", "wb") as f:
            pickle.dump(activity, f)
    else:
        with open(f"grc_act_{hash(path)}.pickle", "rb") as f:
            activity = pickle.load(f)


    if not frozen:
        mf_glom = network.get_connectivity_set("mossy_to_glomerulus").get_dataset()
        glom_grc = network.get_connectivity_set("glomerulus_to_granule").get_dataset()
        active_glom = mf_glom[np.isin(mf_glom[:, 0], MFs), 1]
        active_dendrites = glom_grc[np.isin(glom_grc[:, 0], active_glom), 1]
        active_grc_ids, active_dend_count = np.unique(active_dendrites, return_counts=True)
        act_grc_list = list(active_grc_ids)
        x = [active_dend_count[act_grc_list.index(id)] if id in act_grc_list else 0 for id in ids]
        y = [activity[id] for id in ids]
        with open(f"grc_act2_{hash(path)}.pickle", "wb") as f:
            pickle.dump((active_glom, active_dendrites, active_grc_ids, active_dend_count, x, y), f)
    else:
        with open(f"grc_act2_{hash(path)}.pickle", "rb") as f:
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
    # Because of the 25000 samples of x=0 the whole statistical test was skewed and
    # reporting linear correlations p-values of 0.0 (smaller than the tiniest float value)
    # for whatever data I fed into it... kind of a weak implementation of the test I guess
    # or a weakness of the test in general. I exclude the 25000 zeros to test for a linear
    # correlation in those GrC stimulated by the gloms that certainly isn't affected by
    # the massive amount of zeroes.
    #
    # Afterwards I still received p = 0.0 and reducing the sample size also increased the
    # p values above this impossible value, for example with 100 ramdom samples the p
    # value was 2.4e-31 and with 30 samples p = 2.1e-8. So it's safe to assume that the
    # test actually works and that because of our large sample size p < ϵ where ϵ is the
    # tiniest possible float value that can be represented on this machine. When
    # it reports 0.0 it is reporting p < ϵ

    print("Tiniest possible value on this machine:", np.finfo(float).tiny)
    r, p = scipy.stats.pearsonr(x[x != 0], y[x != 0])
    print("r=", r, " p=", max(p, np.finfo(float).tiny))
    fig = go.Figure()
    return [
        go.Box(
            y=y,
            name=f"{d} active dendrite" + "s" * (d > 1),
            legendgroup=title,
            showlegend=False,
            marker_color=color,
            boxpoints=False,
        )
        for d, y in enumerate(combo_iter()) if d > 0
    ] + [go.Box(y=[1], marker_color=color, name=title, legendgroup=title)]

def meta():
    return {"width": 800, "height": 800}