from bsb.core import from_hdf5
import os, sys, h5py, numpy as np
sys.path.insert(0, os.path.join("..", "plots"))
import selection, plotly.graph_objs as go, scipy.stats
import pickle
from ._paths import *
from glob import glob
from bsb.config import get_result_config
from ennemi import estimate_mi

def latency(data, min):
    c = data[:, 1]
    return np.min(c[c > min])

def plot(net_path=None, stim_start=6000, stim_end=6020, ret_nmi=False):
    if net_path is None:
        net_path = network_path(selection.network)
    network = from_hdf5(net_path)
    ps = network.get_placement_set("granule_cell")
    pos = ps.positions
    border = (pos[:, 0] > 290) | (pos[:, 0] < 10) | (pos[:, 2] > 190) | (pos[:, 2] < 10)
    ids = ps.identifiers[~border]
    MFs = selection.stimulated_mf_poiss
    mf_glom = network.get_connectivity_set("mossy_to_glomerulus").get_dataset()
    glom_grc = network.get_connectivity_set("glomerulus_to_granule").get_dataset()
    active_glom = mf_glom[np.isin(mf_glom[:, 0], MFs), 1]
    active_dendrites = glom_grc[np.isin(glom_grc[:, 0], active_glom), 1]
    d = dict(zip(*np.unique(active_dendrites, return_counts=True)))
    grc_to_dend = np.vectorize(lambda x: d.get(x, 0))
    x = grc_to_dend(ids)
    carry_x = np.empty((0,))
    carry_y = np.empty((0,))
    for path in glob(results_path("..", "pkl_ca", "calcium_data", "*.pickle")):
        id = int(path.split(os.sep)[-1].split(".")[0].split("_")[-1])
        print("Analyzing result", id, path)
        with open(path, "rb") as f:
            result = pickle.load(f)["calcium"]["_data"]
        grc_to_calc = np.vectorize(result.get)
        y = grc_to_calc(ids)
        carry_x = np.concatenate((carry_x, x))
        carry_y = np.concatenate((carry_y, y))
    carry_yg = np.empty((0,))
    for path in glob(results_path("..", "pkl_ca", "calcium_data", "gabazine", "*.pickle")):
        id = int(path.split(os.sep)[-1].split(".")[0].split("_")[-1])
        print("Analyzing result", id, path)
        with open(path, "rb") as f:
            result = pickle.load(f)["calcium"]["_data"]
        grc_to_calc = np.vectorize(result.get)
        y = grc_to_calc(ids)
        carry_yg = np.concatenate((carry_yg, y))


    x = carry_x
    y = carry_y
    yg = carry_yg
    mi = estimate_mi(y, x, normalize=True)[0, 0]
    r, p = scipy.stats.pearsonr(x, y)
    print("mi=", mi, "r=", r, " p=", max(p, np.finfo(float).tiny))
    r, p = scipy.stats.pearsonr(x, yg)
    gmi = estimate_mi(yg, x, normalize=True)[0, 0]
    print("[GABA] mi=", mi, "r=", r, " p=", max(p, np.finfo(float).tiny))
    if ret_nmi:
        return mi, gmi
    fig = go.Figure([
        go.Scatter(
            y=list(np.mean(y[x == i]) for i in range(1, 5)),
            error_y=dict(
                type='data', # value of error bar given in data coordinates
                array=list(np.std(y[x == i]) for i in range(1, 5)),
                visible=True
            ),
            x=np.arange(1, 5) - 0.1,
            name="Control",
            legendgroup="Control",
            showlegend=False,
            mode="markers",
            marker_color="red",
        ),
        go.Scatter(
            y=list(np.mean(yg[x == i]) for i in range(1, 5)),
            error_y=dict(
                type='data', # value of error bar given in data coordinates
                array=list(np.std(yg[x == i]) for i in range(1, 5)),
                visible=True
            ),
            x=np.arange(1, 5) + 0.1,
            name="Gabazine",
            legendgroup="Gabazine",
            showlegend=False,
            mode="markers",
            marker_color="grey",
        ),
    ])
    fig.update_layout(
        title_text="Granule cell calcium concentration",
        xaxis_title="Active dendrites",
        yaxis_title="[Ca<sup>2+</sup>]<sub>in</sub> [μM]",
        showlegend=False,
        xaxis_tickmode="linear",
        xaxis_tick0=1,
        xaxis_dtick=1,
    )
    return fig

def meta():
    return {"width": 800, "height": 800}
