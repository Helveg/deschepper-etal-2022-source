from bsb.core import from_hdf5
import os, sys, h5py, numpy as np
sys.path.insert(0, os.path.join("..", "plots"))
import selection, plotly.graph_objs as go, scipy.stats
import pickle
from ._paths import *
from glob import glob
from bsb.config import get_result_config

def latency(data, min):
    c = data[:, 1]
    return np.min(c[c > min])

def plot(net_path=None, stim_start=6000, stim_end=6020):
    if net_path is None:
        net_path = network_path("batch_1", "*.hdf5")
    paths = glob(net_path)
    carry_x = np.empty((0,))
    carry_y = np.empty((0,))
    for path in paths:
        id = int(path.split("_")[-1].split(".")[0])
        print("Analyzing net", id)
        with open(results_path("..", "pkl_ca", "calcium_data", f"calcium_{id}.pickle"), "rb") as f:
            result = pickle.load(f)["calcium"]["_data"]
        network = from_hdf5(path)
        MFs = selection.mf_batch_1[id]
        ps = network.get_placement_set("granule_cell")
        pos = ps.positions
        border = (pos[:, 0] > 290) | (pos[:, 0] < 10) | (pos[:, 2] > 190) | (pos[:, 2] < 10)
        ids = ps.identifiers[~border]
        mf_glom = network.get_connectivity_set("mossy_to_glomerulus").get_dataset()
        glom_grc = network.get_connectivity_set("glomerulus_to_granule").get_dataset()
        active_glom = mf_glom[np.isin(mf_glom[:, 0], MFs), 1]
        active_dendrites = glom_grc[np.isin(glom_grc[:, 0], active_glom), 1]
        d = dict(zip(*np.unique(active_dendrites, return_counts=True)))
        grc_to_dend = np.vectorize(lambda x: d.get(x, 0))
        grc_to_calc = np.vectorize(result.get)
        x = grc_to_dend(ids)
        y = grc_to_calc(ids)
        carry_x = np.concatenate((carry_x, x))
        carry_y = np.concatenate((carry_y, y))


    x = carry_x
    y = carry_y
    r, p = scipy.stats.pearsonr(x, y)
    print("r=", r, " p=", max(p, np.finfo(float).tiny))
    fig = go.Figure([
        go.Box(
            y=y[x == d] * 1000,
            name=f"{d} active dendrite" + "s" * (d > 1),
            boxpoints=False
        )
        for d in range(5)
    ])
    fig.update_layout(
        title_text="Granule cell calcium concentration",
        xaxis_title="Granule cells",
        yaxis_title="Dendritic calcium concentration [μM]",
        showlegend=False
    )
    return fig
