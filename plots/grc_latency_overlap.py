from bsb.core import from_hdf5
import os, sys, h5py, numpy as np
sys.path.insert(0, os.path.join("..", "plots"))
import selection, plotly.graph_objs as go, scipy.stats
import pickle
from ._paths import *
from glob import glob
import selection
import hashlib
from ennemi import estimate_mi

def hash(s):
    return hashlib.sha256(str(s).encode()).hexdigest()

MFs = selection.stimulated_mf_poiss
# Re-use previous results?
frozen = True

def plot(ret_nmi=False):
    trc_control = plot2(title="Control", batch_path=results_path("balanced_sensory", "*.hdf5"), color='red', shift=-0.1, ret_nmi=ret_nmi)
    trc_gaba = plot2(title="Gabazine", batch_path=results_path("balanced_sensory", "gabazine", "*.hdf5"), color='grey', transpose=True, shift=0.1, ret_nmi=ret_nmi)
    if ret_nmi:
        # These vars will be the MI values
        return trc_control, trc_gaba
    fig = go.Figure(trc_control + trc_gaba)
    fig.update_layout(title_text="Granule cell latency", xaxis_title="Granule cells", yaxis_title="Number of spikes")
    fig.update_layout(
        xaxis_title="Granule cells",
        yaxis_title="Latency of first spike [ms]",
        xaxis_tickmode="linear",
        xaxis_tick0=1,
        xaxis_dtick=1,
    )
    return fig

def latency(data, min, max, transpose=False):
    if transpose:
        data = data[()].T
    c = data[:, 1]
    return np.min(c[(c > min) & (c < max)], initial=float("+inf"))

def plot2(title=None, batch_path=None, net_path=None, stim_start=6000, stim_end=6020, color='red', transpose=True, shift=0.0, ret_nmi=False):
    if batch_path is None:
        batch_path = results_path("balanced_sensory", "*.hdf5")
    if net_path is None:
        net_path = network_path(selection.network)
    paths = glob(batch_path)
    network = from_hdf5(net_path)
    if not frozen:
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
        x, y = [], []
        for path in paths:
            with h5py.File(path, "r") as f:
                print("Analyzing", path)
                latency_batch = np.fromiter(
                    (
                        (
                            latency(
                                f["recorders/soma_spikes/" + str(id)],
                                stim_start,
                                stim_end,
                                transpose=transpose
                            )
                            - stim_start
                        )
                        for id in ids
                    ),
                    dtype=float
                )
                mask = latency_batch != float("+inf")
                y.append(latency_batch[mask])
                x.append(grc_to_dend(ids[mask]))
        x, y = map(np.concatenate, (x, y))
        with open(f"grc_lat_{hash(batch_path)}.pickle", "wb") as f:
            pickle.dump((x, y), f)
    else:
        with open(f"grc_lat_{hash(batch_path)}.pickle", "rb") as f:
            x, y = pickle.load(f)

    m = np.column_stack((x, y))
    combos, counts = np.unique(m, return_counts=True, axis=0)

    x = np.array(x)
    print("prefilter", len(x))
    y = np.array(y)[x != 0]
    x = x[x != 0]
    print("postfilter", len(x))
    mi = estimate_mi(y, x, normalize=True)[0, 0]
    print("Tiniest possible value on this machine:", np.finfo(float).tiny)
    r, p = scipy.stats.pearsonr(x, y)
    print("mi=", mi, "r=", r, " p=", max(p, np.finfo(float).tiny))
    if ret_nmi:
        return mi
    return [
        go.Scatter(
            y=list(np.mean(y[x == i]) for i in range(1, 5)),
            error_y=dict(
                type='data', # value of error bar given in data coordinates
                array=list(np.std(y[x == i]) for i in range(1, 5)),
                visible=True
            ),
            x=np.arange(1, 5) + shift,
            name=title,
            legendgroup=title,
            showlegend=False,
            mode="markers",
            marker_color=color,
        )
    ]


def meta():
    return {"width": 800, "height": 800}
