from bsb.core import from_hdf5
import selection, h5py, numpy as np
from scipy import sparse
from plotly import graph_objs as go

def crop(data, min, max, indices=False):
    if len(data.shape) > 1:
        c = data[:, 1]
    else:
        c = data
    if indices:
        return np.where((c > min) & (c < max))[0]
    return c[(c > min) & (c < max)]
from ._paths import *
from glob import glob
import selection

def plot(path=None, net_path=None):
    if path is None:
        path = glob(results_path("sensory_burst", "*"))[0]
    if net_path is None:
        net_path = network_path(selection.network)
    network = from_hdf5(net_path)
    MFs = selection.stimulated_mf_sync
    mf_glom = network.get_connectivity_set("mossy_to_glomerulus").get_dataset()
    glom_grc = network.get_connectivity_set("glomerulus_to_granule").get_dataset()
    active_glom = mf_glom[np.isin(mf_glom[:, 0], MFs), 1]
    active_dendrites = glom_grc[np.isin(glom_grc[:, 0], active_glom), 1]
    active_grc_ids, active_dend_count = np.unique(active_dendrites, return_counts=True)
    very_active_grc = active_grc_ids[active_dend_count >= 2]

    grc_ps = network.get_placement_set("granule_cell")
    sc_ps = network.get_placement_set("stellate_cell")
    grc_sc = network.get_connectivity_set("parallel_fiber_to_stellate").get_dataset().astype(int)
    sc_m = sparse.coo_matrix((np.ones(len(grc_sc)), (grc_sc[:, 0], grc_sc[:, 1])), shape=(max(grc_ps.identifiers) + 1, max(sc_ps.identifiers) + 1)).tocsr()

    sc_roi = grc_sc[np.isin(grc_sc[:, 0], very_active_grc)]
    sc_x = np.unique(sc_roi, return_counts=True)[1]
    figs = {"structural": go.Figure(go.Histogram(x=sc_x))}
    with h5py.File(path, "r") as f:
        g = f["recorders/soma_spikes"]
        spikes_inc_sc = {id: [] for id in sc_ps.identifiers}
        for id, d in g.items():
            if d.attrs["label"] != "granule_cell":
                continue
            spikes = crop(d[()], 6000, 6100)
            targets = sc_m.getrow(id).nonzero()[1]
            for t in targets:
                spikes_inc_sc[t].extend(spikes)
        spikes_inc_sc = {id: np.array(v) for id, v in spikes_inc_sc.items()}
        sc_spikes = np.array([np.bincount(np.ceil((v - 6000) / 5).astype(int), minlength=21) for v in spikes_inc_sc.values()])
        fig = go.Figure()
        for sc in sc_spikes:
            fig.add_trace(go.Scatter(x=np.arange(6000, 6100, 5), y=sc, mode="lines", line=dict(color="blue"), opacity=0.1))
        figs["skyscrapers"] = fig
        fig = go.Figure()
        for sc in sc_spikes:
            fig.add_trace(go.Scatter(x=np.arange(6000, 6100, 5) + np.random.rand(20) * 5 - 2.5, y=sc, mode="markers", marker=dict(color="blue", size=2)))
        figs["manhattan"] = fig
        fig = go.Figure()
        for i, bin in enumerate(sc_spikes.T):
            print(i, len(bin))
            fig.add_trace(go.Violin(x0=6000 + i * 5, y=bin, showlegend=False, fillcolor="aquamarine", line_color="black"))
        figs["violin"] = fig
        fig.show()
        spikes_per_impulse = np.array([[len(crop(v, 6000 + i * 25, 6000 + (i + 1) * 25)) for i in range(4)] for v in spikes_inc_sc.values()])
        for i in range(4):
            figs[f"impulse{i+1}"] = go.Figure(go.Histogram(x=spikes_per_impulse[:, i]))
    return figs
