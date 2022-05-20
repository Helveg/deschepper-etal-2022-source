from scipy.spatial import distance_matrix
import numpy as np
import plotly.graph_objs as go
import h5py
from scipy.stats import zscore
import os
import pickle
import networkx as nx
from bsb.core import from_hdf5
from scipy.sparse import coo_matrix
import itertools
import collections

random = np.random.default_rng()

def goc_graph(netw):
    G = nx.DiGraph()
    goc = netw.get_placement_set("golgi_cell")
    gap_goc = netw.get_connectivity_set("gap_goc")
    l = len(gap_goc)
    conn_m = coo_matrix((np.ones(l), (gap_goc.from_identifiers, gap_goc.to_identifiers)), shape=(l, l))
    conn_m.eliminate_zeros()
    G.add_nodes_from(goc.identifiers)
    collections.deque((G.add_edges_from(zip(itertools.repeat(from_), row.indices, map(lambda a: dict([a]), map(tuple, zip(itertools.repeat("weight"), 1 / row.data))))) for from_, row in enumerate(map(conn_m.getrow, range(len(goc))))), maxlen=0)
    return G

def coincident(a, b, diff=5):
    return np.any(np.abs(np.tile(b, (len(a), 1)) - a.reshape(-1, 1)) <= diff, axis=1)

def coincidence_matrix(tracks, diff, selected):
    co = np.zeros((len(tracks), len(tracks), 2))
    for gid, track in enumerate(tracks.values()):
        for ogid, otrack in enumerate(tracks.values()):
            if not selected[gid, ogid]:
                # Skip diagonal (self)
                continue
            co[gid, ogid, :] = (sum(coincident(track, otrack, diff)), len(track))

    return co

def skip_self(m):
    m = m.copy()
    for i in range(len(m)):
        m[i, i] = False
    return m

def include_self(m):
    m = m.copy()
    for i in range(len(m)):
        m[i, i] = True
    return m

def rem_unselected(d, sel):
    r = {}
    for k, v in d.items():
        x = v.copy()
        x[~sel, 1] = 0
        r[k] = x
    return r

def plot(result="results/results_gapx2.5.hdf5", result_ko="results/results_gap_knockout.hdf5", pkl="golgi_spike_sync.pkl", pkl_goc="golgi_tracks_25.pkl", pkl_goc_ko="golgi_gko_tracks.pkl"):
    netw = from_hdf5("networks/balanced.hdf5")
    ps = netw.get_placement_set("golgi_cell")
    ps_pos = ps.positions
    dist = 100
    selected = distance_matrix(ps_pos, ps_pos) < dist

    if not os.path.exists(pkl_goc):
        print("Reading", result)
        with h5py.File(result, "r") as f:
            golgi_tracks = {g.attrs["cell_id"]: (x := g[()][:, 1])[(x > 5500) & (x < 6000) | (x > 6500)] for g in f["recorders/soma_spikes"].values() if g.attrs["label"] == "golgi_cell"}
            with open(pkl_goc, "wb") as g:
                pickle.dump(golgi_tracks, g)
    else:
        with open(pkl_goc, "rb") as g:
            golgi_tracks = pickle.load(g)

    if not os.path.exists(pkl_goc_ko):
        with h5py.File(result_ko, "r") as f:
            golgi_gko_tracks = {g.attrs["cell_id"]: (x := g[()][:, 1])[(x > 5500) & (x < 6000) | (x > 6500)] for g in f["recorders/soma_spikes"].values() if g.attrs["label"] == "golgi_cell"}
            with open(pkl_goc_ko, "wb") as g:
                pickle.dump(golgi_gko_tracks, g)
    else:
        with open(pkl_goc_ko, "rb") as g:
            golgi_gko_tracks = pickle.load(g)
    if not os.path.exists(pkl):
        bin_widths = np.arange(0, 5.5, 0.5)
        # Spoof data for reference to uniformly random baseline
        fake_tracks = {gid: random.random(len(track)) * 2000 for gid, track in golgi_tracks.items()}
        pos = {id: p for id, p in zip(ps.identifiers, ps_pos)}
        co = {(bw, dist): coincidence_matrix(golgi_tracks, bw, skip_self(selected)) for bw in bin_widths}
        koco = {(bw, dist): coincidence_matrix(golgi_gko_tracks, bw, skip_self(selected)) for bw in bin_widths}
        fco = {(bw, dist): coincidence_matrix(fake_tracks, bw, skip_self(selected)) for bw in bin_widths}
        with open(pkl, "wb") as g:
            pickle.dump((co, koco, fco), g)
    else:
        with open(pkl, "rb") as g:
            co, koco, fco = pickle.load(g)

    G = goc_graph(netw)
    pathss = nx.shortest_path(G)
    for node, paths in pathss.items():
        for P, path in paths.items():
            selected[node, P] = selected[node, P] and len(path) == 2

    co = rem_unselected(co, selected)
    fco = rem_unselected(fco, selected)
    koco = rem_unselected(koco, selected)

    return go.Figure(
        [
            go.Scatter(
                x=list(ck[0] for ck in co.keys()),
                y=[print(np.sum(c[:, :, 1] != 0)) or np.mean((c[:, :, 0] / c[:, :, 1])[c[:, :, 1] != 0]) for c in co.values()],
                error_y=dict(
                    type="data",
                    array=[np.std((c[:, :, 0] / c[:, :, 1])[c[:, :, 1] != 0]) / np.sqrt(np.sum(c[:, :, 1] != 0)) for c in co.values()],
                    visible=True
                ),
                name="Results"
            ),
            go.Scatter(
                x=list(ck[0] for ck in co.keys()),
                y=[print(np.sum(c[:, :, 1] != 0)) or np.mean((c[:, :, 0] / c[:, :, 1])[c[:, :, 1] != 0]) for c in koco.values()],
                error_y=dict(
                    type="data",
                    array=[np.std((c[:, :, 0] / c[:, :, 1])[c[:, :, 1] != 0]) / np.sqrt(np.sum(c[:, :, 1] != 0)) for c in koco.values()],
                    visible=True
                ),
                name="Knockout"
            ),
            go.Scatter(
                x=list(ck[0] for ck in fco.keys()),
                y=[np.mean((c[:, :, 0] / c[:, :, 1])[c[:, :, 1] != 0]) for c in fco.values()],
                error_y=dict(
                    type="data",
                    array=[np.std((c[:, :, 0] / c[:, :, 1])[c[:, :, 1] != 0]) / np.sqrt(np.sum(c[:, :, 1] != 0)) for c in fco.values()],
                    visible=True
                ),
                name="Random"
            )
        ],
        layout=dict(
            yaxis_tickformat = '.0%',
            yaxis_title="% Coincident spikes",
            xaxis_title="Time lag window (±ms)",
            title_text="Golgi cell spike coincidence"
        )
    )
