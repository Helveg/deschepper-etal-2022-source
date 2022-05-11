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
        bin_widths = np.arange(0, 5.5, 0.5)
        ps = from_hdf5("networks/balanced.hdf5").get_placement_set("golgi_cell")
        ps_pos = ps.positions
        selected = distance_matrix(ps_pos, ps_pos) < dist
        # Spoof data for reference to uniformly random baseline
        fake_tracks = {gid: random.random(len(track)) * 2500 for gid, track in golgi_tracks.items()}
        pos = {id: p for id, p in zip(ps.identifiers, ps_pos)}
        co = {(bw, dist): coincidence_matrix(golgi_tracks, bw, skip_self(selected)) for bw in bin_widths}
        nsco = {(bw, dist): coincidence_matrix(golgi_tracks, bw, include_self(selected)) for bw in bin_widths}
        fco = {(bw, dist): coincidence_matrix(fake_tracks, bw, skip_self(selected)) for bw in bin_widths}
        with open("golgi_nsync.pkl", "wb") as g:
            pickle.dump((co, nsco, fco), g)
    else:
        with open("golgi_nsync.pkl", "rb") as g:
            co, nsco, fco = pickle.load(g)

    return go.Figure(
        [
            go.Scatter(
                x=list(ck[0] for ck in co.keys()),
                y=[np.mean((c[:, :, 0] / c[:, :, 1])[c[:, :, 1] != 0]) for c in co.values()],
                error_y=dict(
                    type="data",
                    array=[np.std((c[:, :, 0] / c[:, :, 1])[c[:, :, 1] != 0]) for c in co.values()],
                    visible=True
                ),
                name="Results"
            ),
            go.Scatter(
                x=list(ck[0] for ck in fco.keys()),
                y=[np.mean((c[:, :, 0] / c[:, :, 1])[c[:, :, 1] != 0]) for c in fco.values()],
                error_y=dict(
                    type="data",
                    array=[np.std((c[:, :, 0] / c[:, :, 1])[c[:, :, 1] != 0]) for c in fco.values()],
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
