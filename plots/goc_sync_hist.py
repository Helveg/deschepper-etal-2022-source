from scipy.spatial import distance_matrix
from plotly.subplots import make_subplots
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
import pymannkendall as mk

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

def overlap_bins(a, b, binsize=0.5):
    if not (len(a) and len(b)):
        return 0
    bincount = int(max(np.max(a), np.max(b)) // binsize + 2)
    a_binned = np.bincount((a / binsize).astype(int), minlength=bincount)
    b_binned = np.bincount((b / binsize).astype(int), minlength=bincount)
    return sum(a_binned + b_binned - np.maximum(a_binned, b_binned))

def crosscor(a, b, step=0.1, binsize=0.5, slide=100):
    b = b + slide
    corr_spikes = [overlap_bins(a + plus_a, b, binsize=binsize) for plus_a in np.arange(0, slide * 2, step)]
    return corr_spikes

def z_title(t1, t2, samples):
    return f"t1: {len(t1)}, t2: {len(t2)}; N: {len(samples)}; mean: {np.mean(samples)}; std: {np.std(samples)}"

def plot(pkl="goc_sync_hist.pkl", track_pkl="golgi_tracks_25.pkl", track_file="results/results_gapx2.5.hdf5", track_ko_pkl="golgi_gko_tracks.pkl", track_ko_file="results/results_gap_knockout.hdf5"):
    if not os.path.exists(track_pkl):
        with h5py.File(track_file, "r") as f:
            golgi_tracks = {g.attrs["cell_id"]: (x := g[()][:, 1])[(x > 5500) & (x < 6000) | (x > 6500)] for g in f["recorders/soma_spikes"].values() if g.attrs["label"] == "golgi_cell"}
        with open(track_pkl, "wb") as g:
            pickle.dump(golgi_tracks, g)
    else:
        with open(track_pkl, "rb") as g:
            golgi_tracks = pickle.load(g)

    if not os.path.exists(track_ko_pkl):
        with h5py.File(track_ko_file, "r") as f:
            golgi_gko_tracks = {g.attrs["cell_id"]: (x := g[()][:, 1])[(x > 5500) & (x < 6000) | (x > 6500)] for g in f["recorders/soma_spikes"].values() if g.attrs["label"] == "golgi_cell"}
        with open(track_ko_pkl, "wb") as g:
            pickle.dump(golgi_gko_tracks, g)
    else:
        with open(track_ko_pkl, "rb") as g:
            golgi_gko_tracks = pickle.load(g)

    step = 0.1
    binsize = 0.5
    binsteps = int(binsize / 2 // step)
    slide = 5
    lag0 = int(slide // step)
    netw = from_hdf5("networks/balanced.hdf5")
    G = goc_graph(netw)
    id_map = dict(zip(G.nodes(), itertools.count()))
    pos = netw.get_placement_set("golgi_cell").positions
    pdist = distance_matrix(pos, pos)
    L = len(G.nodes())
    if not os.path.exists(pkl):
        netw_dist = np.empty((L, L))
        zscore_m = np.empty((L, L, lag0 * 2 + 2))
        zscore_mko = np.empty((L, L, lag0 * 2 + 2))
        for node, paths in nx.shortest_path_length(G, weight="weight"):
            print("It", node)
            for k, d in paths.items():
                netw_dist[node, k] = d
                me, other = golgi_tracks[id_map[node]], golgi_tracks[id_map[k]]
                cross = crosscor(me, other, binsize=binsize, step=step, slide=slide)
                zscore_m[node, k, :] = zscore(cross)
                meko, otherko = golgi_gko_tracks[id_map[node]], golgi_gko_tracks[id_map[k]]
                crossko = crosscor(meko, otherko, binsize=binsize, step=step, slide=slide)
                zscore_mko[node, k, :] = zscore(crossko)
        with open(pkl, "wb") as f:
            pickle.dump((netw_dist, zscore_m, zscore_mko), f)
    else:
        with open(pkl, "rb") as g:
            netw_dist, zscore_m, zscore_mko = pickle.load(g)

    pathss = nx.shortest_path(G)
    steps = np.empty((L, L))
    selected = np.zeros((L, L), dtype=bool)
    for node, paths in pathss.items():
        for P, path in paths.items():
            steps[node, P] = len(path)
            if pdist[node, P] < 100:
                selected[node, P] = True

    fig = go.Figure(
        [
            go.Scatter(name=f"Direct pairs", line_color="#332EBC", x=np.linspace(-5, 5, 100), y=np.nanmean(zscore_m[selected & (steps == 2), :], axis=0)),
            go.Scatter(name=f"Indirect pairs", line_color="#9c99ff", x=np.linspace(-5, 5, 100), y=np.nanmean(zscore_m[selected & (steps > 2), :], axis=0)),
            go.Scatter(name=f"Knockout", line_color="#DC143C", x=np.linspace(-5, 5, 100), y=np.nanmean(zscore_mko[selected & (steps > 1), :], axis=0)),
        ],
        layout_title_text="Golgi millisecond precision",
        layout_xaxis_title="Time lag (ms)",
        layout_yaxis_title="Z-score"
    )

    return fig
