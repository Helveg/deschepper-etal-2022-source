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

def make_mutex_tracks(n=50, t=8000, bin_pow=1.5):
    space, step = np.linspace(0, t, int(n  ** bin_pow), retstep=True)
    track = random.choice(space, n * 2, replace=False)
    t1, t2 = track[:n], track[n:]
    return track, t1, t2, step / 2

def make_sync(t1, t2, sync=0.9, jitter=0.3):
    l1, l2 = int(len(t1) * sync), int(len(t2) * (1 - sync))
    return np.concatenate((
        random.choice(t1, l1),
        random.choice(t2, l2)
    )) + random.random(l1 + l2) * jitter - jitter / 2

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

if not os.path.exists("golgi_tracks.pkl"):
    with h5py.File("results/golgi_spike_example.hdf5", "r") as f:
        golgi_tracks = {g.attrs["cell_id"]: (x := g[()][:, 1])[(x > 5500) & (x < 6000) | (x > 6050)] for g in f["recorders/soma_spikes"].values() if g.attrs["label"] == "golgi_cell"}
        with open("golgi_tracks.pkl", "wb") as g:
            pickle.dump(golgi_tracks, g)
else:
    with open("golgi_tracks.pkl", "rb") as g:
        golgi_tracks = pickle.load(g)

def plot():
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
    if not os.path.exists("goc_histcorr.pkl"):
        netw_dist = np.empty((L, L))
        zscore_m = np.empty((L, L, lag0 * 2 + 2))
        for node, paths in nx.shortest_path_length(G, weight="weight"):
            print("It", node)
            for k, d in paths.items():
                netw_dist[node, k] = d
                me, other = golgi_tracks[id_map[node]], golgi_tracks[id_map[k]]
                cross = crosscor(me, other, binsize=binsize, step=step, slide=slide)
                zscore_m[node, k, :] = zscore(cross)
        with open("goc_histcorr.pkl", "wb") as f:
            pickle.dump((netw_dist, zscore_m), f)
    else:
        with open("goc_histcorr.pkl", "rb") as g:
            netw_dist, zscore_m = pickle.load(g)

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
            go.Scatter(name=f"{s-1} steps away", x=np.linspace(-5, 5, 100), y=np.nanmean(zscore_m[selected & (steps == s), :], axis=0))
            for s in range(2, 5)
        ]
        +
        [
            go.Scatter(name=f"mean", x=np.linspace(-5, 5, 100), y=np.nanmean(zscore_m[selected & (steps > 1), :], axis=0))
        ]
        # +
        # [
        #     go.Scatter(x=np.linspace(-5, 5, 100), y=z)
        #     for z in zscore_m[selected, :]
        # ]
    )

    return fig
