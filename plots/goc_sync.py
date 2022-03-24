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
    bincount = int(max(np.max(a), np.max(b)) // binsize + 2)
    a_binned = np.bincount((a / binsize).astype(int), minlength=bincount)
    b_binned = np.bincount((b / binsize).astype(int), minlength=bincount)
    return sum(a_binned + b_binned > np.maximum(a_binned, b_binned))

def crosscor(a, b, step=0.1, binsize=0.5, slide=100):
    b = b + slide
    corr_spikes = [overlap_bins(a + plus_a, b, binsize=binsize) for plus_a in np.arange(0, slide * 2, step)]
    return corr_spikes

if not os.path.exists("golgi_tracks.pkl"):
    with h5py.File("results/golgi_spike_example.hdf5", "r") as f:
        golgi_tracks = {g.attrs["cell_id"]: g[()][:, 1] for g in f["recorders/soma_spikes"].values() if g.attrs["label"] == "golgi_cell"}
        with open("golgi_tracks.pkl", "wb") as g:
            pickle.dump(golgi_tracks, g)
else:
    with open("golgi_tracks.pkl", "rb") as g:
        golgi_tracks = pickle.load(g)

def plot():
    step = 0.05
    binsize = 0.3
    slide = 100
    f0, f1, f2, bin = make_mutex_tracks(n=20, t=100, bin_pow=1.5)
    sync = make_sync(f1, f2, sync=0.9, jitter=bin/2)
    like_fig = crosscor(f1, sync, step=step, binsize=binsize, slide=slide)
    prelim_vars = (step, binsize, slide)
    step = 0.5
    binsize = 5
    slide = 100
    t0, t1, t2, bin = make_mutex_tracks()
    tracks = [make_sync(t1, t2, sync=(1 - i), jitter=bin/2) for i in range(2)]
    art_hit = crosscor(t1, tracks[0], step=step, binsize=binsize, slide=slide)
    art_miss = crosscor(t1, tracks[-1], step=step, binsize=binsize, slide=slide)
    z = crosscor(golgi_tracks[59], golgi_tracks[65], step=step, binsize=binsize, slide=slide)
    z_long = crosscor(golgi_tracks[29], golgi_tracks[26], step=step, binsize=binsize, slide=slide)
    netw = from_hdf5("networks/balanced.hdf5")
    G = goc_graph(netw)
    id_map = dict(zip(G.nodes(), itertools.count()))
    L = len(G.nodes())
    if not os.path.exists("goc_netw_crosscorr.pkl"):
        netw_dist = np.empty((L, L))
        crosscor_m = np.empty((L, L))
        for node, paths in nx.shortest_path_length(G, weight="weight"):
            print("It", node)
            for k, d in paths.items():
                netw_dist[node, k] = d
                me, other = golgi_tracks[id_map[node]], golgi_tracks[id_map[k]]
                crosscor_m[node, k] = max(crosscor(me, other, binsize=binsize, step=step, slide=5))
        with open("goc_netw_crosscorr.pkl", "wb") as f:
            pickle.dump((netw_dist, crosscor_m), f)
    else:
        with open("goc_netw_crosscorr.pkl", "rb") as g:
            netw_dist, crosscor_m = pickle.load(g)

    dmax = np.max(netw_dist)
    x = np.linspace(0, dmax, 100)
    w = 0.2
    avg = []
    sd = []
    for p in x:
        p_select = crosscor_m[(netw_dist >= p - w) & (netw_dist < p + w)]
        avg.append(np.mean(p_select))
        sd.append(np.std(p_select))

    pathss = nx.shortest_path(G)
    steps = np.empty((L, L))
    for node, paths in pathss.items():
        for P, path in paths.items():
            steps[node, P] = len(path)
    R = [*range(1, int(np.max(steps) + 1))]
    bar_groups = [
        crosscor_m[steps == i]
        for i in R
    ]
    bar_y = [np.mean(group) for group in bar_groups]
    bar_err = [np.std(group) for group in bar_groups]

    return {
        "like_figure": go.Figure(go.Scatter(x=np.arange(-prelim_vars[2], prelim_vars[2], prelim_vars[0]), y=zscore(like_fig))),
        "art_hit": go.Figure(go.Scatter(x=np.arange(-slide, slide, step), y=zscore(art_hit))),
        "art_miss": go.Figure(go.Scatter(x=np.arange(-slide, slide, step), y=zscore(art_miss))),
        "direct": go.Figure(go.Scatter(x=np.arange(-slide, slide, step), y=z)),
        "indirect": go.Figure(go.Scatter(x=np.arange(-slide, slide, step), y=z_long)),
        "relation": go.Figure([
            go.Scatter(x=netw_dist.ravel(), y=crosscor_m.ravel(), mode="markers"),
            go.Scatter(x=x, y=avg),
        ]),
        "steps": go.Figure(
            go.Scatter(
                x=[r - 1 for r in R],
                y=bar_y,
                error_y=dict(type="data", array=bar_err, visible=True),
            )
        ),
    }
