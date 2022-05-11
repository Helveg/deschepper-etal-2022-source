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

def plot(pkl="goc_sync_", track_result="results/golgi_spike_example.hdf5", track_pkl="golgi_tracks.pkl", track_ko_result="results/results_gap_knockout.hdf5", track_ko_pkl="golgi_gko_tracks.pkl"):
    if not os.path.exists(track_pkl):
        with h5py.File(track_result, "r") as f:
            golgi_tracks = {g.attrs["cell_id"]: (x := g[()][:, 1])[(x > 5500) & (x < 6000) | (x > 6500)] for g in f["recorders/soma_spikes"].values() if g.attrs["label"] == "golgi_cell"}
        with open(track_pkl, "wb") as g:
            pickle.dump(golgi_tracks, g)
    else:
        with open(track_pkl, "rb") as g:
            golgi_tracks = pickle.load(g)

    if not os.path.exists(track_ko_pkl):
        with h5py.File(track_ko_result, "r") as f:
            golgi_gko_tracks = {g.attrs["cell_id"]: (x := g[()][:, 1])[(x > 5500) & (x < 6000) | (x > 6500)] for g in f["recorders/soma_spikes"].values() if g.attrs["label"] == "golgi_cell"}
        with open(track_ko_pkl, "wb") as g:
            pickle.dump(golgi_gko_tracks, g)
    else:
        with open(track_ko_pkl, "rb") as g:
            golgi_gko_tracks = pickle.load(g)
    zpkl = pkl + "zscore.pkl"
    ccpkl = pkl + "cross.pkl"
    step = 0.5
    binsize = 5
    binsteps = int(binsize / 2 // step)
    slide = 100
    lag0 = int(slide // step)
    netw = from_hdf5("networks/balanced.hdf5")
    G = goc_graph(netw)
    id_map = dict(zip(G.nodes(), itertools.count()))
    L = len(G.nodes())
    if not os.path.exists(zpkl):
        netw_dist = np.empty((L, L))
        zscore_m = np.empty((L, L))
        zscore_mko = np.empty((L, L))
        cc_m = None
        for node, paths in nx.shortest_path_length(G, weight="weight"):
            print("It", node)
            for k, d in paths.items():
                netw_dist[node, k] = d
                me, other = golgi_tracks[id_map[node]], golgi_tracks[id_map[k]]
                cross = crosscor(me, other, binsize=binsize, step=step, slide=slide)
                z = zscore(cross)
                zscore_m[node, k] = max(z[(lag0 - binsteps):(lag0 + binsteps)])
                meko, otherko = golgi_gko_tracks[id_map[node]], golgi_gko_tracks[id_map[k]]
                crossko = crosscor(meko, otherko, binsize=binsize, step=step, slide=slide)
                if cc_m is None:
                    cc_m = np.empty((L, L, len(crossko)))
                cc_m[node, k, :] = crossko
                zko = zscore(crossko)
                zscore_mko[node, k] = max(zko[(lag0 - binsteps):(lag0 + binsteps)])
        with open(zpkl, "wb") as f:
            pickle.dump((netw_dist, zscore_m, zscore_mko), f)
        with open(ccpkl, "wb") as f:
            pickle.dump((netw_dist, cc_m), f)
    else:
        with open(zpkl, "rb") as g:
            netw_dist, zscore_m, zscore_mko = pickle.load(g)

    def trend(x, y, w=0.2, tw=0.03):
        xmax = np.max(x)
        xmin = np.min(x)
        fx = np.linspace(xmin, xmax, 100)
        avg = []
        sd = []
        for p in fx:
            p_select = y[(x >= p - w) & (x < p + w)]
            avg.append(np.nanmean(p_select))
            sd.append(np.nanstd(y[(x >= p - tw) & (x < p + tw)]))
        return fx, np.array(avg), np.array(sd)

    pathss = nx.shortest_path(G)
    steps = np.empty((L, L))
    for node, paths in pathss.items():
        for P, path in paths.items():
            steps[node, P] = len(path)
    R = [*range(1, int(np.max(steps) + 1))]
    bar_groups = [
        zscore_m[steps == i]
        for i in R
    ]
    bar_groupsko = [
        zscore_mko[steps == i]
        for i in R
    ]
    bar_y = [np.nanmean(group) for group in bar_groups]
    bar_err = [np.nanstd(group) for group in bar_groups]

    mask = netw_dist != 0
    pos = netw.get_placement_set("golgi_cell").positions
    pdist = distance_matrix(pos, pos)
    pdr = pdist[mask].ravel()
    zsmr = zscore_m[mask].ravel()
    print("MK test of distance:", mk.original_test(zsmr[np.argsort(pdr)]))

    mk_medians = [np.median(zscore_m[netw_dist == d]) for d in np.sort(np.unique(netw_dist))]
    print("MK test of relationship:", mk.original_test(mk_medians))
    bar_y = [np.nanmedian(group) for group in bar_groups[1:-1]]
    print("MK test of steps:", mk.original_test(bar_y))

    tnd_x, tnd_y, tnd_err = trend(netw_dist[mask], zscore_m[mask])
    ted_x, ted_y, ted_err = trend(pdist[mask], zscore_m[mask], w=10, tw=10)

    #KO

    zsmrko = zscore_mko[mask].ravel()
    print("MK test of distance:", mk.original_test(zsmrko[np.argsort(pdr)]))

    mk_medians = [np.median(zscore_mko[netw_dist == d]) for d in np.sort(np.unique(netw_dist))]
    print("MK test of relationship:", mk.original_test(mk_medians))
    bar_yko = [np.nanmedian(group) for group in bar_groupsko[1:-1]]
    print("MK test of steps:", mk.original_test(bar_yko))
    bar_errko = [np.nanstd(group) for group in bar_groupsko]

    tndko_x, tndko_y, tndko_err = trend(netw_dist[mask], zscore_mko[mask])
    tedko_x, tedko_y, tedko_err = trend(pdist[mask], zscore_mko[mask], w=10, tw=10)

    fig = make_subplots(cols=3, rows=1)
    fig.update_layout(title_text="Effect of gap junction coupling on Golgi cell synchrony")
    for p, traces in enumerate(
        (
            [
                go.Scatter(x=pdr, y=zsmr, mode="markers"),
                go.Scatter(x=ted_x, y=ted_y + ted_err, line_width=0),
                go.Scatter(x=ted_x, y=ted_y, fillcolor="rgba(52, 235, 177, 0.3)", name="Trend euclid dist"),
                go.Scatter(x=ted_x, y=ted_y - ted_err, line_width=0, fillcolor="rgba(52, 235, 177, 0.3)"),
                go.Scatter(x=tedko_x, y=tedko_y + tedko_err, line_width=0),
                go.Scatter(x=tedko_x, y=tedko_y, fillcolor="rgba(52, 235, 177, 0.3)", name="Trend euclid dist KO"),
                go.Scatter(x=tedko_x, y=tedko_y - tedko_err, line_width=0, fillcolor="rgba(52, 235, 177, 0.3)"),
            ],
            [
                go.Scatter(x=netw_dist[mask].ravel(), y=zsmr, mode="markers"),
                go.Scatter(x=tnd_x, y=tnd_y + tnd_err, line_width=0),
                go.Scatter(x=tnd_x, y=tnd_y, fillcolor="rgba(52, 235, 177, 0.3)", name="Trend net dist"),
                go.Scatter(x=tnd_x, y=tnd_y - tnd_err, line_width=0, fillcolor="rgba(52, 235, 177, 0.3)"),
                go.Scatter(x=tndko_x, y=tndko_y + tndko_err, line_width=0),
                go.Scatter(x=tndko_x, y=tndko_y, fillcolor="rgba(52, 235, 177, 0.3)", name="Trend net dist KO"),
                go.Scatter(x=tndko_x, y=tndko_y - tndko_err, line_width=0, fillcolor="rgba(52, 235, 177, 0.3)"),
            ],
            [
                go.Scatter(
                    x=[r - 1 for r in R[1:-1]],
                    y=bar_y,
                    error_y=dict(type="data", array=bar_err[1:-1], visible=True),
                ),
                go.Scatter(
                    x=[r - 1 for r in R[1:-1]],
                    y=bar_yko,
                    error_y=dict(type="data", array=bar_errko[1:-1], visible=True),
                )
            ]
        )
    ):
        for t in traces:
            fig.add_trace(t, row=1, col=p+1)

    for p, (xaxis_title, yaxis_title) in enumerate(
        (
            ("Distance (μm)", "Max. crosscorrelation z-score"),
            ("Electrotonic distance", "Max. crosscorrelation z-score"),
            ("Shortest path length", "Median of max. crosscorrelation z-score"),
        )
    ):
        if xaxis_title is not None:
            fig.update_xaxes(title=xaxis_title, row=1, col=p+1)
        if yaxis_title is not None:
            fig.update_yaxes(title=yaxis_title, row=1, col=p+1)
    fig.update_xaxes(dtick=1, row=1, col=3)

    return fig
