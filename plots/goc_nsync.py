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

def coincident(a, b, diff=5):
    return np.any(np.abs(np.tile(b, (len(a), 1)) - a.reshape(-1, 1)) <= diff, axis=1)

if not os.path.exists("golgi_tracks.pkl"):
    with h5py.File("results/golgi_spike_example.hdf5", "r") as f:
        golgi_tracks = {g.attrs["cell_id"]: (x := g[()][:, 1])[(x > 5500) & (x < 6000) | (x > 6500)] for g in f["recorders/soma_spikes"].values() if g.attrs["label"] == "golgi_cell"}
        with open("golgi_tracks.pkl", "wb") as g:
            pickle.dump(golgi_tracks, g)
else:
    with open("golgi_tracks.pkl", "rb") as g:
        golgi_tracks = pickle.load(g)

if not os.path.exists("golgi_gko_tracks.pkl"):
    with h5py.File("results/results_gap_knockout.hdf5", "r") as f:
        golgi_gko_tracks = {g.attrs["cell_id"]: (x := g[()][:, 1])[(x > 5500) & (x < 6000) | (x > 6500)] for g in f["recorders/soma_spikes"].values() if g.attrs["label"] == "golgi_cell"}
        with open("golgi_gko_tracks.pkl", "wb") as g:
            pickle.dump(golgi_gko_tracks, g)
else:
    with open("golgi_gko_tracks.pkl", "rb") as g:
        golgi_gko_tracks = pickle.load(g)

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

def plot():
    dist = 100
    bin_widths = np.arange(0, 5.5, 0.5)
    ps = from_hdf5("networks/balanced.hdf5").get_placement_set("golgi_cell")
    ps_pos = ps.positions
    selected = skip_self(distance_matrix(ps_pos, ps_pos) < dist)
    if not os.path.exists("golgi_nsync.pkl"):
        # Spoof data for reference to uniformly random baseline
        fake_tracks = {gid: random.random(len(track)) * 2000 for gid, track in golgi_tracks.items()}
        pos = {id: p for id, p in zip(ps.identifiers, ps_pos)}
        co = {(bw, dist): coincidence_matrix(golgi_tracks, bw, skip_self(selected)) for bw in bin_widths}
        koco = {(bw, dist): coincidence_matrix(golgi_tracks, bw, include_self(selected)) for bw in bin_widths}
        fco = {(bw, dist): coincidence_matrix(fake_tracks, bw, skip_self(selected)) for bw in bin_widths}
        with open("golgi_nsync.pkl", "wb") as g:
            pickle.dump((co, koco, fco), g)
    else:
        with open("golgi_nsync.pkl", "rb") as g:
            co, koco, fco = pickle.load(g)

    return go.Figure(
        [
            go.Scatter(
                x=list(ck[0] for ck in co.keys()),
                y=[np.sum(c[:, :, 0]) / np.sum(c[:, :, 1]) for c in co.values()],
                error_y=dict(
                    type="data",
                    array=[np.nanstd(c[:, :, 0] / c[:, :, 1]) / np.sqrt(len(np.nonzero(c[:, :, 1])[0])) for c in co.values()],
                ),
                name="Coupled",
                legendgroup="coupled",
                line_color="#332EBC",
            ),
            go.Scatter(
                x=list(ck[0] for ck in koco.keys()),
                y=[np.sum(c[:, :, 0]) / np.sum(c[:, :, 1]) for c in koco.values()],
                error_y=dict(
                    type="data",
                    array=[np.nanstd(c[:, :, 0] / c[:, :, 1]) / np.sqrt(len(np.nonzero(c[:, :, 1])[0])) for c in koco.values()],
                ),
                name="Knockout",
                legendgroup="ko",
                line_color="#DC143C",
            ),
            go.Scatter(
                x=list(ck[0] for ck in fco.keys()),
                y=[np.sum(c[:, :, 0]) / np.sum(c[:, :, 1]) for c in fco.values()],
                error_y=dict(
                    type="data",
                    array=[np.nanstd(c[:, :, 0] / c[:, :, 1]) / np.sqrt(len(np.nonzero(c[:, :, 1])[0])) for c in fco.values()],
                ),
                name="Random",
                legendgroup="rand",
                line_color="black",
            )
        ],
        layout=dict(
            yaxis_tickformat = '.0%',
            yaxis_title="% Coincident spikes",
            xaxis_title="Time lag window (±ms)",
            title_text="Golgi cell spike coincidence"
        )
    )
