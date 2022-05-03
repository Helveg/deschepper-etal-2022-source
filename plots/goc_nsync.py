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
        golgi_tracks = {g.attrs["cell_id"]: (x := g[()][:, 1])[x > 5500] for g in f["recorders/soma_spikes"].values() if g.attrs["label"] == "golgi_cell"}
        with open("golgi_tracks.pkl", "wb") as g:
            pickle.dump(golgi_tracks, g)
else:
    with open("golgi_tracks.pkl", "rb") as g:
        golgi_tracks = pickle.load(g)

def coincidence_matrix(tracks, diff, skip_self=True):
    co = np.zeros((len(tracks), len(tracks), 2))
    for gid, track in enumerate(tracks.values()):
        for ogid, otrack in enumerate(tracks.values()):
            if skip_self and gid == ogid:
                # Skip diagonal (self)
                continue
            co[gid, ogid, :] = (sum(coincident(track, otrack, diff)), len(track))

    return co

def plot():
    if not os.path.exists("golgi_nsync.pkl"):
        dists = np.arange(0, 5.5, 0.5)
        fake_tracks = {gid: random.random(len(track)) * 2800 for gid, track in golgi_tracks.items()}
        co = {d: coincidence_matrix(golgi_tracks, d) for d in dists}
        nsco = {d: coincidence_matrix(golgi_tracks, d, skip_self=False) for d in dists}
        fco = {d: coincidence_matrix(fake_tracks, d) for d in dists}
        with open("golgi_nsync.pkl", "wb") as g:
            pickle.dump((co, nsco, fco), g)
    else:
        with open("golgi_nsync.pkl", "rb") as g:
            co, nsco, fco = pickle.load(g)

    total = go.Figure(
        [
            go.Scatter(
                x=list(co.keys()),
                y=[np.sum(c[:, :, 0]) / np.sum(c[:, :, 1]) for c in co.values()],
                name="Results"
            ),
            go.Scatter(
                x=list(nsco.keys()),
                y=[np.sum(c[:, :, 0]) / np.sum(c[:, :, 1]) for c in nsco.values()],
                name="Noskip"
            ),
            go.Scatter(
                x=list(fco.keys()),
                y=[np.sum(c[:, :, 0]) / np.sum(c[:, :, 1]) for c in fco.values()],
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
    return {
        "total": total
    }
