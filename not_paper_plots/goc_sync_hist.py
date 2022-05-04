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

if not os.path.exists("golgi_tracks.pkl"):
    with h5py.File("results/golgi_spike_example.hdf5", "r") as f:
        golgi_tracks = {g.attrs["cell_id"]: (x := g[()][:, 1])[x > 5500] for g in f["recorders/soma_spikes"].values() if g.attrs["label"] == "golgi_cell"}
        with open("golgi_tracks.pkl", "wb") as g:
            pickle.dump(golgi_tracks, g)
else:
    with open("golgi_tracks.pkl", "rb") as g:
        golgi_tracks = pickle.load(g)

def close_diffs(a, b, diff=5):
    sq_diff = np.tile(b, (len(a), 1)) - a.reshape(-1, 1)
    return np.ravel(sq_diff[(sq_diff < diff) & (sq_diff > -diff)])

def plot():
    binsize = 0.1
    width = 5
    diffs = []
    for mid, mtrack in golgi_tracks.items():
        for oid, otrack in golgi_tracks.items():
            if mid == oid:
                continue
            diffs.append(close_diffs(mtrack, otrack))
    diffs = np.concatenate(diffs)



    return go.Figure(go.Histogram(x=diffs))
