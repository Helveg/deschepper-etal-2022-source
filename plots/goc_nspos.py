from scipy.spatial import distance_matrix
import numpy as np
import plotly.graph_objs as go
import h5py
from scipy.stats import zscore, gaussian_kde
import os
import pickle
import networkx as nx
from bsb.core import from_hdf5
from scipy.sparse import coo_matrix
from scipy.spatial import distance_matrix
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
    if not os.path.exists("golgi_npos.pkl"):
        dists = np.arange(0, 5.5, 0.5)
        fake_tracks = {gid: random.random(len(track)) * 2500 for gid, track in golgi_tracks.items()}
        co = {d: coincidence_matrix(golgi_tracks, d) for d in dists}
        nsco = {d: coincidence_matrix(golgi_tracks, d, skip_self=False) for d in dists}
        fco = {d: coincidence_matrix(fake_tracks, d) for d in dists}
        with open("golgi_npos.pkl", "wb") as g:
            pickle.dump((co, nsco, fco), g)
    else:
        with open("golgi_npos.pkl", "rb") as g:
            co, nsco, fco = pickle.load(g)

    netw = from_hdf5("networks/balanced.hdf5")
    gpos = netw.get_placement_set("golgi_cell").positions
    subtractor = gpos.reshape(-1, 1, 3)
    gdist = np.tile(gpos, (len(gpos), 1, 1)) - subtractor
    relation = gdist.ravel()
    coupling = co[5.].ravel()
    x = np.linspace(-300, 300, 50)
    y = np.linspace(-300, 300, 50)
    surf_coupled = gaussian_kde(np.vstack((relation[::3], relation[2::3])), weights=coupling[::2])
    surf_total = gaussian_kde(np.vstack((relation[::3], relation[2::3])), weights=coupling[1::2])
    nonz_mask = coupling[1::2] != 0
    nonz_x = relation[::3][nonz_mask]
    nonz_y = relation[2::3][nonz_mask]
    surf_div = gaussian_kde(np.vstack((nonz_x, nonz_y)), weights=coupling[::2][nonz_mask] / coupling[1::2][nonz_mask])

    raster = np.meshgrid(x, y)
    coords = np.vstack((raster[0].ravel(), raster[1].ravel()))

    total = go.Figure(
        [
            go.Surface(
                x=x,
                y=y,
                z=(surf_div(coords)).reshape((50,50)),
            ),
        ]
    )
    scattered = go.Figure(
        [
            go.Scatter3d(
                x=relation[::3],
                y=relation[2::3],
                z=coupling[::2] / coupling[1::2],
                marker_size=2,
                name="Results",
                mode="markers",
            ),
        ],
        layout=dict(
            scene_zaxis_tickformat = '.0%',
            scene_zaxis_title="% Coincident spikes with center",
            scene_xaxis_title="X",
            scene_yaxis_title="Z",
            title_text="Positional spike coupling"
        )
    )
    return {
        "surf": total,
        "scattered": scattered
    }
