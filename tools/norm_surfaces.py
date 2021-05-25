from bsb.core import from_hdf5
import os, sys, h5py, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "plots"))
import plotly.graph_objs as go, scipy.stats
import pickle
from _paths import *
from glob import glob

net_path = sys.argv[1] if len(sys.argv) > 1 else None

if net_path is None:
    net_path = os.path.abspath(network_path("batch_1", "*.hdf5"))
paths = glob(net_path)
surfaces = {}
for path in paths:
    network = from_hdf5(path)
    id = int(path.split("_")[-1].split(".")[0])
    print("Creating normalization map for network", id)
    ps_grc = network.get_placement_set("granule_cell")
    points = ps_grc.positions[:, [0, 2]]
    grid_offset = np.array([0.0, 0.0])  # x z
    grid_spacing = np.array([5., 5.])  #um
    gpoints = np.round((points - grid_offset) / grid_spacing).astype(int)
    coords, counts = np.unique(gpoints, axis=0, return_counts=True)
    surface = np.ones((int(np.max(coords[:, 0]) + 1), int(np.max(coords[:, 1]) + 1)))
    surface[tuple(coords.T)] = counts
    surfaces[id] = surface

with open("pkl_ca/norms.pickle", "wb") as f:
    pickle.dump(surfaces, f)
