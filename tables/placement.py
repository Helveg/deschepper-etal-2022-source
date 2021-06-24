from bsb.core import from_hdf5
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import plotly.graph_objects as go
import h5py, itertools, glob
from plots._paths import *
import plots.selection as selection

def table(path=None, net_path=None, cutoff=4000, duration=8000):
    if path is None:
        path = network_path(selection.network)
    for p in glob.glob(path)[:1]:
        tracker = dict()
        network = from_hdf5(p)
        for k,v in network.statistics.cells_placed.items():
            tracker.setdefault(k, []).append(v)
    pop_n = list(zip(tracker.keys(), map(np.mean, tracker.values()), map(np.std, tracker.values())))
    pop_n.insert(0, ("cell_name", "mean", "stdev"))
    return pop_n
