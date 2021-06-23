import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import plotly.graph_objects as go
import h5py, itertools, glob
from plots._paths import *
from bsb.core import from_hdf5
import plots.selection as selection

def table(path=None, net_path=None, start=7500, end=8000):
    if path is None:
        path = glob.glob(results_path("sensory_burst", "*.hdf5"))
    if net_path is None:
        net_path = network_path(selection.network)
    network = from_hdf5(net_path)
    for path in path:
        with h5py.File(path, "r") as f:
            tracker = dict()
            for ds in itertools.chain(f["/recorders/soma_spikes"].values(), f["/recorders/input/background"].values()):
                cell_type = ds.attrs.get("label", "mossy_fibers")
                pop_stats = tracker.setdefault(cell_type, [])
                spikes = ds[:, 1] if len(ds.shape) == 2 else ds[()]
                pop_stats.append(sum((spikes > start) & (spikes < end)) / (end - start) * 1000)

        assert all(len(v) == len(network.get_placement_set(k)) for k,v in tracker.items()), "We are short on cells"
        pop_freq = list(zip(tracker.keys(), map(np.mean, tracker.values()), map(np.std, tracker.values())))
        pop_freq.insert(0, ("cell_name", "mean", "stdev"))
        return pop_freq
