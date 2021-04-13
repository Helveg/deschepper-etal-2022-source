import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import plotly.graph_objects as go
import h5py, itertools

def table(path=None, net_path=None, cutoff=4000, duration=8000):
    with h5py.File(path, "r") as f:
        tracker = dict()
        for ds in itertools.chain(f["/recorders/soma_spikes"].values(), f["/recorders/input/background"].values()):
            cell_type = ds.attrs.get("label", "mossy_fibers")
            pop_stats = tracker.setdefault(cell_type, [])
            spikes = ds[:, 1] if len(ds.shape) == 2 else ds[()]
            pop_stats.append(sum((spikes > cutoff) & (spikes < duration)) / (duration - cutoff) * 1000)

    go.Figure(go.Histogram(x=tracker.get("golgi_cell"))).show()
    pop_freq = list(zip(tracker.keys(), map(np.mean, tracker.values()), map(np.std, tracker.values())))
    pop_freq.insert(0, ("cell_name", "mean", "stdev"))
    return pop_freq
