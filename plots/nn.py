from bsb.core import from_hdf5
from scipy.spatial import KDTree
from plotly import graph_objs as go
import numpy as np

def plot():
    network = from_hdf5("networks/300x_200z.hdf5")

    figs = {}
    for ct in network.get_cell_types(entities=False):
        ps = network.get_placement_set(ct)
        print("Looking into", ps.tag)
        tree = KDTree(ps.positions)
        matches, ind = tree.query(ps.positions, k=2)
        figs[ps.tag] = go.Figure(go.Histogram(x=matches[:, 1]), layout=dict(title_text=ps.tag, xaxis_range=[0, np.max(matches[:, 1]) + np.min(matches[:, 1])]))

    return figs
