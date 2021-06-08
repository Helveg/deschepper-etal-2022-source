from bsb.core import from_hdf5
from scipy.spatial import KDTree
from plotly import graph_objs as go
import numpy as np
from ._paths import *
from glob import glob
import selection
import scipy.spatial.distance

def plot(net_path=None):
    if net_path is None:
        net_path = network_path(selection.network)
    network = from_hdf5(net_path)

    figs = {}
    for ct in network.get_cell_types(entities=False):
        if ct.name == "granule_cell":
            continue
        ps = network.get_placement_set(ct)
        print("Looking into", ps.tag)
        dists = scipy.spatial.distance.pdist(ps.positions)
        figs[ps.tag] = go.Figure(go.Histogram(x=dists), layout=dict(title_text=ps.tag, xaxis_range=[0, np.max(dists)]))

    return figs
