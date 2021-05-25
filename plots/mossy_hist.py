from bsb.core import from_hdf5
from plotly import graph_objs as go
import numpy as np
from ._paths import *
from glob import glob
import selection, itertools

def plot(net_path=None):
    if net_path is None:
        net_path = network_path(selection.network)
    network = from_hdf5(net_path)

    d = dict(zip(network.get_placement_set("mossy_fibers").identifiers, itertools.repeat(0)))
    print(len(d))
    print(len(np.unique(network.get_connectivity_set("mossy_to_glomerulus").get_dataset()[:, 0], return_counts=True)[1]))
    d.update(zip(*np.unique(network.get_connectivity_set("mossy_to_glomerulus").get_dataset()[:, 0], return_counts=True)))
    fig = go.Figure(go.Histogram(
        x=list(d.values()),
        xbins=dict(
            start=0,
            end=45,
            size=1
        ),
        autobinx=False
    ))

    return fig
