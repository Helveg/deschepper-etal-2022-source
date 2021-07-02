import os, sys, plotly.graph_objects as go
from bsb.core import from_hdf5
from bsb.plotting import plot_network
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from plots._paths import *
from glob import glob
import selection

def plot(population, net_path=None):
    if net_path is None:
        net_path = network_path(selection.network)
    network = from_hdf5(net_path)
    pos = network.get_placement_set(population).positions[()][:, [0, 2]]
    ids = network.get_placement_set(population).identifiers
    fig = go.Figure(go.Scatter(x=pos[:,0], y=pos[:, 1], text=ids, mode="markers"))
    return fig
