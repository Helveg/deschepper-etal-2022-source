import os, plotly.graph_objects as go
from bsb.core import from_hdf5
from bsb.plotting import plot_network
from ._paths import *
from glob import glob
import selection

def plot(net_path=None):
    if net_path is None:
        net_path = network_path(selection.network)
    network = from_hdf5(net_path)
    fig = plot_network(network, from_memory=False, show=False)
    for axis in ("x", "y", "z"):
        getattr(fig.layout.scene, axis + "axis").visible = False
    return fig

def meta():
    return {"width": 800, "height": 800}
