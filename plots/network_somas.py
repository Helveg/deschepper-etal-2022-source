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
    fig.layout.scene.xaxis.tick0=0
    fig.layout.scene.xaxis.dtick=150
    fig.layout.scene.yaxis.tick0=0
    fig.layout.scene.yaxis.dtick=150
    fig.layout.scene.zaxis.tick0=0
    fig.layout.scene.zaxis.dtick=150
    return fig
