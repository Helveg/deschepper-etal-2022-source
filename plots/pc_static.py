from bsb.core import from_hdf5
from bsb.plotting import plot_morphology #, plot_synapses
from plotly import graph_objs as go
import selection, numpy as np
from colour import Color
from ._paths import *
from glob import glob
import selection

def plot(net_path=None):
    if net_path is None:
        net_path = network_path(selection.network)
    network = from_hdf5(net_path)
    mr = network.morphology_repository
    m = mr.get_morphology("PurkinjeCell")
    fig = plot_morphology(m, show=False)
    return fig

if __name__ == "__main__":
    plot()
