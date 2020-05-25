import os, plotly.graph_objects as go
from scaffold.core import from_hdf5
from scaffold.plotting import plot_network

test_path = os.path.join(os.path.dirname(__file__), "..", "networks", "preliminary_test.hdf5")

def plot():
    scaffold = from_hdf5(test_path)
    fig = plot_network(scaffold, from_memory=False, show=False)
    return fig
