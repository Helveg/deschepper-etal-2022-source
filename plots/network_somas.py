import os, plotly.graph_objects as go
from scaffold.core import from_hdf5
from scaffold.plotting import plot_network

test_path = os.path.join(
    os.path.dirname(__file__), "..", "networks", "preliminary_test.hdf5"
)


def plot():
    scaffold = from_hdf5(test_path)
    fig = plot_network(scaffold, from_memory=False, show=False)
    fig.layout.scene.xaxis.tick0=0
    fig.layout.scene.xaxis.dtick=150
    fig.layout.scene.yaxis.tick0=0
    fig.layout.scene.yaxis.dtick=150
    fig.layout.scene.zaxis.tick0=0
    fig.layout.scene.zaxis.dtick=150
    return fig
