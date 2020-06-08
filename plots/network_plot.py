import os, plotly.graph_objects as go
from plotly.subplots import make_subplots
from scaffold.core import from_hdf5
from scaffold.plotting import (
    plot_network,
    MorphologyScene,
    set_scene_range,
    get_soma_trace,
)
from scaffold.output import MorphologyRepository
import numpy as np

test_path = os.path.join(
    os.path.dirname(__file__), "..", "networks", "neuron.hdf5"
)


def plot():
    scaffold = from_hdf5(test_path)
    return plot_network(scaffold, from_memory=False, show=False)
