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
    return molecular_scene(scaffold)


def molecular_scene(scaffold, basket=3, stellate=3, granule=10):
    ms = MorphologyScene()
    mr = MorphologyRepository(file=test_path)
    skip = [
        "glomerulus",
        "golgi_cell",
        "purkinje_cell",
        "mossy_fibers",
    ]
    count = {"basket_cell": basket, "stellate_cell": stellate, "granule_cell": granule}
    for cell_type in scaffold.configuration.cell_types.values():
        if cell_type.name in skip:
            continue
        segment_radius = 2.5
        positions = np.random.permutation(
            scaffold.get_placement_set(cell_type).positions
        )[: count[cell_type.name]]
        morpho = mr.get_morphology(cell_type.list_all_morphologies()[0])
        for cell_pos in positions:
            ms.add_morphology(
                morpho,
                cell_pos,
                color=cell_type.plotting.color,
                soma_radius=cell_type.placement.soma_radius,
                segment_radius=segment_radius,
            )
    ms.prepare_plot()
    return ms.fig
