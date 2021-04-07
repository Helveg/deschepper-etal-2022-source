import os, plotly.graph_objects as go
from plotly.subplots import make_subplots
from bsb.core import from_hdf5
from bsb.plotting import (
    plot_network,
    MorphologyScene,
    set_scene_range,
    get_soma_trace,
)
from bsb.output import MorphologyRepository
import numpy as np
from ._paths import *
from glob import glob
import selection

def plot(net_path=None):
    if net_path is None:
        net_path = network_path(selection.network)
    network = from_hdf5(net_path)
    multipanel = make_subplots(
        rows=2,
        cols=2,
        specs=[
            [{"type": "scene", "rowspan": 2}, {"type": "scene"}],
            [None, {"type": "scene"}],
        ],
    )
    multipanel.update_layout(width=1200, height=800)
    # Generate top left panel with network somas plotted
    fig = plot_network(network, from_memory=False, show=False)
    for d in fig.data:
        multipanel.add_trace(d, row=1, col=1)
    # Generate the top right panel with a Golgi cell and some granules.
    fig = granular_layer_scene(network)
    for d in fig.data:
        multipanel.add_trace(d, row=1, col=2)
    # Generate the bottom left panel with all Purkinje cells and some granules.
    fig = purkinje_layer_scene(network)
    for d in fig.data:
        multipanel.add_trace(d, row=2, col=2)
    set_scene_range(multipanel.layout.scene3, [[-100, 200], [50, 350], [-100, 200]])
    return multipanel


def granular_layer_scene(network, golgis=1, granules=20):
    ms = MorphologyScene()
    mr = network.morphology_repository
    skip = [
        "glomerulus",
        "basket_cell",
        "stellate_cell",
        "purkinje_cell",
        "mossy_fibers",
    ]
    count = {"golgi_cell": golgis, "granule_cell": granules}
    for cell_type in network.configuration.cell_types.values():
        if cell_type.name in skip:
            continue
        segment_radius = 2.5
        if cell_type.name == "granule_cell":
            segment_radius = 1.0
        positions = np.random.permutation(
            network.get_placement_set(cell_type).positions
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


def purkinje_layer_scene(network, purkinjes=8, granules=200):
    ms = MorphologyScene()
    mr = network.morphology_repository
    skip = ["glomerulus", "basket_cell", "stellate_cell", "golgi_cell", "mossy_fibers"]
    spacing = {"purkinje_cell": purkinjes, "granule_cell": granules}
    for cell_type in network.configuration.cell_types.values():
        if cell_type.name in skip:
            continue
        segment_radius = 2.5
        if cell_type.name == "granule_cell":
            segment_radius = 1.0
        positions = network.get_placement_set(cell_type).positions[
            :: spacing[cell_type.name]
        ]
        morpho = mr.get_morphology(cell_type.list_all_morphologies()[0])
        for cell_pos in positions:
            ms.add_morphology(
                morpho,
                cell_pos,
                color=cell_type.plotting.color,
                soma_radius=cell_type.placement.soma_radius,
                segment_radius=segment_radius,
                reduce_branches=True,
            )
        if cell_type.name == "purkinje_cell":
            positions = network.get_placement_set(cell_type).positions
            for cell_pos in positions:
                ms.fig.add_trace(
                    get_soma_trace(
                        cell_type.placement.soma_radius,
                        cell_pos,
                        "rgba(150, 150, 150, 1)",
                    )
                )
    ms.prepare_plot()
    return ms.fig
