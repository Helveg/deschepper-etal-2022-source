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
    fig = purkinje_layer_scene(network)
    set_scene_range(fig.layout.scene, [[-50, 310], [0, 350], [-50, 250]])
    fig.layout.scene.xaxis.tick0=0
    fig.layout.scene.xaxis.dtick=150
    fig.layout.scene.yaxis.tick0=0
    fig.layout.scene.yaxis.dtick=150
    fig.layout.scene.zaxis.tick0=0
    fig.layout.scene.zaxis.dtick=150
    return fig

def purkinje_layer_scene(network, purkinjes=6, granules=100):
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
