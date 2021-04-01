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


camera = dict(up=dict(x=0,y=0,z=1),center=dict(x=0.014630658146720252,y=0.02534642697221434,z=-0.11515199306852882),eye=dict(x=9.88018501112187,y=0.14634744535299443,z=0.5590972887418161))
net_cam = dict(up=dict(x=0,y=0,z=1),center=dict(x=0.9516507843396513,y=-0.7480738883245212,z=-0.48851824490599444),eye=dict(x=7.556075851557944,y=6.546386629756536,z=2.3610400567393985))

def plot(net_path=None):
    if net_path is None:
        net_path = network_path(selection.network)
    network = from_hdf5(net_path)
    fig = network_scene(network)
    fig.layout.scene.xaxis.range = [-100, 300]
    fig.layout.scene.yaxis.range = [-100, 200]
    fig.layout.scene.zaxis.range = [-100, 400]
    fig.layout.scene.aspectmode="manual"
    fig.layout.scene.aspectratio=dict(x=4, y=3, z=5)
    fig.layout.scene.xaxis.tick0=0
    fig.layout.scene.xaxis.dtick=100
    fig.layout.scene.yaxis.tick0=0
    fig.layout.scene.yaxis.dtick=100
    fig.layout.scene.zaxis.tick0=0
    fig.layout.scene.zaxis.dtick=100
    fig2 = go.Figure(fig)
    fig.update_layout(scene_camera=net_cam)
    fig2.update_layout(scene_camera=camera)
    return [fig, fig2]

def network_scene(network):
    ms = MorphologyScene()
    mr = network.morphology_repository
    skip = ["glomerulus", "mossy_fibers"]
    for cell_type in network.configuration.cell_types.values():
        if cell_type.name in skip:
            continue
        segment_radius = 1.0
        count = 2
        if cell_type.name != "granule_cell":
            segment_radius = 2.5
        else:
            count = 10
        positions = np.random.permutation(
            network.get_cells_by_type(cell_type.name)[:, 2:5]
        )[:count]
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
