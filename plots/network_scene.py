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
    fig = network_scene(scaffold)
    set_scene_range(fig.layout.scene, [[-100, 250], [-100, 350], [-100, 250]])
    fig.layout.scene.aspectmode="manual"
    fig.layout.scene.aspectratio=dict(x=1, y=1, z=450/350)
    fig.layout.scene.xaxis.tick0=0
    fig.layout.scene.xaxis.dtick=150
    fig.layout.scene.yaxis.tick0=0
    fig.layout.scene.yaxis.dtick=150
    fig.layout.scene.zaxis.tick0=0
    fig.layout.scene.zaxis.dtick=150
    return fig

def network_scene(scaffold):
    ms = MorphologyScene()
    mr = MorphologyRepository(file=test_path)
    skip = ["glomerulus", "mossy_fibers"]
    for cell_type in scaffold.configuration.cell_types.values():
        if cell_type.name in skip:
            continue
        segment_radius = 1.0
        count = 2
        if cell_type.name != "granule_cell":
            segment_radius = 2.5
        else:
            count = 10
        positions = np.random.permutation(
            scaffold.get_cells_by_type(cell_type.name)[:, 2:5]
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
