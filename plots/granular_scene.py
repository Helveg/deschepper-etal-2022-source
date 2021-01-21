import os, plotly.graph_objects as go
from plotly.subplots import make_subplots
from bsb.core import from_hdf5
from bsb.plotting import (
    plot_network,
    plot_morphology,
    MorphologyScene,
    set_scene_range,
    get_soma_trace,
)
from bsb.output import MorphologyRepository
import numpy as np

test_path = os.path.join(
    os.path.dirname(__file__), "..", "networks", "300x_200z.hdf5"
)


def plot():
    scaffold = from_hdf5(test_path)
    fig = granular_layer_scene(scaffold)
    set_scene_range(fig.layout.scene, [[-50, 310], [0, 350], [-50, 250]])
    fig.layout.scene.xaxis.tick0=0
    fig.layout.scene.xaxis.dtick=150
    fig.layout.scene.yaxis.tick0=0
    fig.layout.scene.yaxis.dtick=150
    fig.layout.scene.zaxis.tick0=0
    fig.layout.scene.zaxis.dtick=150
    return fig

def granular_layer_scene(scaffold, golgis=2, granules=100):
    ms = MorphologyScene()
    mr = MorphologyRepository(file=test_path)
    skip = [
        "glomerulus",
        "basket_cell",
        "stellate_cell",
        "purkinje_cell",
        "mossy_fibers",
    ]
    count = {"golgi_cell": golgis, "granule_cell": granules}
    for cell_type in scaffold.configuration.cell_types.values():
        if cell_type.name in skip:
            continue
        segment_radius = 2.5
        if cell_type.name == "granule_cell":
            segment_radius = 1.0
        positions = np.random.permutation(
            scaffold.get_placement_set(cell_type).positions
        )[: count[cell_type.name]]
        morpho = mr.get_morphology(cell_type.list_all_morphologies()[0])
        plot_morphology(morpho)
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
