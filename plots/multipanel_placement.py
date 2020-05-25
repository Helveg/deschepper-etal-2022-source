import os, plotly.graph_objects as go
from plotly.subplots import make_subplots
from scaffold.core import from_hdf5
from scaffold.plotting import plot_network, MorphologyScene
from scaffold.output import MorphologyRepository
import numpy as np

test_path = os.path.join(os.path.dirname(__file__), "..", "networks", "preliminary_test.hdf5")

def plot():
    scaffold = from_hdf5(test_path)
    multipanel = make_subplots(rows=2, cols=2, specs=[
        [{"type": "scene"}, {"type": "scene"}],
        [{"type": "scene"}, {"type": "scene"}]
    ])
    # Generate top left panel with network somas plotted
    fig = plot_network(scaffold, from_memory=False, show=False)
    for d in fig.data:
        multipanel.add_trace(d, row=1, col=1)
    # Generate the top right panel with a Golgi cell and some ascending axons.
    fig = granular_layer_scene(scaffold)
    for d in fig.data:
        multipanel.add_trace(d, row=1, col=2)
    return multipanel

def granular_layer_scene(scaffold, golgis=1, granules=20):
    ms = MorphologyScene()
    mr = MorphologyRepository(file=test_path)
    skip = ["glomerulus", "basket_cell", "stellate_cell", "purkinje_cell", "mossy_fibers"]
    count = {"golgi_cell": golgis, "granule_cell": granules}
    for cell_type in scaffold.configuration.cell_types.values():
        if cell_type.name in skip:
            continue
        segment_radius = 2.5
        if cell_type.name == "granule_cell":
            segment_radius = 1.0
        positions = np.random.permutation(scaffold.get_placement_set(cell_type).positions)[:count[cell_type.name]]
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
