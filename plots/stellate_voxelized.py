from scaffold.core import Scaffold, from_hdf5
from scaffold.plotting import plot_morphology, plot_voxel_cloud
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from . import make_3dsubplots
from random import sample
import os, numpy as np

network = os.path.join(os.path.dirname(__file__), "..", "networks", "pizdaint.hdf5")
scaffold = from_hdf5(network)

def plot():
    connection_name = "stellate_to_purkinje"
    ct = scaffold.get_connection_type(connection_name)
    cs = scaffold.get_connectivity_set(ct.name)
    connections = cs.connections
    from_type = ct.from_cell_types[0]
    to_type = ct.to_cell_types[0]
    ps_pre = scaffold.get_placement_set(from_type)

    pair = sample(connections, 1)[0]
    m_pre2 = scaffold.morphology_repository.get_morphology(from_type.list_all_morphologies()[0])
    m_pre2.voxelize(150)
    pos_pre = ps_pre.positions[int(pair.from_id) - ps_pre.identifiers[0]]
    fig = plot_morphology(m_pre2, offset=pos_pre, show=False, segment_radius=2, color=from_type.plotting.color)
    plot_voxel_cloud(m_pre2.cloud, selected_voxels=[], fig=fig, offset=pos_pre, set_range=False, show=False)
    return fig
