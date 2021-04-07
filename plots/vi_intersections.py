from bsb.core import Scaffold, from_hdf5
from bsb.plotting import plot_morphology, plot_voxel_cloud
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from . import make_3dsubplots
from random import sample
import os, numpy as np, selection
from colour import Color
from ._paths import *
from glob import glob
import selection

camera = dict(up=dict(x=0,y=0,z=1),center=dict(x=-0.0019080692534999733,y=-0.14007415041927948,z=0.09663920838797366),eye=dict(x=0.6488293759298915,y=0.19131033010199103,z=0.26284021600913543))

def plot(net_path=None):
    if net_path is None:
        net_path = network_path(selection.network)
    network = from_hdf5(net_path)
    connection_name = "stellate_to_purkinje"
    ct = network.get_connection_type(connection_name)
    cs = network.get_connectivity_set(ct.name)
    connections = cs.connections
    from_type = ct.from_cell_types[0]
    to_type = ct.to_cell_types[0]
    ps_pre = network.get_placement_set(from_type)
    ps_post = network.get_placement_set(to_type)
    from_comp_types = ct.from_cell_compartments[0]
    to_comp_types = ct.to_cell_compartments[0]

    pair = sample(connections, 1)[0]
    from_id = selection.vi_stellate
    to_id = selection.vi_purkinje
    from_id = pair.from_id
    to_id = pair.to_id
    m_pre = network.morphology_repository.get_morphology(from_type.list_all_morphologies()[0])
    m_pre2 = network.morphology_repository.get_morphology(from_type.list_all_morphologies()[0])
    m_post = network.morphology_repository.get_morphology(to_type.list_all_morphologies()[0])
    m_pre.voxelize(50, compartments=m_pre.get_compartments(from_comp_types))
    m_pre2.voxelize(150)
    m_post.voxelize(50, compartments=m_post.get_compartments(to_comp_types))
    pos_pre = ps_pre.positions[int(from_id) - ps_pre.identifiers[0]]
    pos_post = ps_post.positions[int(to_id) - ps_post.identifiers[0]]

    intersections = ct.intersect_clouds(m_pre.cloud, m_post.cloud, pos_pre, pos_post)
    from_voxels = set()
    to_voxels = []
    for i, tv in enumerate(intersections):
        if tv:
            to_voxels.append(i)
            from_voxels.update(tv)

    from_voxels = list(from_voxels)
    selected_voxels_pre = tuple(np.array(d[from_voxels]) for d in np.nonzero(m_pre.cloud.voxels))
    selected_voxels_post = tuple(np.array(d[to_voxels]) for d in np.nonzero(m_post.cloud.voxels))

    from_colors = list(map(str, Color(from_type.plotting.color).range_to("black", 10)))
    fcd = {"soma": from_colors[0], "dendrites": from_colors[0], "axon": from_colors[1]}
    m_pre.cloud.voxels = np.zeros(m_pre.cloud.voxels.shape, dtype=bool)
    m_pre.cloud.voxels[selected_voxels_pre] = True
    m_post.cloud.voxels = np.zeros(m_post.cloud.voxels.shape, dtype=bool)
    m_post.cloud.voxels[selected_voxels_post] = True
    fig4 = plot_morphology(m_pre, offset=pos_pre, show=False, segment_radius={"axon": 6, "dendrites": 2.5}, color=fcd, set_range=False, soma_radius=from_type.placement.soma_radius)
    # fig4.layout.scene.yaxis.range = [120, 205]
    plot_voxel_cloud(m_pre.cloud, fig=fig4, selected_voxels=[], offset=pos_pre, show=False, set_range=False, cubic=False)
    plot_morphology(m_post, fig=fig4, offset=pos_post, show=False, segment_radius=4, set_range=False, color=to_type.plotting.color, soma_radius=to_type.placement.soma_radius)
    plot_voxel_cloud(m_post.cloud, fig=fig4, offset=pos_post, selected_voxels=[], show=False, set_range=False, cubic=False)
    intersections = [i for i in cs.intersections if i.from_id == from_id and i.to_id == to_id]
    i_pre = np.array([i.from_compartment.end + pos_pre for i in intersections])
    i_post = np.array([i.to_compartment.end + pos_post for i in intersections])
    fig4.add_trace(go.Scatter3d(x=i_pre[:,0], y=i_pre[:,2], z=i_pre[:,1], mode="markers", marker=dict(size=6,color="violet", line=dict(width=1, color="black"))))
    fig4.add_trace(go.Scatter3d(x=i_post[:,0], y=i_post[:,2], z=i_post[:,1], mode="markers", marker=dict(symbol="diamond-open", size=8,color="violet")))
    # fig4.layout.scene.camera = camera
    return fig4
