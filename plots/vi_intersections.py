from bsb.core import Scaffold, from_hdf5
from bsb.plotting import plot_morphology, plot_voxel_cloud
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from . import make_3dsubplots
from random import sample
import os, numpy as np

network = os.path.join(os.path.dirname(__file__), "..", "networks", "neuron.hdf5")
scaffold = from_hdf5(network)

def plot():
    connection_name = "stellate_to_purkinje"
    ct = scaffold.get_connection_type(connection_name)
    cs = scaffold.get_connectivity_set(ct.name)
    connections = cs.connections
    from_type = ct.from_cell_types[0]
    to_type = ct.to_cell_types[0]
    ps_pre = scaffold.get_placement_set(from_type)
    ps_post = scaffold.get_placement_set(to_type)
    from_comp_types = ct.from_cell_compartments[0]
    to_comp_types = ct.to_cell_compartments[0]

    pair = sample(connections, 1)[0]
    m_pre = scaffold.morphology_repository.get_morphology(from_type.list_all_morphologies()[0])
    m_pre2 = scaffold.morphology_repository.get_morphology(from_type.list_all_morphologies()[0])
    m_post = scaffold.morphology_repository.get_morphology(to_type.list_all_morphologies()[0])
    m_pre.voxelize(50, compartments=m_pre.get_compartments(from_comp_types))
    m_pre2.voxelize(150)
    m_post.voxelize(50, compartments=m_post.get_compartments(to_comp_types))
    pos_pre = ps_pre.positions[int(pair.from_id) - ps_pre.identifiers[0]]
    pos_post = ps_post.positions[int(pair.to_id) - ps_post.identifiers[0]]

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

    m_pre.cloud.voxels = np.zeros(m_pre.cloud.voxels.shape, dtype=bool)
    m_pre.cloud.voxels[selected_voxels_pre] = True
    m_post.cloud.voxels = np.zeros(m_post.cloud.voxels.shape, dtype=bool)
    m_post.cloud.voxels[selected_voxels_post] = True
    fig4 = plot_morphology(m_pre, offset=pos_pre, show=False, segment_radius=4, color=from_type.plotting.color)
    fig4.layout.scene.yaxis.range = [0, 160]
    plot_voxel_cloud(m_pre.cloud, fig=fig4, selected_voxels=[], offset=pos_pre, show=False, set_range=False)
    plot_morphology(m_post, fig=fig4, offset=pos_post, show=False, segment_radius=4, set_range=False, color=to_type.plotting.color)
    plot_voxel_cloud(m_post.cloud, fig=fig4, offset=pos_post, selected_voxels=[], show=False, set_range=False)
    intersections = [i for i in cs.intersections if i.from_id == pair.from_id and i.to_id == pair.to_id]
    i_pre = np.array([i.from_compartment.end + pos_pre for i in intersections])
    i_post = np.array([i.to_compartment.end + pos_post for i in intersections])
    fig4.add_trace(go.Scatter3d(x=i_pre[:,0], y=i_pre[:,2], z=i_pre[:,1], mode="markers", marker=dict(size=8,color="violet")))
    fig4.add_trace(go.Scatter3d(x=i_post[:,0], y=i_post[:,2], z=i_post[:,1], mode="markers", marker=dict(symbol="diamond-open", size=8,color="violet")))
    return fig4
