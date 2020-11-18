from bsb.core import Scaffold, from_hdf5
from bsb.plotting import plot_morphology, plot_voxel_cloud, set_scene_aspect
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from . import make_3dsubplots
from random import sample
import os, numpy as np, selection
from colour import Color

network = os.path.join(os.path.dirname(__file__), "..", "networks", "300x_200z.hdf5")
scaffold = from_hdf5(network)
camera = dict(up=dict(x=0,y=0,z=1),center=dict(x=-0.03131647105611321,y=0.039403276571002094,z=0.06362627967144648),eye=dict(x=2.9750537902845013,y=0.89709471640103,z=0.40722192762454806))

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

    # Picks a random connection where the axon reaches into the purkinje voxels from far
    while True:
        pair = sample(connections, 1)[0]
        pos_pre = ps_pre.positions[int(pair.from_id) - ps_pre.identifiers[0]]
        pos_post = ps_post.positions[int(pair.to_id) - ps_post.identifiers[0]]
        if pos_pre[2] > pos_post[2] + 40:
            break

    # Ignore the choice above and replace by hardcoded values for which the `camera` has
    # been chosen.
    from_id = selection.vi_stellate
    to_id = selection.vi_purkinje
    pos_pre = ps_pre.positions[from_id - ps_pre.identifiers[0]]
    pos_post = ps_post.positions[to_id - ps_post.identifiers[0]]

    m_pre = scaffold.morphology_repository.get_morphology(from_type.list_all_morphologies()[0])
    m_pre2 = scaffold.morphology_repository.get_morphology(from_type.list_all_morphologies()[0])
    m_post = scaffold.morphology_repository.get_morphology(to_type.list_all_morphologies()[0])
    m_pre.voxelize(50, compartments=m_pre.get_compartments(from_comp_types))
    m_pre2.voxelize(150)
    m_post.voxelize(50, compartments=m_post.get_compartments(to_comp_types))

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
    fig2 = plot_morphology(m_pre, offset=pos_pre, show=False, set_range=False, color=fcd, soma_radius=from_type.placement.soma_radius, segment_radius={"axon": 6, "dendrites": 2.5})
    plot_voxel_cloud(m_pre.cloud, fig=fig2, selected_voxels=selected_voxels_pre, offset=pos_pre, show=False, cubic=False, set_range=False)
    plot_morphology(m_post, fig=fig2, offset=pos_post, show=False, set_range=False, color=to_type.plotting.color, soma_radius=to_type.placement.soma_radius, segment_radius={"axon": 2.5, "dendrites": 2.5})
    plot_voxel_cloud(m_post.cloud, fig=fig2, offset=pos_post, selected_voxels=selected_voxels_post, show=False, cubic=False, set_range=False)

    set_scene_aspect(fig2.layout.scene, None, mode="data")
    fig2.layout.scene.camera = camera
    return fig2
