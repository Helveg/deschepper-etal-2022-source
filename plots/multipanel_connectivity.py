from bsb.core import Scaffold, from_hdf5
from bsb.plotting import plot_morphology, plot_voxel_cloud
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from . import make_3dsubplots
from random import sample
import os, numpy as np
from ._paths import *
from glob import glob
import selection

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
    m_pre = network.morphology_repository.get_morphology(from_type.list_all_morphologies()[0])
    m_pre2 = network.morphology_repository.get_morphology(from_type.list_all_morphologies()[0])
    m_post = network.morphology_repository.get_morphology(to_type.list_all_morphologies()[0])
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

    fig = make_subplots(**make_3dsubplots(2, 2))

    fig1 = plot_morphology(m_pre2, offset=pos_pre, show=False, segment_radius=2.5)
    plot_voxel_cloud(m_pre2.cloud, selected_voxels=[], fig=fig1, offset=pos_pre, set_range=False, show=False)

    fig2 = plot_morphology(m_pre, offset=pos_pre, show=False, segment_radius=4, color=from_type.plotting.color)
    plot_voxel_cloud(m_pre.cloud, fig=fig2, selected_voxels=selected_voxels_pre, offset=pos_pre, show=False, set_range=False)
    plot_morphology(m_post, fig=fig2, offset=pos_post, show=False, segment_radius=4, color=to_type.plotting.color)
    plot_voxel_cloud(m_post.cloud, fig=fig2, offset=pos_post, selected_voxels=selected_voxels_post, show=False, set_range=False)

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
    fig4.add_trace(go.Scatter3d(x=i_pre[:,0], y=i_pre[:,2], z=i_pre[:,1], mode="markers", marker=dict(size=5,color="red")))
    fig4.add_trace(go.Scatter3d(x=i_post[:,0], y=i_post[:,2], z=i_post[:,1], mode="markers", marker=dict(size=5,color="red")))
    # fig4.show()
    fig3, granule_pos = tasteful_touching_scene(network)
    # fig3.show()

    for t in fig1.data:
        fig.add_trace(t, 1, 1)
    for t in fig2.data:
        fig.add_trace(t, 1, 2)
    for t in fig3.data:
        fig.add_trace(t, 2, 1)
    for t in fig4.data:
        fig.add_trace(t, 2, 2)
    return fig

def tasteful_touching_scene(network):
    cs_aa = network.get_connectivity_set("ascending_axon_to_golgi")
    cs_pf = network.get_connectivity_set("parallel_fiber_to_golgi")
    c_aa = cs_aa.intersections
    c_pf = cs_pf.intersections
    from_type = network.get_cell_type("granule_cell")
    to_type = network.get_cell_type("golgi_cell")
    ps_pre = network.get_placement_set(from_type)
    ps_post = network.get_placement_set(to_type)
    grc_id = ps_pre.identifiers
    grc_pos = ps_pre.positions
    goc_id = ps_post.identifiers
    goc_pos = ps_post.positions
    mgr = network.morphology_repository.get_morphology("GranuleCell")
    mgc = network.morphology_repository.get_morphology("GolgiCell")
    flat = {}
    # Flatten both connection sets based on from id
    for c in c_pf:
        if c.from_id not in flat:
            flat[c.from_id] = []
        flat[c.from_id].append(c.to_id)
    flat_ = {}
    for c in c_aa:
        if c.from_id not in flat_:
            flat_[c.from_id] = []
            flat_[c.from_id].append(c.to_id)

    # Pick only granule cells that are connected to atleast 2 golgi's by pf and 1 by aa
    flat = {k: v for k, v in flat.items() if len(v) > 1 and k in flat_}
    flat_aa = {k: [x for x in v if x in flat_[k]] for k, v in flat.items() if [x for x in v if x in flat_[k]]}

    candidates = []
    for grc, goc in flat_aa.items():
        goc1 = goc[0]
        other_gocs = flat[grc]
        other_gocs.remove(goc1)
        goc2 = sample(other_gocs, 1)[0]
        candidates.append((grc, goc1, goc2))

    f_grc, f_aa, f_pf = sample(candidates, 1)[0]
    intersections_aa = []
    intersections_pf = []
    frc_pos = grc_pos[grc_id.tolist().index(f_grc)]
    gaa_pos = goc_pos[goc_id.tolist().index(f_aa)]
    gpf_pos = goc_pos[goc_id.tolist().index(f_pf)]
    for intersection in c_aa:
        if intersection.from_id == f_grc and intersection.to_id == f_aa:
            intersections_aa.append(intersection)
    for intersection in c_pf:
        if intersection.from_id == f_grc and intersection.to_id == f_aa:
            intersections_pf.append((intersection, gaa_pos))
        if intersection.from_id == f_grc and intersection.to_id == f_pf:
            intersections_pf.append((intersection, gpf_pos))
    from_i_pos = np.array([i.from_compartment.end + frc_pos for i in intersections_aa])
    to_i_pos = np.array([i.to_compartment.end + gaa_pos for i in intersections_aa])
    from_p_pos = np.array([i[0].from_compartment.end + frc_pos for i in intersections_pf])
    to_p_pos = np.array([i[0].to_compartment.end + i[1] for i in intersections_pf])
    fig = plot_morphology(mgr, show=False, offset=frc_pos, set_range=False, color=from_type.plotting.color)
    fig.add_trace(go.Scatter3d(x=from_i_pos[:,0], y=from_i_pos[:,2], z=from_i_pos[:,1], mode="markers", marker=dict(size=5,color="red")))
    fig.add_trace(go.Scatter3d(x=to_i_pos[:,0], y=to_i_pos[:,2], z=to_i_pos[:,1], mode="markers", marker=dict(size=5,color="red")))
    fig.add_trace(go.Scatter3d(x=from_p_pos[:,0], y=from_p_pos[:,2], z=from_p_pos[:,1], mode="markers", marker=dict(size=5,color="green")))
    fig.add_trace(go.Scatter3d(x=to_p_pos[:,0], y=to_p_pos[:,2], z=to_p_pos[:,1], mode="markers", marker=dict(size=5,color="green")))
    plot_morphology(mgc, show=False, fig=fig, offset=gaa_pos, segment_radius=2.5, set_range=False, color=to_type.plotting.color)
    plot_morphology(mgc, show=False, fig=fig, offset=gpf_pos, segment_radius=2.5, set_range=False, color="#639EEC")
    return fig, frc_pos

def blurred_out_nasty_scene(network):
    cs = network.get_connectivity_set("stellate_to_purkinje")
    conn = cs.intersections
    from_type = network.get_cell_type("stellate_cell")
    to_type = network.get_cell_type("purkinje_cell")
    ps_pre = network.get_placement_set(from_type)
    ps_post = network.get_placement_set(to_type)
    scid = ps_pre.identifiers
    scpos = ps_pre.positions
    pcid = ps_post.identifiers
    ppos = ps_post.positions
    m_pre = network.morphology_repository.get_morphology("StellateCell")
    m_post = network.morphology_repository.get_morphology("PurkinjeCell")

    t = sample(conn, 1)[0]
    intersections = [i for i in conn if i.from_id == t.from_id and i.to_id == t.to_id]
    sc_pos = grc_pos[grc_id.tolist().index(f_grc)]

    from_i_pos = np.array([i.from_compartment.end + _pos for i in intersections])
    to_i_pos = np.array([i.to_compartment.end + gaa_pos for i in intersections])
    fig = plot_morphology(mgr, show=False, offset=frc_pos, set_range=False, color=from_type.plotting.color)
    fig.add_trace(go.Scatter3d(x=from_i_pos[:,0], y=from_i_pos[:,2], z=from_i_pos[:,1], mode="markers", marker=dict(size=5,color="red")))
    fig.add_trace(go.Scatter3d(x=to_i_pos[:,0], y=to_i_pos[:,2], z=to_i_pos[:,1], mode="markers", marker=dict(size=5,color="red")))
    fig.add_trace(go.Scatter3d(x=from_p_pos[:,0], y=from_p_pos[:,2], z=from_p_pos[:,1], mode="markers", marker=dict(size=5,color="green")))
    fig.add_trace(go.Scatter3d(x=to_p_pos[:,0], y=to_p_pos[:,2], z=to_p_pos[:,1], mode="markers", marker=dict(size=5,color="green")))
    plot_morphology(mgc, show=False, fig=fig, offset=gaa_pos, segment_radius=2.5, set_range=False, color=to_type.plotting.color)
    plot_morphology(mgc, show=False, fig=fig, offset=gpf_pos, segment_radius=2.5, set_range=False, color="#639EEC")
    return fig, frc_pos
