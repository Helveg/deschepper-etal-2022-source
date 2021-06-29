from bsb.core import Scaffold, from_hdf5
from bsb.plotting import plot_morphology, plot_voxel_cloud, set_scene_range, set_scene_aspect
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from . import make_3dsubplots
from random import sample
import os, numpy as np
from ._paths import *
from glob import glob
import selection

camera = dict(up=dict(x=0,y=0,z=1),center=dict(x=-0.006744664498454609,y=0.0009131327109698769,z=-0.02473425322323764),eye=dict(x=-0.6723971512027274,y=0.755599119995846,z=0.4780722796154636))

def plot(net_path=None):
    if net_path is None:
        net_path = network_path(selection.network)
    scaffold = from_hdf5(net_path)
    cs_aa = scaffold.get_connectivity_set("ascending_axon_to_golgi")
    cs_pf = scaffold.get_connectivity_set("parallel_fiber_to_golgi")
    c_aa = cs_aa.intersections
    c_pf = cs_pf.intersections
    from_type = scaffold.get_cell_type("granule_cell")
    to_type = scaffold.get_cell_type("golgi_cell")
    ps_pre = scaffold.get_placement_set(from_type)
    ps_post = scaffold.get_placement_set(to_type)
    grc_id = ps_pre.identifiers
    grc_pos = ps_pre.positions
    goc_id = ps_post.identifiers
    goc_pos = ps_post.positions
    mgr = scaffold.morphology_repository.get_morphology("GranuleCell")
    mgc = scaffold.morphology_repository.get_morphology("GolgiCell")
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
    f_grc, f_aa, f_pf = 23211, 13, 54
    print("Picked:", f_grc, f_aa, f_pf)
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
    from_i_pos = np.array([i.from_compartment.midpoint + frc_pos for i in intersections_aa])
    to_i_pos = np.array([i.to_compartment.midpoint + gaa_pos for i in intersections_aa])
    from_p_pos = np.array([i[0].from_compartment.midpoint + frc_pos for i in intersections_pf])
    to_p_pos = np.array([i[0].to_compartment.midpoint + i[1] for i in intersections_pf])
    fig = plot_morphology(mgr, show=False, offset=frc_pos, set_range=False, color=from_type.plotting.color, segment_radius=3, soma_radius=from_type.placement.soma_radius)
    fig.add_trace(go.Scatter3d(x=from_i_pos[:,0], y=from_i_pos[:,2], z=from_i_pos[:,1], name="Presyn. AA location", mode="markers", marker=dict(size=5,color="grey", line=dict(width=1, color="black"))))
    fig.add_trace(go.Scatter3d(x=to_i_pos[:,0], y=to_i_pos[:,2], z=to_i_pos[:,1], name="Postsyn. AA location",  mode="markers", marker=dict(symbol="diamond-open", size=8,color="grey")))
    fig.add_trace(go.Scatter3d(x=from_p_pos[:,0], y=from_p_pos[:,2], z=from_p_pos[:,1],  name="Presyn. PF location", mode="markers", marker=dict(size=5,color="violet")))
    fig.add_trace(go.Scatter3d(x=to_p_pos[:,0], y=to_p_pos[:,2], z=to_p_pos[:,1],  name="Postsyn. PF location", mode="markers", marker=dict(symbol="diamond-open", size=8,color="violet")))
    plot_morphology(mgc, show=False, fig=fig, offset=gaa_pos, segment_radius=2.5, set_range=False, color=to_type.plotting.color, soma_radius=to_type.placement.soma_radius)
    plot_morphology(mgc, show=False, fig=fig, offset=gpf_pos, segment_radius=2.5, set_range=False, color="#639EEC", soma_radius=to_type.placement.soma_radius)
    rng = [[-100, 170], [-90, 300], [-10, 240]]
    set_scene_range(fig.layout.scene, rng)
    set_scene_aspect(fig.layout.scene, rng)
    fig.layout.scene.camera = camera
    fig.update_layout(scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False))
    return fig
