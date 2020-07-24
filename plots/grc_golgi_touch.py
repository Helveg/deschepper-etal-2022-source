from scaffold.core import Scaffold, from_hdf5
from scaffold.plotting import plot_morphology, plot_voxel_cloud
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from . import make_3dsubplots
from random import sample
import os, numpy as np

network = os.path.join(os.path.dirname(__file__), "..", "networks", "neuron.hdf5")
scaffold = from_hdf5(network)

def plot():
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
    fig = plot_morphology(mgr, show=False, offset=frc_pos, set_range=False, color=from_type.plotting.color, segment_radius=3)
    fig.add_trace(go.Scatter3d(x=from_i_pos[:,0], y=from_i_pos[:,2], z=from_i_pos[:,1], mode="markers", marker=dict(size=8,color="black")))
    fig.add_trace(go.Scatter3d(x=to_i_pos[:,0], y=to_i_pos[:,2], z=to_i_pos[:,1], mode="markers", marker=dict(symbol="diamond-open", size=8,color="black")))
    fig.add_trace(go.Scatter3d(x=from_p_pos[:,0], y=from_p_pos[:,2], z=from_p_pos[:,1], mode="markers", marker=dict(size=8,color="violet")))
    fig.add_trace(go.Scatter3d(x=to_p_pos[:,0], y=to_p_pos[:,2], z=to_p_pos[:,1], mode="markers", marker=dict(symbol="diamond-open", size=8,color="violet")))
    plot_morphology(mgc, show=False, fig=fig, offset=gaa_pos, segment_radius=2.5, set_range=False, color=to_type.plotting.color)
    plot_morphology(mgc, show=False, fig=fig, offset=gpf_pos, segment_radius=2.5, set_range=False, color="#639EEC")
    fig.layout.scene.yaxis.range = [-200, 200]
    return fig