import os, plotly.graph_objects as go
from plotly.subplots import make_subplots
from bsb.core import from_hdf5
from bsb.plotting import (
    plot_network,
    MorphologyScene,
    plot_morphology,
    set_scene_range,
    get_soma_trace,
)
from bsb.output import MorphologyRepository
import numpy as np
from ._paths import *
from glob import glob
import selection

max_golgis = 2

def plot(net_path=None):
    if net_path is None:
        net_path = network_path(selection.network)
    scaffold = from_hdf5(net_path)
    cs = scaffold.get_connectivity_set("glomerulus_to_golgi")
    conns = cs.get_dataset()
    from_type = scaffold.get_cell_type("glomerulus")
    to_type = scaffold.get_cell_type("golgi_cell")
    ps_pre = scaffold.get_placement_set(from_type)
    ps_post = scaffold.get_placement_set(to_type)
    glom_id = ps_pre.identifiers
    glom_poss = ps_pre.positions
    goc_id = ps_post.identifiers
    goc_pos = ps_post.positions
    mgc = scaffold.morphology_repository.get_morphology("GolgiCell")
    glom = int(np.random.choice(glom_id, size=1)[0])
    golgis = conns[conns[:, 0] == glom, 1]
    if len(golgis) > max_golgis:
        golgis = np.random.choice(golgis, size=max_golgis)


    glom_pos = glom_poss[glom_id.tolist().index(glom)]
    golgi_poss = []
    for g in golgis:
        golgi_poss.append(goc_pos[goc_id.tolist().index(int(g))])

    intersections = []
    for intersection in cs.intersections:
        if intersection.from_id == glom and intersection.to_id in golgis:
            intersection.to_pos = np.array(intersection.to_compartment.end) + golgi_poss[golgis.tolist().index(intersection.to_id)]
            intersections.append(intersection)

    fig = go.Figure()
    for gp in golgi_poss:
        plot_morphology(mgc, show=False, fig=fig, offset=gp, set_range=False, color=to_type.plotting.color, segment_radius=3)
    fig.add_trace(get_soma_trace(50, glom_pos, color=from_type.plotting.color, opacity=0.2, steps=20))
    i_pos = np.array([i.to_pos for i in intersections])
    fig.add_trace(go.Scatter3d(x=i_pos[:,0], y=i_pos[:,2], z=i_pos[:,1], mode="markers", marker=dict(symbol="diamond-open", size=8,color="violet")))
    set_scene_range(fig.layout.scene, [[-100, 300], [-100, 300], [-100, 300]])
    return fig
