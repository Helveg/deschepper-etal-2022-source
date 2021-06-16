from bsb.core import from_hdf5
import os, sys, h5py, numpy as np
sys.path.insert(0, os.path.join("..", "plots"))
import selection, plotly.graph_objs as go, scipy.stats
import h5py, selection, _layouts
import pickle
from ._paths import *
from glob import glob
import selection, random
from plotly.subplots import make_subplots

def make_mf(fig, network, selected_mf):
    cpos = {selection_mf: [] for selection_mf in selected_mf}
    cs = network.get_connectivity_set("mossy_to_glomerulus")
    ps = network.get_placement_set("glomerulus")
    pos_map = {c.id: c.position for c in ps.cells}
    for mf_id, glom_id in cs.get_dataset():
        if mf_id in selected_mf:
            cpos[mf_id].append(pos_map[int(glom_id)])

    centroids = dict(zip(cpos.keys(), (np.mean(cp, axis=0) if len(cp) else np.nan for cp in cpos.values())))
    for mf, pos in cpos.items():
        centroid = centroids[mf]
        fig.add_trace(
            go.Scatter3d(
                x=[p[0] for p in pos],
                y=[p[2] for p in pos],
                z=[p[1] for p in pos],
                mode="markers",
                marker=dict(
                    size=1.8,
                    color="black"
                ),
                showlegend=False,
            ),
            row=1,
            col=2
        )
        fig.add_trace(
            go.Scatter3d(
                x=[centroid[0]] * 2,
                y=[centroid[2]] * 2,
                z=[-10, -50],
                mode="lines",
                line=dict(
                    width=2,
                    color="black"
                ),
                showlegend=False,
            ),
            row=1,
            col=2
        )
        for p in pos:
            fig.add_trace(
                go.Scatter3d(
                    x=[centroid[0], p[0]],
                    y=[centroid[2], p[2]],
                    z=[-10, p[1]],
                    mode="lines",
                    opacity=0.5,
                    line=dict(
                        color="black"
                    ),
                    showlegend=False,
                ),
                row=1,
                col=2
            )

def make_grc_bundle(fig, network, selected_mf):
    ps = network.get_placement_set("granule_cell")
    ids = ps.identifiers
    mf_glom = network.get_connectivity_set("mossy_to_glomerulus").get_dataset()
    glom_grc = network.get_connectivity_set("glomerulus_to_granule").get_dataset()
    active_glom = mf_glom[np.isin(mf_glom[:, 0], selected_mf), 1]
    active_dendrites = glom_grc[np.isin(glom_grc[:, 0], active_glom), 1]
    d = dict(zip(*np.unique(active_dendrites, return_counts=True)))
    grc_to_dend = np.vectorize(lambda x: d.get(x, 0))
    act = grc_to_dend(ids) >= 2
    pos = ps.positions[act]
    fig.add_trace(
        go.Scatter3d(
            x=pos[:, 0],
            y=pos[:, 2],
            z=pos[:, 1],
            mode="markers",
            marker=dict(
                size=1.8,
                color=ps.type.plotting.color,
            ),
            opacity=0.5,
            showlegend=False,
        ),
        row=1,
        col=2
    )
    for p in pos:
        fig.add_trace(
            go.Scatter3d(
                x=[p[0]] * 2,
                y=[p[2]] * 2,
                z=[p[1], p[1] + 150],
                mode="lines",
                line=dict(
                    color=ps.type.plotting.color,
                ),
                opacity=0.1,
                showlegend=False,
            ),
            row=1,
            col=2
        )

def plot(net_path=None):
    if net_path is None:
        net_path = network_path(selection.network)
    network = from_hdf5(net_path)
    selected_mf = [240, 241, 274, 186]
    fig = make_subplots(rows=1, cols=3, specs=[[{},{"type": "scene"},{},]], horizontal_spacing=0)
    scene = dict(
        xaxis_title="X",
        yaxis_title="Z",
        zaxis_title="Y",
        camera=dict(up=dict(x=0,y=0,z=1),center=dict(x=0,y=0,z=0),eye=dict(x=-1.2076139791955867,y=2.43591978874818,z=1.1391801122814587)),
    )
    fig.update_layout(scene=scene)

    make_mf(fig, network, selected_mf)
    make_grc_bundle(fig, network, selected_mf)


    return fig
