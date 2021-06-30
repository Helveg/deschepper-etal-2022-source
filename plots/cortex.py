from bsb.core import from_hdf5
from bsb.plotting import plot_morphology
import os, sys, h5py, numpy as np
sys.path.insert(0, os.path.join("..", "plots"))
import selection, plotly.graph_objs as go, scipy.stats
import h5py, selection, _layouts
import pickle, itertools
from ._paths import *
from glob import glob
import selection, random
from plotly.subplots import make_subplots

camera = dict(up=dict(x=0,y=0,z=1),center=dict(x=0.03490834696804547,y=0.0180800609069456,z=0.012772107184562962),eye=dict(x=0.47881935900921174,y=-1.2633362639062184,z=0.26700147496428983))

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
                opacity=0.03,
                showlegend=False,
            ),
            row=1,
            col=2
        )
        fig.add_trace(
            go.Scatter3d(
                x=[p[0]] * 2,
                y=[-50, 300],
                z=[p[1] + 150] * 2,
                mode="lines",
                line=dict(
                    color=ps.type.plotting.color,
                ),
                opacity=0.03,
                showlegend=False,
            ),
            row=1,
            col=2
        )

def make_others(fig, network):
    morpho = {"stellate_cell": "StellateCell", "basket_cell": "BasketCell", "purkinje_cell": "PurkinjeCell", "golgi_cell": "GolgiCell"}
    for i, pos, type in zip(
        itertools.count(),
        ([180, 230, 130], [-70, 260, 120], [150, 200, 100], [-70, 230, 100], [130, 130, 100], [-70, 130, 90], [130, 120, 50], [-20, 140, 150]),
        ["stellate_cell"] * 2 + ["basket_cell"] * 2 + ["purkinje_cell"] * 2 + ["golgi_cell"] * 2
    ):
        onbeam = not (i % 2)
        ps = network.get_placement_set(type)
        # opacity = (0.3 + onbeam * 0.7)
        # rgba = [float.fromhex(ps.type.plotting.color[(i*2+1):((i+1)*2+1)]) for i in range(3)] + [opacity * 255]
        # print(rgba)
        # color = "rgb(" + ", ".join(map(str, rgba)) + ")"
        color = ps.type.plotting.color
        soma_radius = ps.type.placement.soma_radius
        m = network.morphology_repository.get_morphology(morpho.get(type))
        mfig = plot_morphology(m, offset=pos, show=False, color=color, segment_radius=1 + onbeam * 2 , soma_radius=soma_radius)
        for t in mfig.data:
            fig.add_trace(t, row=1, col=2)

def plot(net_path=None):
    if net_path is None:
        net_path = network_path(selection.network)
    network = from_hdf5(net_path)
    selected_mf = selection.stimulated_mf_poiss
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
    make_others(fig, network)

    fig.update_layout(scene=dict(camera=camera, yaxis_visible=False, xaxis_visible=False, zaxis_visible=False))

    return fig
