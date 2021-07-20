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

start = 5750
stop = 6250

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
            col=1
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
            col=1
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
                col=1
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
        col=1
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
                opacity=0.09,
                showlegend=False,
            ),
            row=1,
            col=1
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
                opacity=0.09,
                showlegend=False,
            ),
            row=1,
            col=1
        )

def make_others(fig, network):
    morpho = {"stellate_cell": "StellateCell", "basket_cell": "BasketCell", "purkinje_cell": "PurkinjeCell", "golgi_cell": "GolgiCell"}
    for i, pos, type in zip(
        itertools.count(),
        ([180, 230, 130], [-70, 260, 120], [150, 200, 100], [-70, 230, 100], [130, 130, 100], [-70, 130, 90], [130, 120, 50], [-20, 140, 150]),
        ["stellate_cell"] * 2 + ["basket_cell"] * 2 + ["purkinje_cell"] * 2 + ["golgi_cell"] * 2
    ):
        onbeam = not (i % 2)
        if not onbeam:
            continue
        ps = network.get_placement_set(type)
        color = ps.type.plotting.color
        soma_radius = ps.type.placement.soma_radius
        m = network.morphology_repository.get_morphology(morpho.get(type))
        mfig = plot_morphology(m, offset=pos, show=False, color=color, segment_radius=1 + onbeam * 2 , soma_radius=soma_radius)
        for t in mfig.data:
            fig.add_trace(t, row=1, col=1)

def make_traces(fig, network, path, defs, col, xshift=-5500):
    defs = dict(map(lambda kv: (network.get_cell_type(kv[0]), kv[1]), defs.items()))
    with h5py.File(path, "r") as f:
        time = f["time"][()]
        time_mask = (time >= start) & (time <= stop)
        for i, (ct, id) in enumerate(defs.items()):
            if ct.name == "granule_cell":
                for j, grc_id in enumerate(id):
                    data = [d for d in f["recorders/granules"].values() if d.attrs["cell_id"] == grc_id][0][()]
                    fig.add_scatter(
                        x=time[time_mask] + xshift,
                        y=data[time_mask],
                        line_color=ct.plotting.color,
                        row=i*4+1+j,
                        col=col
                    )
            else:
                if ct.name == "purkinje_cell" and col == 4:
                    fig.add_scatter(
                        x = [time[time_mask][20] + xshift] * 2,
                        y = [0, -20],
                        mode="lines",
                        line=dict(
                            width=2,
                            color="black"
                        )
                    )
                data = f[f"recorders/soma_voltages/{id}"][()]
                fig.add_scatter(
                    x=time[time_mask] + xshift,
                    y=data[time_mask],
                    line_color=ct.plotting.color,
                    row=i*4+1,
                    col=col
                )

def make_onbeam_traces(fig, network, path):
    onbeam = {
        "basket_cell": selection.basket_cells["High activity"],
        "stellate_cell": selection.stellate_cells["High activity"],
        "purkinje_cell": selection.purkinje_cells["On beam"],
        "golgi_cell": selection.golgi_cells["High activity"],
        "granule_cell": [selection.grc_balanced[i] for i in (4, 3, 2, 1)],
    }
    make_traces(fig, network, path, onbeam, 5)

def make_offbeam_traces(fig, network, path):
    offbeam = {
        "basket_cell": selection.basket_cells["Low activity"],
        "stellate_cell": selection.stellate_cells["Low activity"],
        "purkinje_cell": selection.purkinje_cells["Off beam"],
        "golgi_cell": 36,
        "granule_cell": selection.grc_balanced[2],
    }
    make_traces(fig, network, path, offbeam, 4)

def make_plasticity_traces(fig, network, path):
    onbeam = {
        "basket_cell": selection.basket_cells["High activity"],
        "stellate_cell": selection.stellate_cells["High activity"],
        "purkinje_cell": selection.purkinje_cells["On beam"],
        "golgi_cell": selection.golgi_cells["High activity"],
        "granule_cell": [selection.grc_balanced[i] for i in (4, 3, 2, 1)],
    }
    path = results_path("grc_plasticity", "01", "gcu_01.hdf5")
    make_traces(fig, network, path, onbeam, 4)
    path = results_path("grc_plasticity", "09", "gcu_09.hdf5")
    make_traces(fig, network, path, onbeam, 6)


def plot(path=None, net_path=None):
    if path is None:
        path = results_path("sensory_gabazine", "sensory_burst_control.hdf5")
    if net_path is None:
        net_path = network_path(selection.network)
    network = from_hdf5(net_path)
    selected_mf = selection.stimulated_mf_poiss
    fig = make_subplots(
        rows=20,
        cols=6,
        specs=[
            [{"type": "scene", "rowspan": 20, "colspan": 3}, None, None, {"rowspan": 4}, {"rowspan": 4}, {"rowspan": 4}],
            [None] * 6,
            [None] * 6,
            [None] * 6,
        ]
        + [
            [None, None, None, {"rowspan": 4}, {"rowspan": 4}, {"rowspan": 4}],
            [None] * 6,
            [None] * 6,
            [None] * 6,
        ] * 3 + [
            [None, None, None, {}, {}, {}],
            [None, None, None, {}, {}, {}],
            [None, None, None, {}, {}, {}],
            [None, None, None, {}, {}, {}],
        ],
        horizontal_spacing=0.01,
        vertical_spacing=0.02,
        shared_xaxes=True,
    )
    scene = dict(
        xaxis_title="X",
        yaxis_title="Z",
        zaxis_title="Y",
        yaxis_visible=False,
        xaxis_visible=False,
        zaxis_visible=False,
        camera=dict(up=dict(x=0,y=0,z=1),center=dict(x=-0.18991699966962725,y=-0.06435691467383287,z=-0.010174254874020407),eye=dict(x=0.25399401237153907,y=-1.3457732394869972,z=0.24405511290570647)),
    )
    fig.update_layout(scene=scene, showlegend=False)

    make_mf(fig, network, selected_mf)
    make_grc_bundle(fig, network, selected_mf)
    make_others(fig, network)
    make_onbeam_traces(fig, network, path)
    make_plasticity_traces(fig, network, path)
    for i in range(1, 6):
        for j in (4, 5):
            fig.update_yaxes(row=i, col=j, range=[-75, 50], visible=False)

    pattern = [500.0, 504.0, 508.0, 514.0, 520.0]
    sig = go.Figure([])
    for spike in pattern:
        sig.add_shape(
            type="line",
            x0=spike,
            x1=spike,
            y0=0.03,
            y1=0.9,
            line=dict(
                color="black",
                width=2,
            ),
        )
    sig.update_layout(xaxis=dict(
        range=[450, 550],
        tickmode="array",
        tickvals=[450, 500, 520, 550]
    ), yaxis_range=[0, 1], yaxis_visible=False)
    figs = {"main": fig, "sig": sig}
    return figs

def meta(key):
    if key == "main":
        return {"width": 1800 / 2, "height": 1920 / 2}
    elif key == "sig":
        return {"width": 250, "height": 150}
