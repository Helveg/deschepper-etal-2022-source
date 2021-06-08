from bsb.core import from_hdf5
from grc_cloud import granule_disc
from plotly import graph_objs as go
import numpy as np
import bsb.plotting as plotting
import selection
from ._paths import *
from glob import glob
import selection

camera = dict(up=dict(x=0,y=0,z=1),center=dict(x=0,y=0,z=0),eye=dict(x=0.0005827336957326205,y=0.06985506890139645,z=2.163936212038197))

def plot(path=None, net_path=None):
    if path is None:
        path = glob(results_path("sensory_burst", "*"))[0]
    if net_path is None:
        net_path = network_path(selection.network)
    network = from_hdf5(net_path)
    traces = []
    traces.append(granule_disc(net_path, path))

    pm = network.morphology_repository.get_morphology("PurkinjeCell")
    pc_labels = selection.purkinje_cells
    purkinje_set = network.get_placement_set("purkinje_cell")
    purkinje_mask = np.isin(purkinje_set.identifiers, np.array(list(pc_labels.values())))
    purkinje_pos = purkinje_set.positions[purkinje_mask]
    purkinje_color = network.configuration.cell_types["purkinje_cell"].plotting.color

    traces.append(go.Scatter(
        x=purkinje_pos[:, 0],
        y=purkinje_pos[:,2],
        text=list(pc_labels.keys()),
        textposition="bottom center",
        legendgroup="purkinje_ext",
        mode="markers+text",
        marker=dict(
            color=purkinje_color,
            size=15,
        ),
        name="Purkinje cells"
    ))

    show_first_legend = True
    for pos in purkinje_pos:
        ext_min, ext_max = protrusion(pm, pos[1])
        sc = go.Scatter(
            x=[pos[0] + ext_min, pos[0] + ext_max],
            y=[pos[2], pos[2]],
            mode="lines",
            line=dict(
                width=4,
                color=purkinje_color,
            ),
            name="Golgi extension into molecular layer",
            legendgroup="purkinje_ext",
            showlegend=show_first_legend
        )
        # traces.append(sc)
        show_first_legend = False

    fig = go.Figure(traces, layout=dict(
        title_text="Purkinje cells",
        legend_itemsizing="constant",
        #yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis_range=[300, 0],
        yaxis_range=[0, 200],
        xaxis_title="X",
        yaxis_title="Z",
    ))
    return fig


def protrusion(morphology, offset, layer_init=150):
    pos = morphology.flatten()
    x = pos[0][pos[1] + offset > layer_init]
    if len(x) == 0:
        return 0, 0
    return np.min(x), np.max(x)

if __name__ == "__main__":
    plot().show()
