from bsb.core import from_hdf5
from grc_cloud import granule_disc
from plotly import graph_objs as go
import numpy as np
import bsb.plotting as plotting
import selection
from colour import Color

camera = dict(up=dict(x=0,y=0,z=1),center=dict(x=0,y=0,z=0),eye=dict(x=-0.6946206917338876,y=1.9557279422403595,z=0.6165470870545304))

def plot():
    network = from_hdf5("networks/300x_200z.hdf5")
    traces = []
    traces.append(granule_disc("networks/300x_200z.hdf5", "results/results_stim_on_MFs_Poiss.hdf5"))

    gm = network.morphology_repository.get_morphology("GolgiCell")
    goc_labels = selection.golgi_cells
    golgi_set = network.get_placement_set("golgi_cell")
    golgi_mask = np.isin(golgi_set.identifiers, np.array(list(goc_labels.values())))
    golgi_pos = golgi_set.positions[golgi_mask]
    golgi_color = network.configuration.cell_types["golgi_cell"].plotting.color
    golgi_scale = list(Color(golgi_color).range_to("white", 10))
    traces.append(go.Scatter(
        x=golgi_pos[:, 0],
        y=golgi_pos[:,2],
        text=list(goc_labels.keys()),
        textposition="bottom center",
        mode="markers+text",
        marker=dict(
            color=golgi_color,
            size=15
        ),
        legendgroup="golgi_ext",
        name="Golgi cells"
    ))
    show_first_legend = True
    for pos in golgi_pos:
        ext_min, ext_max = protrusion(gm, pos[1])
        sc = go.Scatter(
            x=[pos[0] + ext_min, pos[0] + ext_max],
            y=[pos[2], pos[2]],
            mode="lines",
            line=dict(
                width=4,
                color=golgi_color,
            ),
            name="Golgi extension into molecular layer",
            legendgroup="golgi_ext",
            showlegend=show_first_legend
        )
        # traces.append(sc)
        show_first_legend = False

    fig = go.Figure(traces, layout=dict(
        title_text="Golgi cells",
        legend_itemsizing="constant",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis_range=[0, 300],
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
