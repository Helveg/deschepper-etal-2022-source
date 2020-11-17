from bsb.core import from_hdf5
from grc_cloud import granule_disc
from plotly import graph_objs as go
import numpy as np
import bsb.plotting as plotting
import selection

camera = dict(up=dict(x=0,y=0,z=1),center=dict(x=0,y=0,z=0),eye=dict(x=0.1889550858586422,y=1.8486740326218545,z=1.1109456767267265))

def plot():
    network = from_hdf5("networks/300x_200z.hdf5")
    traces = []
    traces.append(granule_disc("networks/300x_200z.hdf5", "results/results_stim_on_MFs_Poiss.hdf5", bar_l=0.6))
    morphos = {"basket_cell": "BasketCell", "stellate_cell": "StellateCell"}
    titles = {"basket_cell": "Basket", "stellate_cell": "Stellate"}
    for key in ("basket_cell", "stellate_cell"):
        labels = selection.__dict__[f"{key}s"]
        m = network.morphology_repository.get_morphology(morphos[key])
        mli_set = network.get_placement_set(key)
        mli_mask = np.isin(mli_set.identifiers, np.array(list(labels.values())))
        mli_pos = mli_set.positions[mli_mask]
        mli_color = network.configuration.cell_types[key].plotting.color

        traces.append(go.Scatter(
            x=mli_pos[:, 0],
            y=mli_pos[:,2],
            text=list(map(lambda x: x + " " + key.replace("_", " "), labels.keys())),
            mode="markers+text",
            textposition="bottom center",
            marker=dict(
                color=mli_color,
                size=10
            ),
            legendgroup=key,
            name=key.split("_")[0].title() + " cells"
        ))
        show_first_legend = True
        for pos in mli_pos:
            ext_min, ext_max = protrusion(m, pos[1])
            sc = go.Scatter(
                x=[pos[0] + ext_min, pos[0] + ext_max],
                y=[pos[2], pos[2]],
                mode="lines",
                line=dict(
                    width=4,
                    color=mli_color,
                ),
                name=f"{titles[key]} extension into molecular layer",
                legendgroup=key,
                showlegend=show_first_legend
            )
            # traces.append(sc)
            show_first_legend = False

    fig = go.Figure(traces, layout=dict(
        title_text="Molecular layer interneurons",
        legend_itemsizing="constant",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis_range=[300, 0],
        yaxis_range=[0, 200],
        xaxis_title="X",
        yaxis_title="Z",
    ))
    return fig


def protrusion(morphology, offset, layer_init=150):
    pos = np.array([c.midpoint for c in morphology.compartments if "dendrites" in c.labels])
    x = pos[pos[:, 1] + offset > layer_init, 0]
    if len(x) == 0:
        return 0, 0
    return np.min(x), np.max(x)

if __name__ == "__main__":
    plot().show()
