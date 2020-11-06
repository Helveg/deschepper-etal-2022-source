from bsb.core import from_hdf5
from grc_cloud import granule_cloud
from plotly import graph_objs as go
import numpy as np
import bsb.plotting as plotting
import selection

def plot():
    network = from_hdf5("networks/300x_200z.hdf5")
    traces = []
    traces.append(granule_cloud("networks/300x_200z.hdf5", "results/results_stim_on_MFs_Poiss.hdf5"))
    for key in ("basket_cell", "stellate_cell"):
        labels = selection.__dict__[f"{key}s"]
        mli_set = network.get_placement_set(key)
        mli_mask = np.isin(mli_set.identifiers, np.array(list(labels.values())))
        mli_pos = mli_set.positions[mli_mask]
        mli_color = network.configuration.cell_types[key].plotting.color

        traces.append(go.Scatter3d(
            x=mli_pos[:, 0],
            y=mli_pos[:,2],
            z=mli_pos[:,1],
            text=list(labels.keys()),
            mode="markers+text",
            marker=dict(
                color=mli_color,
                size=10
            ),
            name=key.split("_")[0].title() + " cells"
        ))

    fig = go.Figure(traces)
    plotting.network_figure(fig=fig, show=False, cubic=False)
    fig.update_layout(scene=dict(xaxis_range=[0, 300], yaxis_range=[0, 200], zaxis_range=[0, 300]))
    return fig

if __name__ == "__main__":
    plot().show()
