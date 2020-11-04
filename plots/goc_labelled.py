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

    goc_labels = selection.golgi_cells
    golgi_set = network.get_placement_set("golgi_cell")
    golgi_mask = np.isin(golgi_set.identifiers, np.array(list(goc_labels.values())))
    golgi_pos = golgi_set.positions[golgi_mask]
    golgi_color = network.configuration.cell_types["golgi_cell"].plotting.color

    traces.append(go.Scatter3d(
        x=golgi_pos[:, 0],
        y=golgi_pos[:,2],
        z=golgi_pos[:,1],
        text=list(goc_labels.keys()),
        mode="markers+text",
        marker=dict(
            color=golgi_color
        ),
        name="Golgi cells"
    ))

    fig = go.Figure(traces)
    plotting.network_figure(fig=fig, show=False, cubic=False)
    fig.update_layout(scene=dict(xaxis_range=[0, 300], yaxis_range=[0, 200], zaxis_range=[0,120]))
    return fig

if __name__ == "__main__":
    plot().show()
