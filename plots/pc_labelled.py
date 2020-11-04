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

    pc_labels = selection.purkinje_cells
    purkinje_set = network.get_placement_set("purkinje_cell")
    purkinje_mask = np.isin(purkinje_set.identifiers, np.array(list(pc_labels.values())))
    purkinje_pos = purkinje_set.positions[purkinje_mask]
    purkinje_color = network.configuration.cell_types["purkinje_cell"].plotting.color

    traces.append(go.Scatter3d(
        x=purkinje_pos[:, 0],
        y=purkinje_pos[:,2],
        z=purkinje_pos[:,1],
        text=list(pc_labels.keys()),
        mode="markers+text",
        marker=dict(
            color=purkinje_color
        ),
        name="Purkinje cells"
    ))

    fig = go.Figure(traces)
    plotting.network_figure(fig=fig, show=False, cubic=False)
    fig.update_layout(scene=dict(xaxis_range=[0, 300], yaxis_range=[0, 200], zaxis_range=[0,150]))
    return fig

if __name__ == "__main__":
    plot().show()
