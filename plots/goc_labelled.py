from bsb.core import from_hdf5
from grc_cloud import granule_cloud
from plotly import graph_objs as go
import numpy as np
import bsb.plotting as plotting
import selection
from colour import Color

camera = dict(up=dict(x=0,y=0,z=1),center=dict(x=0,y=0,z=0),eye=dict(x=-0.6946206917338876,y=1.9557279422403595,z=0.6165470870545304))

def plot():
    network = from_hdf5("networks/300x_200z.hdf5")
    traces = []
    traces.append(granule_cloud("networks/300x_200z.hdf5", "results/results_stim_on_MFs_Poiss.hdf5"))

    gm = network.morphology_repository.get_morphology("GolgiCell")
    goc_labels = selection.golgi_cells
    golgi_set = network.get_placement_set("golgi_cell")
    golgi_mask = np.isin(golgi_set.identifiers, np.array(list(goc_labels.values())))
    golgi_pos = golgi_set.positions[golgi_mask]
    golgi_color = network.configuration.cell_types["golgi_cell"].plotting.color
    golgi_scale = list(Color(golgi_color).range_to("white", 10))
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
    color_select = [2, 0, 3]
    goc_radii = {"dendrites": 2, "axon": 0}
    for i in range(golgi_pos.shape[0]):
        plotting.plot_morphology(gm, fig=fig, color=str(golgi_scale[color_select[i]]), cubic=False, segment_radius=goc_radii, show=False, set_range=False, offset=golgi_pos[i, :])
    plotting.network_figure(fig=fig, show=False, cubic=False)
    fig.update_layout(scene=dict(camera=camera, aspectratio=dict(x=1, y=0.667, z=1.2), xaxis_range=[0, 300], yaxis_range=[0, 200], zaxis_range=[-60,300]))
    return fig

if __name__ == "__main__":
    plot().show()
