from bsb.core import from_hdf5
from grc_cloud import granule_cloud
from plotly import graph_objs as go
import numpy as np
import bsb.plotting as plot


network = from_hdf5("networks/300x_200z.hdf5")
traces = []
traces.append(granule_cloud("networks/300x_200z.hdf5", "results/results_stim_on_MFs_Poiss.hdf5"))

goc_labels = {14: "A", 24: "B", 35: "C"}
golgi_set = network.get_placement_set("golgi_cell")
golgi_mask = np.isin(golgi_set.identifiers, list(goc_labels.keys()))
golgi_pos = golgi_set.positions[golgi_mask]

print(len(golgi_pos))

traces.append(go.Scatter3d(x=golgi_pos[:, 0],y=golgi_pos[:,2],z=golgi_pos[:,1], text=list(goc_labels.values()), mode="markers+text"))

fig = go.Figure(traces)
plot.network_figure(fig=fig, show=False, cubic=False)
fig.update_layout(scene=dict(xaxis_range=[0, 300], yaxis_range=[0, 200], zaxis_range=[0,120]))
fig.show()
