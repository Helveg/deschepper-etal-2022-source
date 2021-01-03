import os, plotly.graph_objects as go
from bsb.core import from_hdf5
from bsb.plotting import hdf5_plot_spike_raster
import numpy as np, h5py
from scipy import stats

network_path = os.path.join(
    os.path.dirname(__file__), "..", "networks", "results.hdf5"
)
def results_path(*args):
    return os.path.join(
        os.path.dirname(__file__), "..", "results", *args
    )

def get_isis(act):
    npact = act[:, 1]
    last_spike = (npact[()] <= 800).nonzero()[0][-1]
    return (npact[last_spike + 2] - npact[last_spike + 1], npact[last_spike + 1] - npact[last_spike])




def plot():
    network = from_hdf5("networks/300x_200z.hdf5")
    ps_pc = network.get_placement_set("purkinje_cell")
    with h5py.File(results_path("results_stim_on_MFs_4syncImp.hdf5"), "r") as f:
        pc_act = {id: f[f"/recorders/soma_spikes/{int(id)}"] for id in ps_pc.identifiers}
        pc_isis = {id: get_isis(act) for id, act in pc_act.items()}
        fig = go.Figure([go.Scatter(mode="lines", line=dict(color="black", width=0.5), x=[0, 80], y=[0, 80], showlegend=False), go.Scatter(mode="markers", x=[isis[0] for isis in pc_isis.values()], y=[isis[1] for isis in pc_isis.values()])])
        fig.update_layout(yaxis=dict(range=[0,80], title="Poststimulus ISI", scaleanchor="x", scaleratio=1), xaxis=dict(range=[0,80], title="Recovered ISI"))
    return fig
