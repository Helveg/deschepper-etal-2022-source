import os, plotly.graph_objects as go
from bsb.core import from_hdf5
from bsb.plotting import hdf5_plot_psth, hdf5_plot_spike_raster
import numpy as np, h5py
from scipy import stats
import plotly.io as pio


network_path = os.path.join(
    os.path.dirname(__file__), "..", "networks", "300x_200z.hdf5"
)
def results_path(*args):
    return os.path.join(
        os.path.dirname(__file__), "..", "results", *args
    )


def plot():
    scaffoldInstance = from_hdf5(network_path)
    config = scaffoldInstance.configuration
    cells = list(scaffoldInstance.get_cell_types())
    with h5py.File(results_path("results_stim_on_MFs_16075237073796946792023040046.hdf5"), "a") as f:
        order=dict(record_glomerulus_spikes=0, record_granules_spikes=1, record_golgi_spikes=2, record_pc_spikes=3,
            record_sc_spikes=4, record_bc_spikes=5)
        color=dict(record_glomerulus_spikes=config.cell_types["glomerulus"].plotting.color,
                record_granules_spikes=config.cell_types["granule_cell"].plotting.color,
                record_golgi_spikes=config.cell_types["golgi_cell"].plotting.color,
                record_pc_spikes=config.cell_types["purkinje_cell"].plotting.color,
                record_sc_spikes=config.cell_types["stellate_cell"].plotting.color,
                record_bc_spikes=config.cell_types["basket_cell"].plotting.color)
        for g in f["/recorders/soma_spikes"].values():
            if g.attrs["label"] not in order:
                print("Not sorting", g.name, "no order found")
            g.attrs["order"] = order.get(g.attrs["label"], 0)
            g.attrs['color'] = color.get(g.attrs["label"], 0)
        fig = hdf5_plot_psth(scaffoldInstance, f["/recorders/soma_spikes"], show=False, duration=5)
        ranges = [[0, 25], [0, 25], [0, 125], [0, 130], [0, 120], [0, 120]]
        for i in range(len(ranges)):
            fig.update_yaxes(range=ranges[i], row=i + 1, col=1)
    return fig
