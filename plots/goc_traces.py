from bsb.core import from_hdf5
from bsb.plotting import plot_morphology, hdf5_gather_voltage_traces, plot_traces
from plotly import graph_objs as go
import selection, numpy as np, h5py
from colour import Color
from ._paths import *
from glob import glob
import selection

def plot(path=None, net_path=None):
    if path is None:
        path = glob(results_path("sensory_burst", "*"))[0]
    if net_path is None:
        net_path = network_path(selection.network)
    network = from_hdf5(net_path)
    with h5py.File(path, "r") as f:
        traces = hdf5_gather_voltage_traces(f, "recorders/soma_voltages/", map(str, map(int, selection.golgi_cell_ids)))
        traces.set_legends(["Membrane potential"])
        order = [1, 2, 0]
        for label, id in selection.golgi_cells.items():
            traces.cells[id].title = label
        traces.set_colors([network.configuration.cell_types["golgi_cell"].plotting.color])
        traces.reorder(order)
        fig = plot_traces(traces, x=list(f["time"]), input_region=[6000, 6100], range=[5800, 6300], show=False)
        fig.update_yaxes(range=[-70, 25])
    return fig

def meta():
    return {"width": 1920 / 3 * 2}
