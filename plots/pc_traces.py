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
        path = glob(results_path("balanced_sensory", "*.hdf5"))[0]
    if net_path is None:
        net_path = network_path(selection.network)
    network = from_hdf5(net_path)
    with h5py.File(path, "r") as f:
        traces = hdf5_gather_voltage_traces(f, "recorders/soma_voltages/", map(str, map(int, selection.purkinje_cell_ids)))
        traces.set_legends(["Membrane potential [mV]"])
        order = [1, 0]
        for label, id in selection.purkinje_cells.items():
            traces.cells[id].title = label
        traces.set_colors([network.configuration.cell_types["purkinje_cell"].plotting.color])
        traces.reorder(order)
        fig = plot_traces(traces, x=list(f["time"][()] - 5500), input_region=[500, 540], range=[000, 1000], show=False)
        fig.update_yaxes(range=[-80, 50])
        fig.update_layout(showlegend=False)
    return fig

def meta():
    return {"width": 400, "height": 400}

if __name__ == "__main__":
    plot().show()
