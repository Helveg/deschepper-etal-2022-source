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
    bc_info = ("bc", "basket_cell", selection.basket_cells)
    sc_info = ("sc", "stellate_cell", selection.stellate_cells)
    figs = {}
    with h5py.File(path, "r") as f:
        for tag, key, select in (bc_info, sc_info):
            traces = hdf5_gather_voltage_traces(f, "recorders/soma_voltages/", map(str, map(int, select.values())))
            traces.set_legends(["Membrane potential"])
            order = [0, 1]
            for label, id in select.items():
                traces.cells[id].title = label
            traces.set_colors([network.configuration.cell_types[key].plotting.color])
            traces.reorder(order)
            fig = plot_traces(traces, x=list(f["time"]), input_region=[6000, 6100], range=[5800, 6300], show=False)
            fig.update_yaxes(range=[-70, 25])
            cfg = selection.btn_config.copy()
            cfg["filename"] = key + "_traces"
            figs[tag] = fig
    return figs

def meta(key):
    return {"width": 1920 / 3 * 2}

if __name__ == "__main__":
    plot()
