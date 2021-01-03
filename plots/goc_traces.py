from bsb.core import from_hdf5
from bsb.plotting import plot_morphology, hdf5_gather_voltage_traces, plot_traces
from plotly import graph_objs as go
import selection, numpy as np, h5py
from colour import Color

def figmod(fig):
    pass

def plot():
    network = from_hdf5("networks/300x_200z.hdf5")
    with h5py.File("results/results_stim_on_MFs_Poiss.hdf5", "r") as f:
        traces = hdf5_gather_voltage_traces(f, "recorders/soma_voltages/", map(str, map(int, selection.golgi_cell_ids)))
        traces.set_legends(["Membrane potential"])
        order = [1, 2, 0]
        for label, id in selection.golgi_cells.items():
            traces.cells[id].title = label
        traces.set_colors([network.configuration.cell_types["golgi_cell"].plotting.color])
        traces.reorder(order)
        fig = plot_traces(traces, x=list(np.arange(0, 900, 0.1)), input_region=[400, 500], cutoff=3000, show=False)
    return fig

def meta():
    return {"width": 1920 / 3 * 2}
