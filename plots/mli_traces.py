from bsb.core import from_hdf5
from bsb.plotting import plot_morphology, hdf5_gather_voltage_traces, plot_traces
from plotly import graph_objs as go
import selection, numpy as np, h5py
from colour import Color

def plot():
    network = from_hdf5("networks/300x_200z.hdf5")
    bc_info = ("bc", "basket_cell", selection.basket_cells)
    sc_info = ("sc", "stellate_cell", selection.stellate_cells)
    figs = {}
    with h5py.File("results/results_stim_on_MFs_Poiss.hdf5", "r") as f:
        for tag, key, select in (bc_info, sc_info):
            traces = hdf5_gather_voltage_traces(f, "recorders/soma_voltages/", map(str, map(int, select.values())))
            traces.set_legends(["Membrane potential"])
            order = [0, 1]
            for label, id in select.items():
                traces.cells[id].title = label
            traces.set_colors([network.configuration.cell_types[key].plotting.color])
            traces.reorder(order)
            fig = plot_traces(traces, x=list(np.arange(0, 900, 0.1)), show=False, input_region=[400, 500], cutoff=3000)
            cfg = selection.btn_config.copy()
            cfg["filename"] = key + "_traces"
            figs[tag] = fig
    return figs

def meta(key):
    return {"width": 1920 / 3 * 2}

if __name__ == "__main__":
    plot()
