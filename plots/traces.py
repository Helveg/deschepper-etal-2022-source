from bsb.plotting import hdf5_gather_voltage_traces, plot_traces, CellTraceCollection
from h5py import File
import os

network_path = os.path.join(
    os.path.dirname(__file__), "..", "networks", "small.hdf5"
)
def results_path(*args):
    return os.path.join(
        os.path.dirname(__file__), "..", "results", *args
    )

def plot():
    # Open the last generated HDF5 file of the `poc` simulation
    with File(results_path('results_poc_16000919304394250806305914.hdf5'), "r") as f:
        # Collect traces from cells across multiple recording groups.
        cell_traces = hdf5_gather_voltage_traces(f, "/recorders", ["soma_voltages"])
        # Take only those cells that on top of their soma also had a dendrite recorded
        selected = "granule_cell"
        representatives = []
        for c in cell_traces:
            # if c.traces[0].meta["label"] not in selected:
            representatives.append(c)
            selected.add(c.traces[0].meta["label"])
        representatives = CellTraceCollection(representatives)
        representatives.set_legends(["Soma (mV)"])
        representatives.set_colors(["Crimson"])
        fig = plot_traces(representatives, show=False)
    return fig
