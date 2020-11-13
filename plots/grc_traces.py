from bsb.plotting import hdf5_gather_voltage_traces, plot_traces, CellTraceCollection
from bsb.core import from_hdf5
from h5py import File
import os, selection

network_path = os.path.join(
    os.path.dirname(__file__), "..", "networks", "300x_200z.hdf5"
)
def results_path(*args):
    return os.path.join(
        os.path.dirname(__file__), "..", "results", *args
    )

def plot():
    network = from_hdf5(network_path)
    # Open the last generated HDF5 file of the `poc` simulation
    with File(results_path('results_stim_on_MFs_Poiss.hdf5'), "r") as f:
        # Collect traces from cells across multiple recording groups.
        traces = hdf5_gather_voltage_traces(f, "/recorders/", ["granules_Poiss"])
        traces.set_legends(["Soma (mV)"])
        traces.set_colors([network.configuration.cell_types["granule_cell"].plotting.color])
        traces.reorder(map(selection.granule_cell_order.get, traces.cells.keys()))
        for i, t in enumerate(traces.cells.values()):
            r = int(i / 2)
            t.title = f"{r} active dendrite{'s' if r != 1 else ''}"
        def figmod(fig):
            for i in range(len(traces.cells)):
                fig.update_yaxes(range=[-75, 40], row=i)
        fig = plot_traces(traces, show=False, mod=figmod)
    return fig
