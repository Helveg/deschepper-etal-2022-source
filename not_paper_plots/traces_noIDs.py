from bsb.plotting import hdf5_gather_voltage_traces, plot_traces, CellTraceCollection
from h5py import File
import os

network_path = os.path.join(
    os.path.dirname(__file__), "..", "networks", "300x_200z.hdf5"
)
def results_path(*args):
    return os.path.join(
        os.path.dirname(__file__), "..", "results", *args
    )

def figmod(fig):
    pass

def plot():
    # Open the last generated HDF5 file of the `poc` simulation
    with File(results_path('results_1imp4mf_CaCon_V_SynSpikesAndCurr_Alldendr.hdf5'), "r") as f:
        # Collect traces from cells across multiple recording groups.
        cell_traces = hdf5_gather_voltage_traces(f, "/recorders/", ["soma_voltages"])

        selected = "golgi_cell"
        IDs=[53]
        representatives = []
        for c in cell_traces:
            if (c.traces[0].meta["label"] == selected) and (c.traces[0].meta["cell_id"] in IDs):
                #c.order = IDs.index(c.traces[0].meta["cell_id"])+1
                c.title = str(c.traces[0].meta["cell_id"])
                representatives.append(c)

        print(len(representatives))
        representatives = CellTraceCollection(representatives)
        representatives.set_legends(["Soma (mV)"])
        representatives.set_colors(["orange"])
        fig = plot_traces(representatives, show=False, mod=figmod, cutoff=0)
        #fig.write_image("pc.eps")

    return fig
