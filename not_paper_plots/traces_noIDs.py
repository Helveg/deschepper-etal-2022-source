from bsb.plotting import hdf5_gather_voltage_traces, plot_traces, CellTraceCollection
from h5py import File
import os

network_path = os.path.join(
    os.path.dirname(__file__), "..", "networks", "balanced.hdf5"
)
def results_path(*args):
    return os.path.join(
        os.path.dirname(__file__), "..", "results", *args
    )

def figmod(fig):
    pass

def plot():
    # Open the last generated HDF5 file of the `poc` simulation
    with File(results_path('sensory_gabazine','sensory_burst_control.hdf5'), "r") as f:
        # Collect traces from cells across multiple recording groups.
        cell_traces = hdf5_gather_voltage_traces(f, "/recorders/", ["soma_voltages"])

        selected = "golgi_cell"
        #IDs=[53]
        representatives = []
        skipped = 0
        for c in cell_traces:
            if (c.traces[0].meta["label"] == selected) and len(representatives) < 10: # and (c.traces[0].meta["cell_id"] in IDs):
                skipped += 1
                if skipped < 30:
                    continue
                #c.order = IDs.index(c.traces[0].meta["cell_id"])+1
                c.title = str(c.traces[0].meta["cell_id"])
                representatives.append(c)

        print(len(representatives))
        representatives = CellTraceCollection(representatives)
        representatives.set_legends(["Soma (mV)"])
        representatives.set_colors(["orange"])
        fig = plot_traces(representatives, x=list(f["time"]), range=[5500, 6500], show=False) #input_region=[6000, 6100], range=[5800, 6300],
        #fig.write_image("pc.eps")

    return fig
