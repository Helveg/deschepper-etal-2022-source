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
    with File(results_path('results_10imp50Hz4mf_LTPLTD.hdf5'), "r") as f:
        # Collect traces from cells across multiple recording groups.
        cell_traces = hdf5_gather_voltage_traces(f, "/recorders/", ["granules_Poiss"])
        #cell_traces1 = hdf5_gather_voltage_traces(f, "/recorders/", ["granules_la"])
        # Take only those cells that on top of their soma also had a dendrite recorded
        selected = "granule_cell"
        #IDs=[3070, 31681, 3069, 9623, 3068, 11800, 3075, 15265, 3083, 16011]
        IDs=[3070, 31681, 3074, 9163, 3083, 11399, 3764, 15288, 5987, 17372]   # 1 act dend, 2 , 3 and 4
        #IDs=[3070, 3069, 3068, 3075,  3083]
        #IDs1=[3074]    #low = 0 act dend

        representatives = []
        for c in cell_traces:
            if (c.traces[0].meta["label"] == selected) and (c.traces[0].meta["cell_id"] in IDs):
                c.order = IDs.index(c.traces[0].meta["cell_id"])+1
                c.title = str(c.traces[0].meta["cell_id"])
                representatives.append(c)
        representatives = CellTraceCollection(representatives)
        representatives.set_legends(["Soma (mV)"])
        representatives.set_colors(["red"])
        fig = plot_traces(representatives, show=False, mod=figmod, cutoff=3000)
        fig.update_layout(xaxis = dict( tickmode='array',
            tickvals=[0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000],
            ticktext=['0', '100', '200', '300', '400', '500', '600', '700', '800', '900']))
        #fig.write_image("grcs.eps")
    return fig
