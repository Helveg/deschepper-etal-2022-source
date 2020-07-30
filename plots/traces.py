from bsb.plotting import hdf5_gather_voltage_traces, plot_traces, CellTraceCollection
from h5py import File

def plot():
    # Open the last generated HDF5 file of the `poc` simulation
    with File('C:/Users/robin/Documents/GIT/deschepper-etal-2020/selected_reps.hdf5', "r") as f:
        # Collect traces from cells across multiple recording groups.
        cell_traces = hdf5_gather_voltage_traces(f, "/", ["somas", "dendrites"])
        # Take only those cells that on top of their soma also had a dendrite recorded
        selected = set([])
        representatives = []
        for c in cell_traces:
            if c.traces[0].meta["label"] not in selected:
                representatives.append(c)
                selected.add(c.traces[0].meta["label"])
        representatives = CellTraceCollection(representatives)
        representatives.set_legends(["Soma (mV)", "Dendrite (mV)"])
        representatives.set_colors(["Crimson", "Blue"])
        fig = plot_traces(representatives, show=False)
    return fig
