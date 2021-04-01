from bsb.plotting import hdf5_gather_voltage_traces, plot_traces, CellTraceCollection
import h5py, os, random, numpy as np

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
    i = 0
    with h5py.File("in_memory", driver="core", mode='w', backing_store=False) as f:
        with h5py.File("results/results_stim_on_1MF_noGABA.hdf5", "r") as o:
            # Select 5 random GrC
            tags = random.sample(list(o["recorders/granules_mf214"].keys()), 50)
            for tag in tags:
                src = o["recorders/granules_mf214/" + tag]
                o.copy(src, f, name=tag)
                f[tag].attrs["order"] = i
                i = i + 1
        traces = hdf5_gather_voltage_traces(f, "/")
        traces.set_legends(["Membrane potential"])
        traces.set_colors(["Crimson"])
        fig = plot_traces(traces, x=list(np.arange(0, 1000, 0.1)), input_region=[400, 500], cutoff=8000, show=False)
        for i in range(len(traces.cells)):
            fig.update_yaxes(range=[-80, 45], row=i)
        return fig
