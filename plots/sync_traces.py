import h5py, selection, numpy as np
from bsb.core import from_hdf5
from bsb.plotting import plot_traces, hdf5_gather_voltage_traces

def plot():
    i = 0
    network = from_hdf5("networks/300x_200z.hdf5")
    with h5py.File("in_memory", driver="core", mode='w', backing_store=False) as f:
        with h5py.File("results/results_stim_on_MFs_4syncImp.hdf5", "r") as o:
            for tag, ids in selection.sync.items():
                for id in ids:
                    base = o["/recorders/soma_voltages"]
                    src = str(id)
                    if tag == "granule_cell":
                        base = o["/recorders/granules_syncImp"]
                        src = [k for k in base if k.startswith(str(id))][0]
                    base.copy(src, f, name=str(id))
                    f[str(id)].attrs["order"] = i
                    i = i + 1
        traces = hdf5_gather_voltage_traces(f, "/")
        traces.set_legends(["Membrane potential"])
        for cell_traces in traces.cells.values():
            for k,v in selection.sync.items():
                if cell_traces.cell_id in v:
                    for trace in cell_traces.traces:
                        trace.color = network.configuration.cell_types[k].plotting.color
        fig = plot_traces(traces, x=list(np.arange(0, 900, 0.1)), input_region=[400, 500], cutoff=3000, show=False)
        for i in range(len(traces.cells)):
            fig.update_yaxes(range=[-80, 45], row=i)
        return fig

if __name__ == "__main__":
    plot().show()
