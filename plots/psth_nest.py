import os, plotly.graph_objects as go
from bsb.core import from_hdf5
from bsb.plotting import hdf5_plot_psth, hdf5_plot_spike_raster
import numpy as np, h5py
from scipy import stats
import plotly.io as pio
from glob import glob
import itertools

network_path = os.path.join(
    os.path.dirname(__file__), "..", "networks", "balanced.hdf5"
)
def results_path(*args):
    return os.path.join(
        os.path.dirname(__file__), "..", "results", *args
    )


class valueify:
    def __init__(self, iter):
        self.iter = iter

    def values(self):
        return self.iter


def plot():
    return go.Figure(layout=dict(title_text="Alice has to provide NEST results file"))
    scaffoldInstance = from_hdf5(network_path)
    config = scaffoldInstance.configuration
    cells = list(scaffoldInstance.get_cell_types())
    order = dict(
        record_glomerulus_spikes=0,
        record_granules_spikes=1,
        record_golgi_spikes=2,
        record_pc_spikes=3,
        record_sc_spikes=4,
        record_bc_spikes=5
    )
    color=dict(
        record_glomerulus_spikes=config.cell_types["glomerulus"].plotting.color,
        record_granules_spikes=config.cell_types["granule_cell"].plotting.color,
        record_golgi_spikes=config.cell_types["golgi_cell"].plotting.color,
        record_pc_spikes=config.cell_types["purkinje_cell"].plotting.color,
        record_sc_spikes=config.cell_types["stellate_cell"].plotting.color,
        record_bc_spikes=config.cell_types["basket_cell"].plotting.color
    )
    figs = {}

    poisson_path = results_path("nest_PoissFinal2")
    handles = [h5py.File(f, "a") for f in glob(os.path.join(poisson_path, "*.hdf5"))]
    try:
        for run_id, handle in enumerate(handles):
            for g in handle["/recorders/soma_spikes"].values():
                if g.attrs["label"] not in order:
                    print("Not sorting", g.name, "no order found")
                g.attrs["order"] = order.get(g.attrs["label"], 0)
                g.attrs['color'] = color.get(g.attrs["label"], 0)
            for ds in handle["/recorders/soma_spikes"].values():
                # if ds.attrs.get("run_id", None) is not None:
                #     break
                ds.attrs["run_id"] = run_id
            else:
                continue
            print("Run ids already detected in:", handle.filename)
            print("Skipping run id insertion")
            break
        figs = {}
        figs["poiss"] = fig = hdf5_plot_psth(
            scaffoldInstance,
            valueify(itertools.chain(*(handle["/recorders/soma_spikes"].values() for handle in handles))),
            show=False,
            cutoff=300,
            duration=5
        )
        ranges = [[0, 10], [0, 10], [0, 65], [0, 75], [0, 60], [0, 60]]
        for i in range(len(ranges)):
            fig.update_yaxes(range=ranges[i], row=i + 1, col=1)
    finally:
        for handle in handles:
            handle.close()
    with h5py.File(results_path("results_NEST_stim_on_MFs_4syncFinal2_finetuning0-02_noMLIPC.hdf5"), "a") as f:
        for g in f["/recorders/soma_spikes"].values():
            if g.attrs["label"] not in order:
                print("Not sorting", g.name, "no order found")
            g.attrs["order"] = order.get(g.attrs["label"], 0)
            g.attrs['color'] = color.get(g.attrs["label"], 0)
        figs["sync"] = fig = hdf5_plot_psth(scaffoldInstance, f["/recorders/soma_spikes"], show=False, cutoff=300, duration=5)
        ranges = [[0, 30], [0, 29], [0, 125], [0, 130], [0, 120], [0, 120]]
        for i in range(len(ranges)):
            fig.update_yaxes(range=ranges[i], row=i + 1, col=1)
    return figs

def meta(key):
    return {"width": 1920 / 2}
