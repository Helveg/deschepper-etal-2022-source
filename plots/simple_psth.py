    # filename = '/home/claudia/deschepper-etal-2020/networks/300x_200z.hdf5'
    # scaffoldInstance = from_hdf5(filename)
    # config = scaffoldInstance.configuration
import os, plotly.graph_objects as go, itertools
from bsb.core import from_hdf5
from bsb.plotting import hdf5_plot_psth, hdf5_plot_spike_raster
import numpy as np, h5py
from scipy import stats
from glob import glob


network_path = os.path.join(
    os.path.dirname(__file__), "..", "networks", "300x_200z.hdf5"
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
    with h5py.File(results_path("results_150Hz50msPoiss_4mf_4Hzbackgr_SynCurrentRecorders_GABAscGmaxX10.hdf5"), "a") as f:
        order=dict(mossy_fiber=0, granule_cell=1, golgi_cell=2, purkinje_cell=3, stellate_cell=4, basket_cell=5)
        #order=dict(granule_cell=0, golgi_cell=1, purkinje_cell=2, stellate_cell=3, basket_cell=4)

        # color=dict(mossy_fiber=config.cell_types["glomerulus"].plotting.color,
        #         granule_cell=config.cell_types["granule_cell"].plotting.color,
        #         golgi_cell=config.cell_types["golgi_cell"].plotting.color,
        #         purkinje_cell=config.cell_types["purkinje_cell"].plotting.color,
        #         stellate_cell=config.cell_types["stellate_cell"].plotting.color,
        #         basket_cell=config.cell_types["basket_cell"].plotting.color)

        #for g in f["/recorders/soma_spikes"].values():
        for g in f["/all"].values():
            if g.attrs["label"] not in order:
                print("Not sorting", g.name, "no order found")
            g.attrs["order"] = order.get(g.attrs["label"], 0)
            #g.attrs['color'] = color.get(g.attrs["label"], 0)
        fig = hdf5_plot_psth(f["/all"], show=False, cutoff=800, duration=5)
        #fig = hdf5_plot_psth(f["/recorders/soma_spikes"], show=False, cutoff=300, duration=5)
        #g.attrs["order"] = order.get(g.attrs["label"], 0)
        # fig = hdf5_plot_spike_raster(f["/recorders/soma_spikes"], show=False)
        #fig.show()
        #fig.show(config=dict(toImageButtonOptions=dict(format="svg", height=1080, width=1920)))
        #fig.write_image("fig1.eps", engine="kaleido")
    return fig
