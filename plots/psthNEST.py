import os, plotly.graph_objects as go
from bsb.core import from_hdf5
from bsb.plotting import hdf5_plot_psth, hdf5_plot_spike_raster
import numpy as np, h5py
from scipy import stats
import plotly.io as pio


network_path = os.path.join(
    os.path.dirname(__file__), "..", "networks", "300x_200z.hdf5"
)
def results_path(*args):
    return os.path.join(
        os.path.dirname(__file__), "..", "results", *args
    )


def plot():
    filename = '/home/claudia/deschepper-etal-2020/networks/300x_200z.hdf5'
    scaffoldInstance = from_hdf5(filename)
    config = scaffoldInstance.configuration

    with h5py.File(results_path("results_stim_on_MFs_Poiss_NEST.hdf5"), "a") as f:
        order=dict(record_glomerulus_spikes=0, record_granules_spikes=1, record_golgi_spikes=2, record_pc_spikes=3,
            record_sc_spikes=4, record_bc_spikes=5)
        color=dict(record_glomerulus_spikes=config.cell_types["glomerulus"].plotting.color,
                record_granules_spikes=config.cell_types["granule_cell"].plotting.color,
                record_golgi_spikes=config.cell_types["golgi_cell"].plotting.color,
                record_pc_spikes=config.cell_types["purkinje_cell"].plotting.color,
                record_sc_spikes=config.cell_types["stellate_cell"].plotting.color,
                record_bc_spikes=config.cell_types["basket_cell"].plotting.color)
        for g in f["/recorders/soma_spikes"].values():
            if g.attrs["label"] not in order:
                print("Not sorting", g.name, "no order found")
            g.attrs["order"] = order.get(g.attrs["label"], 0)
            g.attrs['color'] = color.get(g.attrs["label"], 0)
        fig = hdf5_plot_psth(f["/recorders/soma_spikes"], show=False, cutoff=300, duration=5)
        # fig = hdf5_plot_spike_raster(f["/recorders/soma_spikes"], show=False)
        #fig.show()
        #fig.show(config=dict(toImageButtonOptions=dict(format="svg", height=1080, width=1920)))
        #fig.write_image("fig1.eps", engine="kaleido")
    return fig

# dset.attrs['label'] = c
# dset.attrs['color'] = color[c]
# dset.attrs['num_neurons'] = len(np.unique(spikes[:,0]))
# dset.attrs['mean_rate'] = (len(spikes[:,0])/dset.attrs['num_neurons'])/duration
