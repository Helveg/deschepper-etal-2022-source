from bsb.core import from_hdf5
from bsb.config import get_result_config
from bsb.plotting import plot_morphology, hdf5_gather_voltage_traces, plot_traces
from scipy import sparse
from plotly import graph_objs as go
import selection, numpy as np, h5py
from colour import Color
from ._paths import *
from glob import glob

def plot(path=None, net_path=None, input_device="mossy_fiber_sensory_burst", buffer=200, cutoff=5000, bin_width=5):
    if path is None:
        paths = glob(results_path("single_impulse", "*.hdf5"))
    if net_path is None:
        net_path = network_path(selection.network)
    network = from_hdf5(net_path)
    for path in paths:
        id = selection.purkinje_cells["On beam"]
        carry = []
        with h5py.File(path, "r") as f:
            res = f["time"].attrs.get("resolution")
            for syn in f[f"recorders/synapses/{id}/current"].values():
                if not syn.attrs.get("type", "").startswith("AMPA"):
                    continue
                data = syn[()]
                d1 = np.diff(data)
                infsmall = np.abs(d1) < 1e-20
                d1[infsmall] = 0
                dropping = d1 < 0
                rise_times = np.nonzero(dropping[1:] & ~dropping[:-1])[0]
                fall_times = np.nonzero(~dropping[1:] & dropping[:-1])[0]
                carry.extend((fall_times - rise_times) * res)
                if any(np.diff(rise_times) < 10):
                    fig = go.Figure()
                    for rise, fall in zip(rise_times, fall_times):
                        print(d1[rise:(fall + 200)])
                        rise_samples = fall - rise
                        rise_time = rise_samples * res
                        fig.add_traces([
                            go.Scatter(mode="lines", y=d1[rise:(fall + 200)]),
                            go.Scatter(mode="lines", y=data[rise:(fall + 200)]),
                        ])
                    fig.show()
                    exit()
        print(np.mean(carry), np.std(carry))

def meta(key):
    return {"width": 1920 / 3 * 2}

if __name__ == "__main__":
    plot().show()
