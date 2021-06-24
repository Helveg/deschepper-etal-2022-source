from bsb.core import from_hdf5
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import selection, grc_cloud
from colour import Color
import numpy as np, h5py, pickle
from glob import glob
from bsb.plotting import plot_morphology
from ._paths import *
import os

frozen = True

def plot(run_path=None, net_path=None):
    if net_path is None:
        net_path = network_path(selection.network)
    if run_path is None:
        run_path = results_path("clamp")
    return plot_average_all(net_path, run_path)


def plot_average_all(network_path, run_path):
    network = from_hdf5(network_path)
    cell = selection.purkinje_cells["High activity"]
    grc_color = network.configuration.cell_types["granule_cell"].plotting.color
    bc_color = network.configuration.cell_types["basket_cell"].plotting.color
    ampa_currents = []
    gaba_currents = []
    if not frozen or not os.path.exists("ff.pickle"):
        files = glob(run_path + "/*.hdf5")
        ti = len(files)
        for i, file in enumerate(files):
            print("Processing", i, file)
            with h5py.File(file, "r") as f:
                cells = list(map(int, f["recorders/synapses"].keys()))
                cells = [cell]
                time = f["time"][()]
                time_mask = (time >= 5500) & (time <= 6500)
                tj = len(cells)
                for j, cell in enumerate(cells):
                    print(f"Run {i+1}/{ti}, cell {j+1}/{tj}", end="\r", flush=True)
                    ampa_current = sum(d[time_mask] for d in f[f"recorders/synapses/{cell}/current"].values() if d.attrs.get("type", None) == "AMPA")
                    gaba_current = sum(d[time_mask] for d in f[f"recorders/synapses/{cell}/current"].values() if d.attrs.get("type", None) == "GABA")
                    ampa_currents.append(ampa_current)
                    gaba_currents.append(gaba_current)
                time = f["time"][()]
        avg_ampa_current = sum(ampa_currents) / len(ampa_currents)
        avg_gaba_current = sum(gaba_currents) / len(gaba_currents)
        with open("ff.pickle", "wb") as f:
            pickle.dump((time[time_mask], avg_ampa_current, avg_gaba_current), f)
    else:
        with open("ff.pickle", "rb") as f:
            time, avg_ampa_current, avg_gaba_current = pickle.load(f)
    norm_ampa = avg_ampa_current / min(avg_ampa_current)
    norm_gaba = avg_gaba_current / max(avg_gaba_current)
    start = 5950
    end = 6050
    time_mask = (time >= start) & (time <= end)
    dt = time[1] - time[0]
    time = time[time_mask]
    # First autocorr on the full signal
    span = len(norm_gaba) * dt
    corr_x = np.arange(0, span, time[1] - time[0]) - span / 2
    corr_y = np.correlate(norm_gaba, norm_ampa, mode="same")
    norm_ampa = norm_ampa[time_mask]
    norm_gaba = norm_gaba[time_mask]
    span = len(norm_gaba) * dt
    # Then zoom in on the RoI
    corr_fig = make_subplots(rows=2, cols=1, vertical_spacing=0.05, x_title="Time (ms)")
    corr_fig = corr_fig.add_trace(go.Scatter(x=corr_x, y=corr_y, name="Cross corr."), row=2, col=1)
    max_x = np.argmax(corr_y)
    max_y = corr_y[max_x]
    corr_fig.add_annotation(x=corr_x[max_x], y=max_y, text=round(corr_x[max_x], 2), startstandoff=13, standoff=3, ay=20, ax=70, row=2, col=1)
    corr_fig.add_trace(go.Scatter(x=time, y=norm_ampa, name="Norm. AMPA", xaxis="x2", line=dict(color=grc_color)), row=1, col=1)
    corr_fig.add_trace(go.Scatter(x=time, y=norm_gaba, name="Norm. GABA", line=dict(color=bc_color)), row=1, col=1)
    corr_fig.update_yaxes(visible=False)
    # corr_fig.update_yaxes(range=[max_y / 1.5, max_y * 1.01], row=2, col=1)

    return corr_fig

def meta():
    return {"width": 800, "height": 800}
