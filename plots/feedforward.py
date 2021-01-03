from bsb.core import from_hdf5
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import selection, grc_cloud
from colour import Color
import numpy as np, h5py
from glob import glob
from bsb.plotting import plot_morphology


def plot():
    return plot_average_all()


def plot_average_all():
    network = from_hdf5("networks/300x_200z.hdf5")
    grc_color = network.configuration.cell_types["granule_cell"].plotting.color
    bc_color = network.configuration.cell_types["basket_cell"].plotting.color
    ampa_currents = []
    gaba_currents = []
    files = glob("results/mli/feedforward/*.hdf5")
    ti = len(files)
    for i, file in enumerate(files):
        with h5py.File(file, "r") as f:
            cells = list(map(int, f["recorders/synapses"].keys()))
            tj = len(cells)
            for j, cell in enumerate(cells):
                print(f"Run {i+1}/{ti}, cell {j+1}/{tj}")
                ampa_current = sum(d[3000:] for d in f[f"recorders/synapses/{cell}/current"].values() if d.attrs.get("type", None) == "AMPA")
                gaba_current = sum(d[3000:] for d in f[f"recorders/synapses/{cell}/current"].values() if d.attrs.get("type", None) == "GABA")
                ampa_currents.append(ampa_current)
                gaba_currents.append(gaba_current)
            time = np.arange(0, f.attrs.get("duration", 1000) - 300, f.attrs.get("resolution", 0.1))
    avg_ampa_current = sum(ampa_currents) / len(ampa_currents)
    norm_ampa = avg_ampa_current / min(avg_ampa_current[4000:4100])
    avg_gaba_current = sum(gaba_currents) / len(gaba_currents)
    norm_gaba = (avg_gaba_current - 0.022) / max(avg_gaba_current[4000:4100] - 0.022)
    corr_x = np.arange(0, len(norm_gaba) / 10, 0.1) - len(norm_gaba) / 20
    corr_y = np.correlate(norm_gaba, norm_ampa, mode="same")
    corr_fig = make_subplots(rows=2, cols=2, specs=[[{"rowspan": 2}, {}], [None, {}]], vertical_spacing=0.05, y_title="Current (nA)", x_title="Time (ms)", subplot_titles=("Synaptic currents", "Normalized and rectified currents", "Cross correlation"))
    corr_fig = corr_fig.add_trace(go.Scatter(x=corr_x, y=corr_y, name="Cross corr."), row=2, col=2)
    max_x = np.argmax(corr_y)
    max_y = corr_y[max_x]
    corr_fig.add_annotation(x=corr_x[max_x], y=max_y, text=round(corr_x[max_x], 2), standoff=3, row=2, col=2)
    corr_fig.add_trace(go.Scatter(x=time, y=avg_ampa_current, name="AMPA", line=dict(color=grc_color)), row=1, col=1)
    corr_fig.add_trace(go.Scatter(x=time, y=avg_gaba_current, name="GABA", line=dict(color=bc_color)), row=1, col=1)
    corr_fig.add_trace(go.Scatter(x=time, y=norm_ampa, name="Norm. AMPA", line=dict(color=grc_color)), row=1, col=2)
    corr_fig.add_trace(go.Scatter(x=time, y=norm_gaba, name="Norm. GABA", line=dict(color=bc_color)), row=1, col=2)
    corr_fig.update_xaxes(range=[350, 450])
    corr_fig.update_xaxes(range=[-50, 50], row=2, col=2)

    return corr_fig

def meta():
    return {"width": 1920 / 2}
