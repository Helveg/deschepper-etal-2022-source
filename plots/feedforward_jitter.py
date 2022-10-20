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
from scipy.stats import norm, ttest_ind

import plotly
colors = plotly.colors.DEFAULT_PLOTLY_COLORS

frozen = False

def distr(cont, sp):
    print("M", sp)
    print("C", sp[0:1], sp[1:2], sp[2:3], sp[3:])
    cont[0].extend(sp[0:1])
    cont[1].extend(sp[1:2])
    cont[2].extend(sp[2:3])
    cont["3+"].extend(sp[3:])
    cont["all"].extend(sp)

def plot(net_path=None, wt_path=None, ko_path=None):
    if net_path is None:
        net_path = network_path(selection.network)
    if wt_path is None:
        wt_path = results_path("single_impulse", "sensory_burst", "single_impulse_spot.hdf5")
    if ko_path is None:
        ko_path = results_path("single_impulse", "sensory_burst", "single_impulse_spot_nomli.hdf5")
    network = from_hdf5(net_path)
    cells = selection.onbeam["purkinje_cell"]
    fig = go.Figure()
    if os.path.exists("ffj.pkl"):
        with open("ffj.pkl", "rb") as f:
            pop_spikes = pickle.load(f)
    else:
        pop_spikes = {"wt": {}, "ko": {}}
        for lbl, fp in (("wt", wt_path), ("ko", ko_path)):
            with h5py.File(fp, "r") as f:
                for k, v in f[f"recorders/soma_spikes/"].items():
                    id = int(k)
                    if v.attrs["label"] == "purkinje_cell":
                        spikes = v[:, 1]
                        spikes = spikes[(spikes > 5950) & (spikes < 6050)]
                        pop_spikes[lbl][id] = spikes
        with open("ffj.pkl", "wb") as f:
            pickle.dump(pop_spikes, f)

    x = np.linspace(6000, 6010, 1000)
    wt_spikes = np.concatenate(list(pop_spikes["wt"].values()))
    ko_spikes = np.concatenate(list(pop_spikes["ko"].values()))
    stimulus = wt_spikes[(wt_spikes > 6003) & (wt_spikes < 6006)]
    mu, sd = norm.fit(stimulus)
    p = norm.pdf(x, mu, sd)
    kstimulus = ko_spikes[(ko_spikes > 6004) & (ko_spikes < 6008)]
    kmu, ksd = norm.fit(kstimulus)
    kp = norm.pdf(x, kmu, ksd)
    pval = round(ttest_ind(stimulus, kstimulus).pvalue, 2)
    print(f"p={pval}" if pval else "p<0.01")
    print("jitter:", sd, ksd)
    sfig = make_subplots(rows=2, cols=2, subplot_titles=("Wildtype", "", "Knockout"), specs=[[{}, {"rowspan": 2, "secondary_y": True}], [{}, None]])
    sfig.add_trace(
        go.Scatter(
            x=wt_spikes,
            y=np.concatenate([[int(k)] * len(v) for k,
            v in pop_spikes["wt"].items()]),
            mode="markers",
            marker_size=4,
            showlegend=False
        ),
        row=1,
        col=1,
    )
    sfig.add_trace(
        go.Scatter(
            x=ko_spikes,
            y=np.concatenate([[int(k)] * len(v) for k,
            v in pop_spikes["ko"].items()]),
            mode="markers",
            marker_size=4,
            showlegend=False
        ),
        row=2,
        col=1,
    )
    sfig.add_trace(
        go.Histogram(
            x=wt_spikes[(wt_spikes < 6010) & (wt_spikes > 6000)],
            name="Wildtype",
            nbinsx=50,
            marker_line_width=0,
            # histnorm="probability density",
            marker_color="gray",
        ),
        secondary_y=False,
        row=1, col=2,
    )
    sfig.add_trace(
        go.Histogram(
            x=ko_spikes[(ko_spikes < 6010) & (ko_spikes > 6000)],
            name="Knockout",
            nbinsx=50,
            marker_line_width=0,
            # histnorm="probability density",
            marker_color="darkred",
        ),
        secondary_y=False,
        row=1, col=2,
    )
    sfig.update_layout(barmode="overlay", bargap=0, bargroupgap=0)
    sfig.update_traces(opacity=0.75)
    sfig.add_trace(
        go.Scatter(x=x, y=p, name="Wildtype PDF", mode="lines", line_color="black"),
        secondary_y=True,
        row=1, col=2,
    )
    sfig.add_trace(
        go.Scatter(x=x, y=kp, name="KO PDF", mode="lines", line_color="red"),
        secondary_y=True,
        row=1, col=2,
    )
    sfig.update_yaxes(title="Spikes per bin", row=1, col=2)
    sfig.update_yaxes(title="Probability density", row=1, col=2, secondary_y=True, range=[0, 1])
    sfig.update_xaxes(title="Time (ms)")
    sfig.update_xaxes(range=[5990, 6050], col=1)
    return sfig
