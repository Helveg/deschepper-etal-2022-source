import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from bsb.core import Scaffold
from bsb.config import JSONConfig, get_result_config
from bsb.output import MorphologyRepository

import numpy as np
from bsb.plotting import *
import plotly.graph_objects as go
import h5py
from scipy import signal, fft, blackman
import collections
from collections import defaultdict
import selection
from ._paths import *
from glob import glob
import math

def plot(path=None, net_path=None):
    if path is None:
        path = results_path("oscillations", "oscillations_4Hz.hdf5")
    if net_path is None:
        net_path = network_path(selection.network)

    cfg = get_result_config(path)
    figs = {}

    with h5py.File(path, "r") as handle:
        bin_width = 100
        win_width = 51 # bins
        cutoff = 4
        duration = 8
        zero_skip = 5
        population = "granule_cell"

        time = handle["time"][()] / 1000
        x = time[(time >= cutoff) & (time <= duration)]
        N = len(x)
        T = list(cfg.simulations.values())[0].resolution / 1000
        t = round(max(x))

        N = math.ceil(N / bin_width)
        T *= bin_width

    with h5py.File(path, "r") as handle:
        spikes = np.concatenate(list(ds[:, 1] for ds in handle["recorders/soma_spikes"].values() if ds.attrs["label"] == population))
        spikes /= 1000
        spikes = spikes[(spikes > cutoff) & (spikes < duration)] - cutoff
        binned_spikes = np.bincount(np.floor(spikes / T).astype(int))
        N = len(binned_spikes)
        figs["raster"] = go.Figure(
            [
                go.Scatter(x=spikes, y=list(i % 100 for i in range(len(spikes))), mode="markers", marker_size=2),
                go.Scatter(x=x, y=binned_spikes)
            ],
            layout_title_text="raster " + path
        )
    w = blackman(N)
    sig = binned_spikes
    wsig = binned_spikes * w
    fft_wsig = 2.0/N * np.square(np.abs(fft.fft(wsig)[zero_skip:N//2]))
    fft_freq = fft.fftfreq(N, T)[zero_skip:N // 2]
    fft_wfiltered = signal.savgol_filter(fft_wsig, win_width, 6)
    figs["fft"] = go.Figure(
        [
            go.Scatter(
                x=fft_freq, y=fft_wsig, name="granule cells unfiltered"
            ),
            go.Scatter(
                x=fft_freq, y=fft_wfiltered, name="granule cells"
            ),
        ],
        layout=dict(
            title_text="Fourier Transform of granule cell population",
            xaxis_title="Frequency (Hz)",
            yaxis_title="Power",
        ),
    )

    return figs["fft"]
