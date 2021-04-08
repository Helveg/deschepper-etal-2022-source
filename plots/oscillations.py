import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from bsb.core import Scaffold
from bsb.config import JSONConfig, get_result_config
from bsb.output import MorphologyRepository

import numpy as np
from bsb.plotting import *
import plotly.graph_objects as go
import h5py
from scipy import signal, fft
import collections
from collections import defaultdict
import selection
from ._paths import *
from glob import glob
import math

def plot(path=None, net_path=None):
    if path is None:
        path = results_path("oscillations", "oscillations_nomf.hdf5")
    if net_path is None:
        net_path = network_path(selection.network)

    cfg = get_result_config(path)
    figs = {}

    with h5py.File(path, "r") as handle:
        bin_width = 100
        win_width = 51 # bins
        cutoff = 1
        duration = 4
        zero_skip = 5
        population = "golgi_cell"

        time = handle["time"][()] / 1000
        x = time[(time >= cutoff) & (time <= duration)]
        N = len(x)
        T = list(cfg.simulations.values())[0].resolution / 1000
        t = round(max(x))

        N = math.ceil(N / bin_width)
        T *= bin_width

        osc_f = 8
        n_periods = t * osc_f
        period_t = 1 / osc_f
        period = N / (t * osc_f)
        n_spikes = 100000

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
    from scipy.signal import blackman
    from statsmodels.tsa.stattools import acf
    w = blackman(N)
    sig = binned_spikes
    wsig = binned_spikes * w
    fft_sig = 2.0/N * np.abs(fft.fft(sig)[zero_skip:N//2])
    fft_wsig = 2.0/N * np.abs(fft.fft(wsig)[zero_skip:N//2])
    fft_freq = fft.fftfreq(N, T)[zero_skip:N // 2]
    fft_filtered = signal.savgol_filter(fft_sig, win_width, 6)
    fft_wfiltered = signal.savgol_filter(fft_wsig, win_width, 6)
    figs["fft"] = go.Figure(
        [
            go.Scatter(
                x=x, y=sig, name="signal"
            ),
            go.Scatter(
                x=x[::bin_width], y=acf(sig, nlags=len(sig), fft=True), name="sig_autocorr"
            ),
            go.Scatter(
                x=fft_freq, y=fft_sig, name="fft"
            ),
            go.Scatter(
                x=fft_freq, y=fft_filtered, name="fft_filtered"
            ),
            go.Scatter(
                x=fft_freq, y=acf(fft_filtered, nlags=1000, fft=True), name="fft_autocorr"
            ),
            go.Scatter(
                x=fft_freq, y=fft_wsig, name="fftw"
            ),
            go.Scatter(
                x=fft_freq, y=fft_wfiltered, name="fftw_filtered"
            ),
        ],
        layout_title_text="fft"
    )

    return figs["fft"]
