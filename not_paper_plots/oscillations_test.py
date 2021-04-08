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
        path = results_path("oscillations", "results_test_oscillations_20Hz.hdf5")
    if net_path is None:
        net_path = network_path(selection.network)

    cfg = get_result_config(path)
    figs = {}

    ###### PERIODIC NEURAL OSCILLATION

    # t = 8
    # N = 320001
    # T = 0.000025

    with h5py.File(path, "r") as handle:
        bin_width = 100
        win_width = 51 # bins
        cutoff = 4
        duration = 8

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



    periods = np.floor(np.random.rand(n_spikes) * n_periods)
    phase = np.floor(np.random.normal(0.5, 0.1, size=n_spikes) * period)
    spikes = periods * period_t + phase * T
    spikes = spikes[(spikes > cutoff) & (spikes < duration)] - cutoff
    binned_spikes = np.bincount((spikes / T).astype(int))

    figs["raster_periodic"] = go.Figure(
        [
            go.Scatter(x=spikes, y=np.floor(np.random.rand(n_spikes) * 100), mode="markers", marker_size=2),
            go.Scatter(x=x, y=binned_spikes)
        ],
        layout_title_text="raster periodic"
    )

    sig = binned_spikes
    # autocorr_sig = np.correlate(sig, sig, mode="full")
    fft_sig = 2.0/N * np.abs(fft.fft(sig)[0:N//2])
    fft_freq = fft.fftfreq(N, T)[:N // 2]

    figs["pneural"] = go.Figure(
        [
            go.Scatter(
                x=x, y=sig, name="signal"
            ),
            go.Scatter(
                x=fft_freq, y=fft_sig, name="fft"
            ),
            # go.Scatter(
            #     y=autocorr_sig, name="autocorr"
            # ),
        ],
        layout_title_text="periodic"
    )

    period_options = np.cumsum(1 / np.random.normal(loc=osc_f, scale=1.4, size=n_periods))
    periods = np.floor(np.random.rand(n_spikes) * n_periods).astype(int)
    phase = np.floor(np.random.normal(0.5, 0.001, size=n_spikes) * period)
    spikes = period_options[periods] + phase * T
    spikes = spikes[(spikes > cutoff) & (spikes < duration)] - cutoff
    binned_spikes = np.bincount((spikes / T).astype(int))
    sig = binned_spikes
    fft_sig = 2.0/N * np.abs(fft.fft(sig)[0:N//2])
    fft_freq = fft.fftfreq(N, T)[:N // 2]
    figs["pneural_cyclic"] = go.Figure(
        [
            go.Scatter(
                x=x, y=sig, name="signal"
            ),
            go.Scatter(
                x=fft_freq, y=fft_sig, name="fft"
            ),
        ],
        layout_title_text="cyclic"
    )

    figs["raster_cyclic"] = go.Figure(
        [
            go.Scatter(x=spikes, y=list(i % 100 for i in range(len(spikes))), mode="markers", marker_size=2),
            go.Scatter(x=x, y=binned_spikes)
        ],
        layout_title_text="raster cyclic"
    )
    fft_sig = signal.savgol_filter(fft_sig, win_width, 4)
    fft_freq = fft.fftfreq(N, T)[:N // 2]
    figs["filtered_cyclic"] = go.Figure(
        [
            go.Scatter(
                x=x, y=sig, name="signal"
            ),
            go.Scatter(
                x=fft_freq, y=fft_sig, name="fft"
            ),
        ],
        layout_title_text="filtered cyclic"
    )

    ########## THE REAL DEAL

    with h5py.File(path, "r") as handle:
        spikes = np.concatenate(list(ds[:, 1] for ds in handle["recorders/soma_spikes"].values() if ds.attrs["label"] == "golgi_cell"))
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
    w = blackman(N)
    sig = binned_spikes
    wsig = binned_spikes * w
    fft_sig = 2.0/N * np.abs(fft.fft(sig)[10:N//2])
    fft_wsig = 2.0/N * np.abs(fft.fft(wsig)[10:N//2])
    fft_freq = fft.fftfreq(N, T)[10:N // 2]
    fft_filtered = signal.savgol_filter(fft_sig, win_width, 6)
    fft_wfiltered = signal.savgol_filter(fft_wsig, win_width, 6)
    figs["fft"] = go.Figure(
        [
            go.Scatter(
                x=x, y=sig, name="signal"
            ),
            go.Scatter(
                x=fft_freq, y=fft_sig, name="fft"
            ),
            go.Scatter(
                x=fft_freq, y=fft_filtered, name="fft_filtered"
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

    ###### SINUSWAVE

    N = 320001
    T = 0.000025

    x = np.linspace(0, N * T , N , endpoint=False)
    sig = np.sin(2.0 * np.pi * x) + np.sin(2 * np.pi * x * 3.4)
    # autocorr_sig = np.correlate(sig, sig, mode="full")
    fft_sig = 2.0/N * np.abs(fft.fft(sig)[0:N//2])
    fft_freq = fft.fftfreq(N, T)[:N // 2]

    figs["sinus"] = go.Figure(
        [
            go.Scatter(
                x=x, y=sig, name="signal"
            ),
            go.Scatter(
                x=fft_freq, y=fft_sig, name="fft"
            ),
            # go.Scatter(
            #     y=autocorr_sig, name="autocorr"
            # ),
        ],
        layout_title_text="sinusoidal"
    )
    del figs["sinus"]

    return figs["fft"]
