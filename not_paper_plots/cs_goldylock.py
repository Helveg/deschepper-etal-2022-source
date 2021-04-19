import plotly.graph_objs as go
from ._paths import *
from scipy import signal
import h5py
import numpy as np

def pie(dends):
    l = list(range(5)) + list(range(-4, 0))
    print("xlist", l)
    dends = list(d[0] for d in dends)
    print("disinh", dends)
    return go.Bar(x=l, y=list(reversed(dends)) + dends)

def get_pop_spike_strength(h5f, rancz, coherence, dendrites, gabazine):
    trials = h5f[f"{rancz}/{coherence}/{dendrites}/{gabazine}"]
    spikes = []
    for name, d in trials.items():
        t, v = d[:, 0], d[:, 1]
        peaks, info = signal.find_peaks(v, height=-20)
        pmask = (t[peaks] > 500) & (t[peaks] < max(t) - 60)
        peaks = peaks[pmask]
        n_peaks = sum(pmask)
        spikes.append(n_peaks)
        # if dendrites == 4:
        #     go.Figure([go.Scatter(x=t, y=v), go.Scatter(x=t[peaks],y=v[peaks], mode="markers", name="peaks")], layout_title_text=f"id: {name}, r: {rancz}, c: {coherence}, g: {gabazine}").show()
    return np.mean(spikes), np.std(spikes)

def get_pop_spike_diff(h5f, rancz, coherence, dendrites):
    g, gs = get_pop_spike_strength(h5f, rancz, coherence, dendrites, True)
    c, cs = get_pop_spike_strength(h5f, rancz, coherence, dendrites, False)
    return g - c, gs + cs


def plot(path=None):
    if path is None:
        path = results_path("weekend-rancor.hdf5")
    with h5py.File(path, "r") as h5f:
        ps, std = get_pop_spike_strength(h5f, 1.2, 0.5, 3, True)
        print("Pop strength:", ps, "+-", std)
        for rancz in np.linspace(0.1, 2, 20):
            for coh in (0.05, 0.25, 0.5, 0.75, 1):
                disinh = [get_pop_spike_diff(h5f, rancz, 0.5, n) for n in range(5)]
                go.Figure(pie(disinh), layout_title_text=f"{rancz}, {coh}").show()
