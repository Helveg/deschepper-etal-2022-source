import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "plots"))

import numpy as np
import plotly.graph_objects as go
import h5py
from scipy import signal, fft, blackman
import selection
from plots._paths import *
from glob import glob
from scipy.stats import ttest_ind

def table():
    rows = check(results_path("balanced_sensory", "*.hdf5"))
    rows_gaba = check(results_path("balanced_sensory", "gabazine", "*.hdf5"), T=True)
    p = ttest_ind(rows, rows_gaba)[1]
    table = [["control", "gabazine", "std_c", "std_g", "p"], [np.mean(x) for x in (rows, rows_gaba)] + [np.std(x) for x in (rows, rows_gaba)] + [p]]
    return table

def check(path, net_path=None, T=False):
    if net_path is None:
        net_path = network_path(selection.network)

    figs = {}
    population = "golgi_cell"
    rows = []

    for path in glob(path):
        with h5py.File(path, "r") as handle:
            ISIs = np.concatenate(list(np.diff((ds if not T else ds[()].T)[:, 1]) for ds in handle["recorders/soma_spikes"].values() if ds.attrs["label"] == population))
            shortest = np.min(ISIs)
            print("Shortest ISI:", shortest)
            rows.append(shortest)

    return rows

def meta():
    return {"width": 1920 / 2, "height": 1920 / 2}
