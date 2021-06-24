from bsb.core import from_hdf5
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import plotly.graph_objects as go
import h5py, itertools, glob
from plots._paths import *
import plots.selection as selection

def table(path=None, net_path=None, cutoff=4000, duration=8000):
    if path is None:
        path = network_path(selection.network)
    path = glob.glob(path)[0]
    table = list()
    table.append(["conn_name", "conv_mean", "conv_stdev", "div_mean", "div_stdev", "n_syn", "syn_pp_mean", "syn_pp_std"])
    network = from_hdf5(path)
    for cs in network.get_connectivity_sets():
        div = cs.get_divergence_list()
        conv = cs.get_convergence_list()
        syn_per_pair = np.unique(cs.get_dataset(), axis=0, return_counts=True)[1]
        table.append([cs.tag, np.mean(conv), np.std(conv), np.mean(div), np.std(div), len(cs), np.mean(syn_per_pair), np.std(syn_per_pair)])
    return table
