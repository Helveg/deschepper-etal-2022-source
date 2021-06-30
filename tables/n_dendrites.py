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
    table.append(["n_dend", "n"])
    network = from_hdf5(path)
    ps = network.get_placement_set("granule_cell")
    ids = ps.identifiers
    MFs = selection.stimulated_mf_poiss
    mf_glom = network.get_connectivity_set("mossy_to_glomerulus").get_dataset()
    glom_grc = network.get_connectivity_set("glomerulus_to_granule").get_dataset()
    active_glom = mf_glom[np.isin(mf_glom[:, 0], MFs), 1]
    active_dendrites = glom_grc[np.isin(glom_grc[:, 0], active_glom), 1]
    d = dict(zip(*np.unique(active_dendrites, return_counts=True)))
    grc_to_dend = np.vectorize(lambda x: d.get(x, 0))
    x = grc_to_dend(ids)
    n_dend, n = np.unique(x, return_counts=True)
    table.extend(zip(n_dend, n))
    return table
