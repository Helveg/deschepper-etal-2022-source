import os, plotly.graph_objects as go
from bsb.core import from_hdf5
import numpy as np, h5py
from scipy import stats
from scipy.spatial import Delaunay

colorbar_grc = ['rgb(158,188,218)', 'rgb(140,150,198)', 'rgb(140,107,177)', 'rgb(136,65,157)', 'rgb(129,15,124)', 'rgb(77,0,75)']
colorbar_pc = "thermal"
from ._paths import *
from glob import glob
import selection


def crop(data, min, max, indices=False):
    c = data[:, 1]
    if indices:
        return np.where((c > min) & (c < max))[0]
    return c[(c > min) & (c < max)]

def pairs(arr):
    for i in range(len(arr) - 1):
        yield arr[i], arr[i + 1]

def get_isis(spikes, selected):
    if len(selected) == 1:
        return [spikes[selected[0]] - spikes[selected[0] - 1]]
    return [spikes[second] - spikes[first] for first, second in pairs(selected)]

def get_parallel(subset, set):
    return np.where(np.isin(set, subset))[0]

inv = lambda x: [1000 / y for y in x]
avg = lambda x: sum(x) / len(x)

def grc_with_act_dends(network, selected_mf, n=1):
    ps = network.get_placement_set("granule_cell")
    ids = ps.identifiers
    mf_glom = network.get_connectivity_set("mossy_to_glomerulus").get_dataset()
    glom_grc = network.get_connectivity_set("glomerulus_to_granule").get_dataset()
    active_glom = mf_glom[np.isin(mf_glom[:, 0], selected_mf), 1]
    active_dendrites = glom_grc[np.isin(glom_grc[:, 0], active_glom), 1]
    d = dict(zip(*np.unique(active_dendrites, return_counts=True)))
    grc_to_dend = np.vectorize(lambda x: d.get(x, 0))
    return grc_to_dend(ids) >= n

def grc_pos_with_act_dends(network, selected_mf, n=1):
    ps = network.get_placement_set("granule_cell")
    indices = grc_with_act_dends(network, selected_mf, n=1)
    pos = ps.positions[indices]
    return pos

def in_hull(hull):
    print("??")
    def check(p):
        print("checking points")
        return hull.find_simplex(p) >= 0

    return check

def plot(path=None, net_path=None, base_start=5700, base_end=5900, stim_start=6000, stim_end=6040):
    if path is None:
        path = glob(results_path("balanced_sensory", "*.hdf5"))[0]
    if net_path is None:
        net_path = network_path(selection.network)
    print("hellooooooo")
    network = from_hdf5(net_path)
    results = h5py.File(path, "r")
    ps = network.get_placement_set("granule_cell")
    all_pos = ps.positions[:, [0, 2]]
    print("still here")
    pos = grc_pos_with_act_dends(network, selection.stimulated_mf_poiss)
    hull = Delaunay(pos[:, [0, 2]])
    print("still here too")
    in_or_out = np.vectorize(in_hull(hull), signature="(m)->()")(all_pos)
    ids = ps.identifiers[in_or_out]

    print("Total pos:", len(all_pos))
    print("In hull pos:", sum(in_or_out))
    with h5py.File(path, "r") as f:
        cells_firing = sum(bool(len(crop(f["recorders/soma_spikes/" + str(id)], stim_start, stim_end))) for id in ids)
        print("Cells firing during stimulus:", cells_firing)

    fig = go.Figure(layout_title_text="Hello world")
    return fig
