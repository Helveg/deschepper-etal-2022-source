import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "plots"))
from bsb.core import from_hdf5
import selection
from _paths import *
import numpy as np

def table():
    network = from_hdf5(network_path(selection.network))
    ps = network.get_placement_set("granule_cell")
    ids = ps.identifiers
    MFs = selection.stimulated_mf_poiss
    mf_glom = network.get_connectivity_set("mossy_to_glomerulus").get_dataset()
    glom_grc = network.get_connectivity_set("glomerulus_to_granule").get_dataset()
    active_glom = mf_glom[np.isin(mf_glom[:, 0], MFs), 1]
    active_dendrites = glom_grc[np.isin(glom_grc[:, 0], active_glom), 1]
    d = dict(zip(*np.unique(active_dendrites, return_counts=True)))
    grc_to_dend = np.vectorize(lambda x: d.get(x, 0))
    table = [["label", "set", "abs", "pct"]]
    for name, id, sets in zip(
        ("onb_pc", "offb_pc", "h_gc", "l_gc", "bc", "sc"),
        (
            selection.purkinje_cells["On beam"],
            selection.purkinje_cells["Off beam"],
            selection.golgi_cells["High activity"],
            selection.golgi_cells["Low activity"],
            selection.basket_cells["High activity"],
            selection.stellate_cells["High activity"],
        ),
        (
            (network.get_connectivity_set("ascending_axon_to_purkinje"), network.get_connectivity_set("parallel_fiber_to_purkinje")),
            (network.get_connectivity_set("ascending_axon_to_purkinje"), network.get_connectivity_set("parallel_fiber_to_purkinje")),
            (network.get_connectivity_set("ascending_axon_to_golgi"), network.get_connectivity_set("parallel_fiber_to_golgi")),
            (network.get_connectivity_set("ascending_axon_to_golgi"), network.get_connectivity_set("parallel_fiber_to_golgi")),
            (network.get_connectivity_set("parallel_fiber_to_basket"),),
            (network.get_connectivity_set("parallel_fiber_to_stellate"),),
        )
    ):
        for set in sets:
            label = set.tag
            data = set.get_dataset()
            grc_ids = data[data[:, 1] == id, 0]
            abs = sum(grc_to_dend(grc_ids) >= 2)
            pct = abs / len(grc_ids)
            table.append((name, label, abs, pct))

    return table
