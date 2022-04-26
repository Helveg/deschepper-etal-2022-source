import h5py
import sys
from glob import glob
import itertools

cell_map = {
    "record_mossy_spikes": "mossy_fibers",
    "record_glomerulus_spikes": "glomerulus",
    "record_grc_spikes": "granule_cell",
    "record_basket_spikes": "basket_cell",
    "record_stellate_spikes": "stellate_cell",
    "record_golgi_spikes": "golgi_cell",
    "record_pc_spikes": "purkinje_cell",
}

label_map = {
    "record_mossy_spikes": "MF",
    "record_glomerulus_spikes": "Glom",
    "record_grc_spikes": "Grc",
    "record_basket_spikes": "BC",
    "record_stellate_spikes": "SC",
    "record_golgi_spikes": "GoC",
    "record_pc_spikes": "PC",
}

order = ["record_mossy_spikes", "record_glomerulus_spikes", "record_grc_spikes", "record_golgi_spikes", "record_pc_spikes", "record_basket_spikes", "record_stellate_spikes"]


files = itertools.chain(*map(glob, sys.argv[1:]))
for f in files:
    with h5py.File(f, "a") as f:
        for g in f["recorders/soma_spikes"].values():
            l = g.attrs["label"]
            g.attrs["cell_types"] = [cell_map[l]]
            g.attrs["order"] = order.index(l)
            if "color" in g.attrs:
                del g.attrs["color"]
