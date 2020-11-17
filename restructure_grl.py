import shutil, h5py
from bsb.core import from_hdf5
from bsb.reporting import set_verbosity
from time import time

set_verbosity(3)

for i in range(5):
    path = f"networks/center_surround_{i+5}.hdf5"
    shutil.copy("networks/300x_200z.hdf5", path)
    network = from_hdf5(path)
    network.get_cells_by_type("granule_cell")
    network.get_cells_by_type("golgi_cell")
    network.get_cells_by_type("glomerulus")
    network.get_entities_by_type("mossy_fibers")
    tags = ["mossy_to_glomerulus", "glomerulus_to_granule", "glomerulus_to_golgi", "golgi_to_glomerulus", "golgi_to_granule"]
    for tag in tags:
        t = time()
        print("Connecting", tag)
        print("Old len", len(network.get_connectivity_set(tag)))
        network.configuration.connection_types[tag].connect()
        print("New len", len(network.cell_connections_by_tag[tag]))
        print("Took", time() - t)
    store = network.cell_connections_by_tag
    with h5py.File(path, "a") as f:
        cells_group = f["/cells"]
        for tag in tags:
            if "connections/" + tag in cells_group:
                del cells_group["connections/" + tag]
            if "connection_compartments/" + tag in cells_group:
                del cells_group["connection_compartments/" + tag]
            if "connection_morphologies/" + tag in cells_group:
                del cells_group["connection_morphologies/" + tag]
            network.cell_connections_by_tag = {tag: store[tag]}
            network.output_formatter.store_cell_connections(cells_group)
            print("Saved", tag)
