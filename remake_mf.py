import os, sys
import h5py
from bsb.core import Scaffold
from bsb.config import JSONConfig
from bsb.output import HDF5Formatter
from bsb.core import from_hdf5
from bsb.reporting import set_verbosity

set_verbosity(3)

for i in range(5,6):
    print('Import HDF5 file...')
    filename_h5 = f"networks/batch_1/network_{i}.hdf5"


    print('Import configuration from .Json ...')

    scaffold = from_hdf5(filename_h5)
    tag = "mossy_to_glomerulus"

    print(f'Generating {tag} connections...')
    scaffold.cells_by_type["glomerulus"] = scaffold.get_cells_by_type("glomerulus")
    scaffold.configuration.connection_types[tag].connect()

    print('Updating HDF5 file...')
    with h5py.File(filename_h5, "a") as f:
        cells_group = f["cells"]
        # Remove the other conn types from the scaffold so no other write operations are attempted
        scaffold.cell_connections_by_tag = {tag: scaffold.cell_connections_by_tag[tag]}
        # Remove any previous data if it exists
        if f"connections/{tag}" in cells_group:
            del cells_group[f"connections/{tag}"]
        if f"connection_compartments/{tag}" in cells_group:
            del cells_group[f"connection_compartments/{tag}"]
        if f"connection_morphologies/{tag}" in cells_group:
            del cells_group[f"connection_morphologies/{tag}"]
        try:
            print("Storing", len(scaffold.cell_connections_by_tag[tag]), "connections")
            print("Data shape:", scaffold.cell_connections_by_tag[tag].shape)
        except Exception as e:
            print("Data format error", str(e))
        # Perform the single-type write operation
        scaffold.output_formatter.store_cell_connections(cells_group)

        print('Update complete.')
