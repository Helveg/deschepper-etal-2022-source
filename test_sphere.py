from scaffold.core import from_hdf5
from scaffold.config import JSONConfig
from scaffold.output import HDF5Formatter
import h5py, numpy as np

# Reconfigure and load
network_name = "networks/neuron.hdf5"
config = JSONConfig("neuron.json")
HDF5Formatter.reconfigure(network_name, config)
scaffold = from_hdf5(network_name)
# Load the cells we need
scaffold.get_cells_by_type("golgi_cell")
scaffold.get_cells_by_type("glomerulus")
# Execute the new connection
scaffold.cell_connections_by_tag = {"glomerulus_to_golgi": np.zeros((0,2))}
scaffold.configuration.connection_types["glomerulus_to_golgi"].connect()
# Overwrite the HDF5 datasets with the result
with h5py.File(network_name, "a") as f:
    print("Old connections:", len(f["cells/connections"]["glomerulus_to_golgi"]))
    del f["cells/connections/glomerulus_to_golgi"]
    del f["cells/connection_compartments/glomerulus_to_golgi"]
    del f["cells/connection_morphologies/glomerulus_to_golgi"]
    scaffold.output_formatter.store_cell_connections(f["/cells"])
    print("New connections:", len(f["cells/connections"]["glomerulus_to_golgi"]))
