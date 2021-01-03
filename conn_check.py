from bsb.core import from_hdf5
import numpy as np
import plots.selection as selection

filename = 'networks/300x_200z.hdf5'
network = from_hdf5(filename)
config = network.configuration

MFs = selection.stimulated_mf_poiss
# Create a metadata dictionary that specifies which cell types and connection types to
# study and which individual cells to investigate
#
# An an entry should specify the name as key and a tuple (labels, ids, dict_of_conn_sets)
selector = {
    "purkinje_cell": (
        selection.purkinje_cell_labels,
        selection.purkinje_cell_ids,
        {"PF": "parallel_fiber_to_purkinje", "AA": "ascending_axon_to_purkinje"},
    ),
    "golgi_cell": (
        selection.golgi_cell_labels,
        selection.golgi_cell_ids,
        {"PF": "parallel_fiber_to_golgi", "AA": "ascending_axon_to_golgi"},
    ),
    "basket_cell": (
        selection.basket_cell_labels,
        selection.basket_cell_ids,
        {"PF": "parallel_fiber_to_basket"},
    ),
    "stellate_cell": (
        selection.stellate_cell_labels,
        selection.stellate_cell_ids,
        {"PF": "parallel_fiber_to_stellate"},
    ),
}

mf_glom = network.get_connectivity_set("mossy_to_glomerulus").get_dataset()
glom_grc = network.get_connectivity_set("glomerulus_to_granule").get_dataset()
# Filter all gloms that are connected to one of the selected MFs
active_glom = mf_glom[np.isin(mf_glom[:, 0], MFs), 1]
# Filter each GrC ID each time it is connected to an active glom
active_dendrites = glom_grc[np.isin(glom_grc[:, 0], active_glom), 1]
# Count how many times each GrC ID was connected to a glom
active_grc_ids, active_dend_count = np.unique(active_dendrites, return_counts=True)
# Filter all the GrC IDs which were connected 2 or more times to active gloms
very_active_grc = active_grc_ids[active_dend_count >= 2]

for (name, data) in selector.items():
    # Unpack the data from the 'selector' metadata dictionary
    labels, ids, conn_sets = data
    # Go over each specified connection set
    for conn_label, conn_tag in conn_sets.items():
        conn_set = network.get_connectivity_set(conn_tag).get_dataset()
        # For each cell (low, med, high) count their connections to the active grc
        for label, id in zip(labels, ids):
            # Get all of the connections to our cell
            roi_mask = conn_set[:, 1] == id
            # Get the grc ids of those connections
            roi = conn_set[roi_mask, 0]
            # Count our connections
            total = np.sum(roi_mask)
            # Count how many of the connections come from very active grc
            activated = np.sum(np.isin(roi, very_active_grc))
            # Report the number
            print(label, name, "has", f"{round(activated / total * 100)}% active {conn_label} synapses")
