import numpy as np

# "networks/300x_200z.hdf5"

stimulated_mf_poiss = np.array([213, 214, 222, 223])
golgi_cell_ids = np.array([14, 24, 35])
golgi_cell_labels = ["Medium activity", "High activity", "Low activity"]
golgi_cells = dict(zip(golgi_cell_labels, golgi_cell_ids))
purkinje_cell_ids = np.array([91, 126, 145])
purkinje_cell_labels = ["Medium activity", "High activity", "Low activity"]
purkinje_cells = dict(zip(purkinje_cell_labels, purkinje_cell_ids))
