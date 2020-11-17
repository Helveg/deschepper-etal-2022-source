import numpy as np

# "networks/300x_200z.hdf5"

stimulated_mf_poiss = np.array([213, 214, 222, 223])
golgi_cell_ids = np.array([14, 24, 35])
golgi_cell_labels = ["Medium activity", "High activity", "Low activity"]
golgi_cells = dict(zip(golgi_cell_labels, golgi_cell_ids))
purkinje_cell_ids = np.array([91, 126, 145])
purkinje_cell_labels = ["Medium activity", "High activity", "Low activity"]
purkinje_cells = dict(zip(purkinje_cell_labels, purkinje_cell_ids))
basket_cell_ids = np.array([329, 383])
basket_cell_labels = ["High activity", "Low activity"]
basket_cells = dict(zip(basket_cell_labels, basket_cell_ids))
stellate_cell_ids = np.array([561, 618])
stellate_cell_labels = ["High activity", "Low activity"]
stellate_cells = dict(zip(stellate_cell_labels, stellate_cell_ids))
granule_cells = np.array([3070, 31681, 3074, 9163, 3083, 11399, 3764, 15288, 5987, 17372])
granule_cell_order = dict(map(lambda t: (t[1], t[0]), enumerate(granule_cells)))

sync = {
    "purkinje_cell": [83, 78, 126],
    "stellate_cell": [695, 715],
    "basket_cell": [306, 310],
    "golgi_cell": [1, 17, 53],
    "granule_cell": [3070, 3069, 3068, 3075, 3083],
}

btn_config = {
    'toImageButtonOptions': {
        'format': 'pdf',
        'height': 1920,
        'width': 1080,
        'scale': 1
    }
}
