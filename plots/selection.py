import numpy as np

def plot():
    pass

# "networks/300x_200z.hdf5"

network = "balanced.hdf5"

stimulated_mf_poiss = np.array([213, 214, 222, 223])
stimulated_mf_sync = np.array([213, 214, 222, 223, 229, 221, 206, 230, 247, 239, 231, 238, 240])
stimulated_mf_lat = np.array([275, 261, 260, 263, 273, 267, 279, 265, 268, 280, 270, 269, 272, 274, 282, 264, 271, 266, 283, 276, 278, 281, 277, 284])
golgi_cell_ids = np.array([23, 24, 0])
golgi_cell_labels = ["Medium activity", "High activity", "Low activity"]
golgi_cells = dict(zip(golgi_cell_labels, golgi_cell_ids))
purkinje_cell_ids = np.array([91, 126, 145])
purkinje_cell_labels = ["Medium activity", "High activity", "Low activity"]
purkinje_cells = dict(zip(purkinje_cell_labels, purkinje_cell_ids))
basket_cell_ids = np.array([342, 366]) #[292, 320])
basket_cell_labels = ["High activity", "Low activity"]
basket_cells = dict(zip(basket_cell_labels, basket_cell_ids))
stellate_cell_ids = np.array([513, 629])
stellate_cell_labels = ["High activity", "Low activity"]
stellate_cells = dict(zip(stellate_cell_labels, stellate_cell_ids))
granule_cells = np.array([3070, 31681, 3074, 9163, 3083, 11399, 3764, 15288, 5987, 17372])
granule_cell_order = dict(map(lambda t: (t[1], t[0]), enumerate(granule_cells)))
granule_cells_balanced = np.array([5987, 6002, 9272, 17285, 17372, 24861, 3764, 3851, 4075,3083, 3114, 3192, 3074, 3076, 3109])

mf_batch_1 = [
    [192, 208, 215, 231],
    [212, 246, 171, 185],
    [231, 254, 272, 276],
    [195, 271, 213, 182],
    [268, 198, 184, 252],
    [195, 194, 170, 199],
    [192, 187, 175, 273],
    [185, 279, 206, 172, 173],
    [240, 241, 274, 186],
    [221, 198, 188, 249]
]

vi_stellate = 570
vi_purkinje = 136

sync = {
    "granule_cell": [3070, 3069, 3068, 3075, 3083],
    "golgi_cell": [1, 17, 53],
    "purkinje_cell": [83, 78, 126],
    "stellate_cell": [695, 715],
    "basket_cell": [306, 310],
}

onbeam = {
    "stellate_cell": [639],
    "basket_cell": [314],
}

btn_config = {
    'toImageButtonOptions': {
        'format': 'pdf',
        'height': 1920,
        'width': 1080,
        'scale': 1
    }
}
