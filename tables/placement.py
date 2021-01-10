from bsb.core import from_hdf5

network = from_hdf5("networks/300x_200z.hdf5")
print("Cell type\tTarget\tObtained")
for cell_type in network.get_cell_types(entities=False):
    ps = network.get_placement_set(cell_type)
    vol = cell_type.placement.layer_instance.volume
    print(cell_type.name, cell_type.placement.get_placement_count() / vol, len(ps) / vol)
