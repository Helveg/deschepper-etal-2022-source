import blenderneuron as bn
import blenderneuron.animation
from bsb.output import MorphologyRepository as MR
import h5py, numpy as np

mr = MR("morphologies.hdf5")
m = mr.get_morphology("PurkinjeCell")

def port_branch(branch):
    point_matrix = branch.points
    coords = point_matrix[:, :-1]
    radii = point_matrix[:, -1]
    bn_branch = bn.create_branch(coords, radii, ref=branch._neuron_sid)
    bn_branch._children = [port_branch(b) for b in branch.children]
    return bn_branch

cell_roots = [port_branch(root) for root in m.roots]
cell = bn.Cell(cell_roots)
cell.register()

encoder = bn.animation.encoders.RangeEncoder(0, 10)
animator = bn.animation.create_animator(encoder)
time_window = bn.animation.create_window(1, 1.1, dt=0.0001, fps=10000)

def get_dendritic_current(group, did, current):
    res = None
    for ds in group.values():
        if ds.attrs["section"] == did and ds.attrs["type"] == "AMPA":
            if res is None:
                print("Found first set", ds[()])
                res = ds[()]
            else:
                print("Found additional set", ds[()])
                res += ds[()]
    return res

with h5py.File("results_dendrites.hdf5", "r") as h5:
    synapses = h5["recorders/synapses/126/current"]
    [(min(v[10000:11000]), max(v[10000:11000])) for v in synapses.values()]
    all_min = min(min(-v[10000:11000]) for v in synapses.values() if v.attrs.get("type", None) == "AMPA")
    all_max = max(max(-v[10000:11000]) for v in synapses.values() if v.attrs.get("type", None) == "AMPA")
    animator.calibrate(all_min, all_max)
    for branch in cell.curve_container._branches:
        print("Checking for NEURON branch", branch._ref)
        current = get_dendritic_current(synapses, branch._ref, "AMPA")
        if current is not None:
            print("Found synaptic currents for", branch._ref)
            print(current)
            animator.animate(branch._material, time_window, -current[10000:11000])
