import neuro3d as n3d
import neuro3d.animation
from bsb.output import MorphologyRepository as MR
import h5py, numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from time import time as _time
import itertools

mr = MR("morphologies.hdf5")
m = mr.get_morphology("GolgiCell")

def port_branch(branch):
    point_matrix = branch.points
    coords = point_matrix[:, :-1]
    radii = point_matrix[:, -1]
    n3d_branch = n3d.create_branch(coords, radii, ref=branch._neuron_sid)
    n3d_branch._children = [port_branch(b) for b in branch.children]
    return n3d_branch

def make_golgi_cell(location=None, rotation=None):
    cell_roots = [port_branch(root) for root in m.roots]
    cell = n3d.Cell(cell_roots, location, rotation)
    cell.register()
    return cell

def synaptic_current_getter(type, exc=True):
    cache = None
    mask, mask_l = None, 0
    def getter(group, id, time):
        nonlocal cache, mask, mask_l
        if mask is None:
            mask = time.as_mask()
            mask_l = sum(mask)
        if cache is None:
            cache = {}
            for ds in group.values():
                if not ds.attrs["type"].startswith(type): continue
                gid = int(ds.attrs["section"])
                if gid not in cache:
                    cache[gid] = np.zeros(mask_l, dtype=float)
                cache[gid] += ds[mask]
        cache_hit = cache.get(id, None)
        return -cache_hit if exc and cache_hit is not None else cache_hit
    return getter

def _default_getter(group, id, time):
    if str(id) in group:
        return group[str(id)][time.as_mask()]
    else:
        return None

def animate_branches(animator, cell, group, window, time, getter=None, skip=True):
    if getter is None:
        getter = _default_getter
    to_skip = set()
    for branch in cell.branches:
        id = int(branch._ref)
        data = getter(group, id, time)
        if data is None:
            to_skip.add(branch)
            continue
        cmin, cmax = animator._encoder._pipe[0]._cmin, animator._encoder._pipe[0]._cmax
        dmin, dmax = min(data), max(data)
        assert dmin >= cmin and dmax <= cmax, f"Calibration error for {id}: [{cmin}, {dmin}] [{cmax}, {dmax}]"
        animator.animate(branch._material, window, data, time)
    if skip:
        for branch in to_skip:
            # Give the rest of the morphology a disabled color
            branch._material.node_tree.nodes["Emission"].inputs[0].default_value = (0.00643645, 0, 0.0134106, 1)

def calibrate_stdev_across(encoder, time, group):
    mean = np.mean([v[time.as_mask()] for v in group.values()])
    stdev = np.std([v[time.as_mask()] for v in group.values()])
    encoder.calibrate(mean, stdev)

def calibrate_minmax_across(encoder, time, group):
    min = np.min([v[time.as_mask()] for v in group.values()])
    max = np.max([v[time.as_mask()] for v in group.values()])
    encoder.calibrate(min, max)

def calibrate_syn_minmax_across(encoder, time, group, type, flip=False):
    mask = time.as_mask()
    empty = np.zeros(np.nonzero(time.as_mask())[0].shape)
    ids = set(int(d.attrs["section"]) for d in group.values() if d.attrs["type"].startswith(type))
    carry = dict(zip(ids, (empty.copy() for _ in itertools.count())))
    for d in group.values():
        if not d.attrs["type"].startswith(type):
            continue
        carry[int(d.attrs["section"])] += d[mask]
    min = np.min(list(carry.values()))
    max = np.max(list(carry.values()))
    if flip:
        min, max = -max, -min
    encoder.calibrate(min, max)

def make_calibratable_animator(scale=10, epsilon=0.003):
    encoder = n3d.encoders.norm()
    pipe = n3d.encoders.pipe(
        encoder,
        n3d.encoders.mult(scale),
        n3d.encoders.rdp(epsilon=epsilon)
    )
    animator = n3d.animation.create_animator(pipe)
    return encoder, animator

# Global animation
frame_window = n3d.animation.create_window(0, 4000, 6, 6.1)
vm_encoder, vm_animator = make_calibratable_animator()
calcium_encoder, calcium_animator = make_calibratable_animator()
ampa_encoder, ampa_animator = make_calibratable_animator()
nmda_encoder, nmda_animator = make_calibratable_animator()
gaba_encoder, gaba_animator = make_calibratable_animator()

# Cells
gid = 24
vm_golgi = make_golgi_cell([-150, 150, 0])
calcium_golgi = make_golgi_cell([0, 150, 0])
ampa_golgi = make_golgi_cell([150, 150, 0])
nmda_golgi = make_golgi_cell([300, 150, 0])
gaba_golgi = make_golgi_cell([450, 150, 0])

with h5py.File("results_golgi_recordings.hdf5", "r") as h5:
    time = n3d.time(h5["time"][()] / 1000).window(6, 6.1)
    g = h5["recorders/dendrite_voltage"]
    vm_group = {str(id): g[f"{gid}.dend[{id}]"] for id in (b._ref for b in vm_golgi.branches) if f"{gid}.dend[{id}]" in g}
    calibrate_minmax_across(vm_encoder, time, vm_group)
    print("Calibrated VM", vm_encoder._cmin, vm_encoder._cmax)
    animate_branches(vm_animator, vm_golgi, vm_group, frame_window, time)
    group = h5[f"recorders/ions/ca/{gid}/concentration/"]
    calibrate_minmax_across(calcium_encoder, time, group)
    print("Calibrated calcium", calcium_encoder._cmin, calcium_encoder._cmax)
    animate_branches(calcium_animator, calcium_golgi, group, frame_window, time)
    group = h5[f"recorders/synapses/{gid}/current/"]
    print("Calibrating AMPA")
    t = _time()
    calibrate_syn_minmax_across(ampa_encoder, time, group, "AMPA", flip=True)
    print("Calibrated", ampa_encoder._cmin, ampa_encoder._cmax)
    t = _time()
    print("Animating AMPA")
    animate_branches(ampa_animator, ampa_golgi, group, frame_window, time, getter=synaptic_current_getter("AMPA"))
    print("Animated", _time() - t)
    print("Calibrating NMDA")
    t = _time()
    calibrate_syn_minmax_across(nmda_encoder, time, group, "NMDA", flip=True)
    print("Calibrated", nmda_encoder._cmin, nmda_encoder._cmax)
    t = _time()
    print("Animating NMDA")
    animate_branches(nmda_animator, nmda_golgi, group, frame_window, time, getter=synaptic_current_getter("NMDA"))
    print("Animated", _time() - t)
    print("Calibrating GABA")
    t = _time()
    calibrate_syn_minmax_across(gaba_encoder, time, group, "GABA")
    print("Calibrated", gaba_encoder._cmin, gaba_encoder._cmax)
    t = _time()
    print("Animating GABA")
    animate_branches(gaba_animator, gaba_golgi, group, frame_window, time, getter=synaptic_current_getter("GABA", exc=False))
    print("Animated", _time() - t)
