import os, plotly.graph_objects as go, sys
from bsb.core import from_hdf5
import numpy as np, h5py
from scipy import stats
import scipy.ndimage, scipy.interpolate, scipy.signal
import pickle, json
import collections
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "plots"))
from _paths import *
from glob import glob
import selection

colorbar_grc = ['rgb(158,188,218)', 'rgb(140,150,198)', 'rgb(140,107,177)', 'rgb(136,65,157)', 'rgb(129,15,124)', 'rgb(77,0,75)']
colorbar_pc = "thermal"
frozen = False

def crop(data, min):
    c = data[:, 1]
    return c[(c > min)]

inv = lambda x: [1000 / y for y in x]
avg = lambda x: sum(x) / len(x) if len(x) else 0

def sec_bool_alg(op, val):
    return f"(x {op} {val})"

def join(op, algs):
    return "(" + f" {op} ".join(algs) + ")"

def parse_bools(struct):
    return join(struct[0], (s if isinstance(s, str) else (sec_bool_alg(*s) if isinstance(s, tuple) else parse_bools(s)) for s in struct[1:]))

def bools_to_lambda(algs):
    if not algs:
        return lambda x: x
    return eval(f"lambda x: {parse_bools(algs)}")

def skip_spike_regions(data, threshold=-20, width=100):
    spikes = scipy.signal.find_peaks(data, threshold=threshold)[0]
    if not len(spikes):
        return data
    excluded = bools_to_lambda(["|", *(["&", (">", s - width), ("<", s + width)] for s in spikes)])
    return data[~excluded(np.arange(len(data)))]

file_time_start = 6000
file_time_stop = 6520

def analyze_calcium(ids, group, start, stop, soma=True, inv=False):
    # IMPORTANT: This assumes that you've pre-cropped your datasets to the ROI
    # and have recorded 1 soma and 1 dendrite per granule cell!!!
    #lin_time = np.linspace(5500, 6500, len(next(iter(group.values()))["concentration/1"]))
    lin_time = np.linspace(file_time_start, file_time_stop, len(next(iter(group.values()))["concentration/0"]))
    mask = (start < lin_time) & (lin_time < stop)
    analysed = {id: 0 for id in ids}
    for key in group:
        f = group[key]
        id = int(key)
        if id not in ids:
            continue
        keys = list(f["concentration/"].keys())
        if soma:
            dataset = f["concentration/0"]
        else:
            # We recorded only the soma and a random dendrite, if we remove
            # the soma ("0"), `keys[0]` is the key of the dendrite.
            keys.remove("0")
            dataset = f[f"concentration/{keys[0]}"]
        data = dataset[mask]
        go.Figure(
            [
                go.Scatter(x=lin_time, y=data)
            ]
        )
        analysed[id] = np.mean(skip_spike_regions(data))
    return analysed


def generate(path=None, net_path=None):
    if path is None:
        path = results_path("grc_ltp_ltd", "single_recon", "*.hdf5")
    if net_path is None:
        net_path = network_path(selection.network)
    paths = glob(path)
    network = from_hdf5(net_path)
    #paths = ["/home/claudia/deschepper-etal-2020/results/CaRecording/sensoryBurst_CaRecordingOnGrCdend.hdf5"]
    for id, path in enumerate(paths):
        with h5py.File(path, "r") as f:
            gen_map(id, f, network)

def gen_map(net_id, f, scaffold):
    stim_start, stim_end = 6000, 6020
    after_effects = 500
    print("Loading network", net_id, f.filename)
    ps_grc = scaffold.get_placement_set("granule_cell")

    surfaces = dict(
        calcium=dict(
            data=lambda f: analyze_calcium(ps_grc.identifiers, f["recorders/ions/ca/"], stim_start, stim_end + after_effects, soma=False),
            smoothen=False,
        )
    )

    for sname, sdict in surfaces.items():
        print(f"Creating {sname} surface", " " * 30, end="\r")
        sdict["_data"] = sdict["data"](f)

    for surface in surfaces.values():
        # Delete the local unpicklable data lambda
        del surface["data"]
        if "reduce" in surface:
            del surface["reduce"]
        if "data_subpop" in surface:
            del surface["data_subpop"]
        if "bin_reduce" in surface:
            del surface["bin_reduce"]
    with open(f"pkl_ca/calcium_data/calcium_{net_id}.pickle", "wb") as f:
        pickle.dump(surfaces, f)


generate()
