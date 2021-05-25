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

colorbar_grc = ['rgb(158,188,218)', 'rgb(140,150,198)', 'rgb(140,107,177)', 'rgb(136,65,157)', 'rgb(129,15,124)', 'rgb(77,0,75)']
colorbar_pc = "thermal"
frozen = False

def make_psi():
    # Use traced data from "A Nonlinear Cable Framework for Bidirectional Synaptic Plasticity"
    # Appearantly the formulas were in Supplementary, lol.
    c = np.array([[0, 428],[179.00,428.00],[192.41,428.44],[209.00,430.00],[226.50,434.50],[241.00,444.50],[253.00,456.00],[268.00,473.50],[282.00,491.50],[295.00,504.00],[304.50,510.00],[310.50,511.50],[315.50,511.50],[324.00,510.00],[330.50,505.00],[339.50,496.00],[346.50,484.50],[354.00,466.50],[360.00,447.50],[367.00,425.50],[384.00,344.50],[402.00,250.00],[417.50,188.00],[431.00,151.50],[444.00,133.50],[459.50,119.50],[471.50,114.50],[499.50,109.50],[529.00,108.00],[595.00,108.50]])
    # norm and milli to micro
    c /= 600
    c[:, 1] = (1 - c[:, 1]) * 1.4 - 0.4014
    c[:, 0] /= 1000
    f_intrp = scipy.interpolate.interp1d(c[:, 0], c[:, 1], kind="linear", bounds_error=False, fill_value=0.745)
    return f_intrp

psi = make_psi()

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

def analyze_calcium(ids, group, start, stop, soma=True, inv=False):
    # IMPORTANT: This assumes that you've pre-cropped your datasets to the ROI
    # and have recorded 1 soma and 1 dendrite per granule cell!!!
    lin_time = np.linspace(start, stop, len(next(iter(group.values()))["concentration/0"]))
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
        x = np.mean(skip_spike_regions(data))
        analysed[id] = psi(x) if not inv else -psi(x)
    return analysed

def binary_category(category):
    def categorizer(datas):
        hits = {k: 0 for k in datas[0].keys()}
        weight = 1 / len(datas)
        for data in datas:
            for k, v in data.items():
                hits[k] += weight * category(float(v))
        return hits

    return categorizer

def generate(path=None, net_path=None):
    if path is None:
        path = results_path("grc_ltp_ltd", "*.hdf5")
    if net_path is None:
        net_path = network_path("batch_1")
    paths = glob(path)
    for path in paths:
        with h5py.File(path, "r") as f:
            id = int(list(json.loads(f.attrs["configuration_string"])["simulations"].keys())[0].split("_")[-1])
            network = from_hdf5(os.path.join(net_path, f"network_{id}.hdf5"))
            gen_map(id, f, network)

def gen_map(net_id, f, scaffold):
    stim_start, stim_end = 6000, 6020
    after_effects = 500
    print("Loading network", net_id, f.filename)
    ps_grc = scaffold.get_placement_set("granule_cell")
    pc_pos = ps_grc.positions
    points = ps_grc.positions[:, [0, 2]]
    grid_offset = np.array([0.0, 0.0])  # x z
    grid_spacing = np.array([5., 5.])  #um
    gpoints = np.round((points - grid_offset) / grid_spacing)

    run_avg = lambda datas: {k: sum(d.get(k) for d in datas) / len(datas) for k in datas[0].keys()}
    surfaces = dict(
        ltp_cont=dict(
            data=lambda f: analyze_calcium(ps_grc.identifiers, f["recorders/ions/ca/"], stim_start, stim_end + after_effects, soma=False),
            data_subpop=lambda v: v > 0,
            bin_reduce=sum,
            smoothen=False,
        ),
        ltd_cont=dict(
            data=lambda f: analyze_calcium(ps_grc.identifiers, f["recorders/ions/ca/"], stim_start, stim_end + after_effects, soma=False, inv=True),
            data_subpop=lambda v: v > 0.00008,
            bin_reduce=sum,
            smoothen=False,
        )
    )

    for sname, sdict in surfaces.items():
        sdict.setdefault("smoothen", True)
        sdict["grid"] = grid = {}
        print(f"Creating {sname} surface", " " * 30, end="\r")
        sdict["_data"] = data = sdict["data"](f)
        subpopulate = "data_subpop" not in sdict
        ids = ps_grc.identifiers if not subpopulate else []
        for i, id in enumerate(ps_grc.identifiers):
            coords = tuple(gpoints[i,:])
            if coords not in grid:
                grid[coords] = []
            d = data[id]
            if subpopulate or sdict["data_subpop"](d):
                grid[coords].append(data[id])
                if subpopulate:
                    ids.append(i)

        bin_reduce = sdict.get("bin_reduce", avg)
        isosurface_values = [bin_reduce(v) for v in grid.values()]
        # "sort" the isosurface {coord: value} map into an ordered square surface matrix
        surface_xcoords = [int(k[0]) for k in grid.keys()]
        surface_ycoords = [int(k[1]) for k in grid.keys()]
        surface = np.zeros((int(max(surface_xcoords) + 1), int(max(surface_ycoords) + 1)))
        for i, (x, y) in enumerate(zip(surface_xcoords, surface_ycoords)):
            surface[x, y] = isosurface_values[i]

        if sdict["smoothen"]:
            if isinstance(sdict["smoothen"], list):
                sigma = sdict["smoothen"]
            sigma = [0.8,0.8]
            surface = scipy.ndimage.filters.gaussian_filter(surface, sigma)

        sdict["surface"] = surface
        if subpopulate:
            sdict["population"] = ids

    for surface in surfaces.values():
        # Delete the local unpicklable data lambda
        del surface["data"]
        if "reduce" in surface:
            del surface["reduce"]
        if "data_subpop" in surface:
            del surface["data_subpop"]
        if "bin_reduce" in surface:
            del surface["bin_reduce"]
    with open(f"pkl_ca/sum_maps/calcium_rawsum_{net_id}.pickle", "wb") as f:
        pickle.dump(surfaces, f)


generate()
