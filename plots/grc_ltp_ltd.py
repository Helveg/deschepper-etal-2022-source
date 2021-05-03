import os, plotly.graph_objects as go, sys
from bsb.core import from_hdf5
import numpy as np, h5py
from scipy import stats
import scipy.ndimage, scipy.interpolate, scipy.signal
import pickle, selection
import collections
from ._paths import *
from glob import glob

colorbar_grc = ['rgb(158,188,218)', 'rgb(140,150,198)', 'rgb(140,107,177)', 'rgb(136,65,157)', 'rgb(129,15,124)', 'rgb(77,0,75)']
colorbar_pc = "thermal"
frozen = False

def crop(data, min):
    c = data[:, 1]
    return c[(c > min)]

inv = lambda x: [1000 / y for y in x]
avg = lambda x: sum(x) / len(x)

def sec_bool_alg(op, val):
    return f"(x {op} val)"

def join(op, algs):
    return "(" + f" {op} ".join(algs) + ")"

def parse_bools(struct):
    return join(struct[0], (s if isinstance(s, str) else (sec_bool_alg(*s) if isinstance(s, tuple) else parse_bools(s)) for s in struct[1:]))

def bools_to_lambda(algs):
    return eval(f"lambda x: {parse_bools()}")

def skip_spike_regions(data, threshold=-20, width=100):
    spikes = scipy.signal.find_peaks(data, threshold=threshold)
    excluded = parse_bools(["|", *(["&", (">", s - width), ("<", s + width)] for s in spikes)])
    return data[not excluded(np.arange(len(data)))]

def analyze_all(ids, group, start):
    analysed = {id: 0 for id in ids}
    for key in group:
        f = group[key]
        id = int(f[0, 0])
        if id not in ids:
            continue
        data = crop(f[()], min=start)
        if len(data) == 0:
            continue
        analysed[id] = np.trapz(skip_spike_regions(data), dx=0.025)
    return analysed

def plot(path=None, network=None):
    if path is None:
        path = results_path("grc_ltp_ltd")
    if network is None:
        network = network_path(selection.network)
    stim_start, stim_end = 6000
    print("Loading network", " " * 30, end="\r")
    scaffold = from_hdf5(network)
    ps_grc = scaffold.get_placement_set("granule_cell")
    pc_pos = ps_grc.positions
    points = ps_grc.positions[:, [0, 2]]
    grid_offset = np.array([0.0, 0.0])  # x z
    grid_spacing = np.array([5., 5.])  #um
    gpoints = np.round((points - grid_offset) / grid_spacing)

    run_avg = lambda datas: {k: sum(d.get(k) for d in datas) / len(datas) for k in datas[0].keys()}
    surfaces = dict(
        control=dict(
            files=glob(os.path.join(path, "*.hdf5")),
            data=lambda f: analyze_all(ps_grc.identifiers, f["recorders/ions/ca/concentration"], stim_start),
            reduce=run_avg,
        )
    )

    if not frozen:
        for sname, sdict in surfaces.items():
            sdict["grid"] = grid = {}
            print(f"Creating {sname} surface", " " * 30, end="\r")
            if "files" in sdict:
                datas = []
                for f in sdict["files"]:
                    with h5py.File(f, "r") as f:
                        datas.append(sdict["data"](f))
                data = sdict["reduce"](datas)
            elif "file" in sdict:
                with h5py.File(sdict["file"], "r") as f:
                    data = sdict["data"](f)
            elif "parent" in sdict:
                data = sdict["data"](surfaces[sdict["parent"]]["_data"])
            else:
                raise Exception(f"Missing `file` or `parent` in {sname} definition.")
            sdict["_data"] = data

            for i, id in enumerate(ps_grc.identifiers):
                coords = tuple(gpoints[i,:])
                if coords not in grid:
                    grid[coords] = []
                grid[coords].append(data[id])

            isosurface_values = [avg(v) for v in grid.values()]
            # "sort" the isosurface {coord: value} map into an ordered square surface matrix
            surface_xcoords = [int(k[0]) for k in grid.keys()]
            surface_ycoords = [int(k[1]) for k in grid.keys()]
            surface = np.zeros((int(max(surface_xcoords) + 1), int(max(surface_ycoords) + 1)))
            for i, (x, y) in enumerate(zip(surface_xcoords, surface_ycoords)):
                surface[x, y] = isosurface_values[i]

            sigma = [0.8,0.8]
            sdict["surface"] = scipy.ndimage.filters.gaussian_filter(surface, sigma)

        for surface in surfaces.values():
            # Delete the local unpicklable data lambda
            del surface["data"]
            if "reduce" in surface:
                del surface["reduce"]
        with open("surfaces.pickle", "wb") as f:
            pickle.dump(surfaces, f)
    else:
        with open("surfaces.pickle", "rb") as f:
            surfaces = pickle.load(f)

    control = surfaces["control"]["surface"]
    plots = {
        "plast_balance": control,
    }
    figs = {}
    for name, z in plots.items():
        cmax = 1.5
        if isinstance(z, tuple):
            cmax = z[1]
            z = z[0]
        fig = go.Figure(go.Surface(z=z,cmin=0, cmax=cmax))
        fig.update_layout(
            title_text=name,
            scene=dict(
                zaxis_range=[-0.3, 1.5],
                xaxis_title="Y",
                yaxis_title="X",
                camera=dict(up=dict(x=0,y=0,z=1),center=dict(x=0,y=0,z=0),eye=dict(x=-0.903754304010216,y=1.34612207303165,z=1.4348113194702579)),
                aspectratio=dict(x=2/3, y=1, z=0.3)
            )
        )
        figs[name] = fig
    return figs
