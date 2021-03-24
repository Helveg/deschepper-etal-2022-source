import os, plotly.graph_objects as go, sys
from bsb.core import from_hdf5
import numpy as np, h5py
from scipy import stats
import scipy.ndimage, scipy.interpolate
import pickle, selection
import collections
from ._paths import *
from glob import glob

colorbar_grc = ['rgb(158,188,218)', 'rgb(140,150,198)', 'rgb(140,107,177)', 'rgb(136,65,157)', 'rgb(129,15,124)', 'rgb(77,0,75)']
colorbar_pc = "thermal"
frozen = False

def crop(data, min, max, indices=False):
    c = data[:, 1]
    if indices:
        return np.where((c > min) & (c < max))[0]
    return c[(c > min) & (c < max)]

inv = lambda x: [1000 / y for y in x]
avg = lambda x: sum(x) / len(x)

def get_activity(ids, group, start, stop):
    num_spikes = {int(id): 0 for id in ids}
    for key in group:
        f = group[key]
        spikes = crop(f[()], min=start, max=stop, indices=True)
        if len(f) == 0:
            continue
        id = int(f[0, 0])
        if id not in ids:
            continue
        num_spikes[id] += len(spikes)
    return num_spikes

def plot(path_control=None, path_gaba=None, network=None):
    if path_control is None:
        path_control = results_path("center_surround", "control")
    if path_gaba is None:
        path_gaba = results_path("center_surround", "gabazine")
    if network is None:
        network = network_path(selection.network)
    base_start, base_end = 600, 800
    stim_start, stim_end = 1000, 1010
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
            files=glob(os.path.join(path_control, "*.hdf5")),
            data=lambda f: get_activity(ps_grc.identifiers, f["recorders/soma_spikes/"], stim_start, stim_end),
            reduce=run_avg,
        ),
        gabazine=dict(
            files=glob(os.path.join(path_gaba, "*.hdf5")),
            data=lambda f: get_activity(ps_grc.identifiers, f["recorders/soma_spikes/"], stim_start, stim_end),
            reduce=run_avg,
        ),
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

    # control = surfaces["control"]["surface"]
    # gabazine = surfaces["gabazine"]["surface"]
    # n2a = surfaces["n2a"]["surface"]
    # n2b_control = surfaces["n2b_control"]["surface"]
    # n2b_gabazine = surfaces["n2b_gabazine"]["surface"]
    # E = n2a / np.max(n2a)
    # I = (n2b_gabazine - n2b_control) / np.max(n2b_gabazine - n2b_control)
    # B = (E - I) / (E + 1)
    # plots = {
    #     "control": control,
    #     "gabazine": gabazine,
    #     "n2a": n2a,
    #     "n2b_control": n2b_control,
    #     "n2b_gabazine": n2b_gabazine,
    #     "excitation": E,
    #     "inhibition": I,
    # }
    control = surfaces["control"]["surface"]
    gabazine = surfaces["gabazine"]["surface"]
    E = control
    I = gabazine - control
    B = (E - I) / (E + 1)
    plots = {
        "control": control,
        "gabazine": gabazine,
        "excitation": E,
        "inhibition": (I, 0.4),
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
                aspectratio=dict(x=2/3, y=1, z=0.3)
            )
        )
        figs[name] = fig
    fig = go.Figure(go.Surface(z=B, colorscale="balance", cmin=-0.5, cmax=0.5))
    fig.update_layout(
        title_text="E/I balance",
        scene=dict(
            zaxis_range=[-1, 1],
            xaxis_title="Y",
            yaxis_title="X",
            zaxis_title="B",
            aspectratio=dict(x=2/3, y=1, z=0.3),
        )
    )
    fig.update_layout(
        scene_yaxis=dict(
            tickmode="array",
            tickvals=[0, 20, 40, 60],
            ticktext=["0", "100", "200", "300"]
        ),
        scene_xaxis=dict(
            tickmode="array",
            tickvals=[0, 10, 20, 30, 40],
            ticktext=["0", "50", "100", "150", "200"]
        )
    )
    figs["balance"] = fig
    return figs
