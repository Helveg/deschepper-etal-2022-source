import os, plotly.graph_objects as go, sys
from bsb.core import from_hdf5
import numpy as np, h5py
from scipy import stats
import scipy.ndimage, scipy.interpolate
import pickle, selection
import collections

colorbar_grc = ['rgb(158,188,218)', 'rgb(140,150,198)', 'rgb(140,107,177)', 'rgb(136,65,157)', 'rgb(129,15,124)', 'rgb(77,0,75)']
colorbar_pc = "thermal"


def network_path(*args):
    return os.path.join(
        os.path.dirname(__file__), "..", "networks", *args
    )

frozen = False

def results_path(*args):
    return os.path.join(
        os.path.dirname(__file__), "..", "results", *args
    )

def crop(data, min, max, indices=False):
    c = data[:, 1]
    if indices:
        return np.where((c > min) & (c < max))[0]
    return c[(c > min) & (c < max)]

def pairs(arr):
    for i in range(len(arr) - 1):
        yield arr[i], arr[i + 1]

def get_isis(spikes, selected):
    if len(selected) == 1:
        if selected[0] - 1 == -1:
            return []
        return [spikes[selected[0]] - spikes[selected[0] - 1]]
    return [spikes[second] - spikes[first] for first, second in pairs(selected)]

def get_parallel(subset, set):
    return np.where(np.isin(set, subset))[0]

inv = lambda x: [1000 / y for y in x]
avg = lambda x: sum(x) / len(x)

def normaliz(lista, maxValue):
    lista = {k: v / maxValue for k, v in lista.items()}
    return lista

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
        path_control = "center_surround/cs_control.hdf5"
    if path_gaba is None:
        path_gaba = "center_surround/cs_gabazine.hdf5"
    if network is None:
        network = selection.network
    base_start, base_end = 600, 800
    stim_start, stim_end = 1000, 1050
    print("Loading network", " " * 30, end="\r")
    scaffold = from_hdf5(network_path(network))
    ps_grc = scaffold.get_placement_set("granule_cell")
    pc_pos = ps_grc.positions
    points = ps_grc.positions[:, [0, 2]]
    grid_offset = np.array([0.0, 0.0])  # x z
    grid_spacing = np.array([5., 5.])  #um
    gpoints = np.round((points - grid_offset) / grid_spacing)

    surfaces = dict(
        control=dict(
            file=results_path(path_control),
            data=lambda f: get_activity(ps_grc.identifiers, f["recorders/soma_spikes/"], stim_start, stim_end),
        ),
        gabazine=dict(
            file=results_path(path_gaba),
            data=lambda f: get_activity(ps_grc.identifiers, f["recorders/soma_spikes/"], stim_start, stim_end),
        ),
        n2a=dict(
            file=results_path(path_control),
            data=lambda f: get_activity(ps_grc.identifiers, f["recorders/soma_spikes/"], 1000, 1003),
        ),
        n2b_control=dict(
            file=results_path(path_control),
            data=lambda f: get_activity(ps_grc.identifiers, f["recorders/soma_spikes/"], 1003, 1010),
        ),
        n2b_gabazine=dict(
            file=results_path(path_gaba),
            data=lambda f: get_activity(ps_grc.identifiers, f["recorders/soma_spikes/"], 1003, 1010),
        ),
    )

    if not frozen:
        for sname, sdict in surfaces.items():
            sdict["grid"] = grid = {}
            print(f"Creating {sname} surface", " " * 30, end="\r")
            if "file" in sdict:
                f = h5py.File(sdict["file"], "r")
                data = sdict["data"](f)
                f.close()
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
        with open("surfaces.pickle", "wb") as f:
            pickle.dump(surfaces, f)
    else:
        with open("surfaces.pickle", "rb") as f:
            surfaces = pickle.load(f)

    control = surfaces["control"]["surface"]
    gabazine = surfaces["gabazine"]["surface"]
    n2a = surfaces["n2a"]["surface"]
    n2b_control = surfaces["n2b_control"]["surface"]
    n2b_gabazine = surfaces["n2b_gabazine"]["surface"]
    E = n2a / np.max(n2a)
    I = (n2b_gabazine - n2b_control) / np.max(n2b_gabazine - n2b_control)
    B = (E - I) / (E + 1)
    plots = {
        "control": control,
        "gabazine": gabazine,
        "n2a": n2a,
        "n2b_control": n2b_control,
        "n2b_gabazine": n2b_gabazine,
        "excitation": E,
        "inhibition": I,
    }
    figs = {}
    for name, z in plots.items():
        fig = go.Figure(go.Surface(z=z))
        fig.update_layout(
            title_text=name,
            scene=dict(
                zaxis_range=[0, 1],
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
