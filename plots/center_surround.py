import os, plotly.graph_objects as go, sys
from bsb.core import from_hdf5
import numpy as np, h5py
from scipy import stats
import scipy.ndimage, scipy.interpolate
import pickle

colorbar_grc = ['rgb(158,188,218)', 'rgb(140,150,198)', 'rgb(140,107,177)', 'rgb(136,65,157)', 'rgb(129,15,124)', 'rgb(77,0,75)']
colorbar_pc = "thermal"


network_path = os.path.join(
    os.path.dirname(__file__), "..", "networks", "300x_200z.hdf5"
)

frozen = True

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

def plot():
    base_start, base_end = 400, 600
    stim_start, stim_end = 700, 820
    print("Loading network", " " * 30, end="\r")
    scaffold = from_hdf5(network_path)
    ps_grc = scaffold.get_placement_set("granule_cell")
    pc_pos = ps_grc.positions
    points = ps_grc.positions[:, [0, 2]]
    grid_offset = np.array([0.0, 0.0])  # x z
    grid_spacing = np.array([5., 5.])  #um
    gpoints = np.round((points - grid_offset) / grid_spacing)

    if not frozen:
        grid_control = {}
        grid_gabaz = {}
        grid_bg = {}
        for fi in range(5):
            print(f"Sampling run {fi+1}", " " * 30, end="\r")
            results = h5py.File(results_path(f"center_surround/results_stim_on_MFs_CS{fi}_control.hdf5"), "r")
            results_gabaz = h5py.File(results_path(f"center_surround/results_stim_on_MFs_CS{fi}_GABAZINE.hdf5"), "r")
            results_bg = h5py.File(results_path(f"center_surround/bg/results_stim_on_MFs_onlyBack20Hz_v{fi+1}.hdf5"), "r")
            g_ctrl = results["recorders/soma_spikes/"]
            g_gabaz = results_gabaz["recorders/soma_spikes/"]
            g_bg = results_bg["recorders/soma_spikes/"]
            activity_control = get_activity(ps_grc.identifiers, g_ctrl, stim_start, stim_end)
            activity_gabazine = get_activity(ps_grc.identifiers, g_gabaz, stim_start, stim_end)
            activity_bg = get_activity(ps_grc.identifiers, g_bg, stim_start, stim_end)
            results.close()
            results_gabaz.close()
            results_bg.close()



            for i, id in enumerate(ps_grc.identifiers):
                coords = tuple(gpoints[i,:])
                if coords not in grid_control:
                    grid_control[coords] = []
                if coords not in grid_gabaz:
                    grid_gabaz[coords] = []
                if coords not in grid_bg:
                    grid_bg[coords] = []
                grid_control[coords].append(activity_control[id])
                grid_gabaz[coords].append(activity_gabazine[id])
                grid_bg[coords].append(activity_bg[id])

        control = [avg(v) for v in grid_control.values()]
        gabaz = [avg(v) for v in grid_gabaz.values()]
        bg = [avg(v) for v in grid_bg.values()]

        surface_xcoords = [int(k[0]) for k in grid_control.keys()]
        surface_ycoords = [int(k[1]) for k in grid_control.keys()]
        surface_z_control = np.zeros((int(max(surface_xcoords) + 1), int(max(surface_ycoords) + 1)))
        surface_z_gabazine = np.zeros((int(max(surface_xcoords) + 1), int(max(surface_ycoords) + 1)))
        surface_z_bg = np.zeros((int(max(surface_xcoords) + 1), int(max(surface_ycoords) + 1)))
        for i, (x, y) in enumerate(zip(surface_xcoords, surface_ycoords)):
            surface_z_control[x, y] = control[i]
            surface_z_gabazine[x, y] = gabaz[i]
            surface_z_bg[x, y] = bg[i]

        sigma = [0.8,0.8]
        smooth_control = scipy.ndimage.filters.gaussian_filter(surface_z_control, sigma)
        smooth_gabazine = scipy.ndimage.filters.gaussian_filter(surface_z_gabazine, sigma)
        smooth_bg = scipy.ndimage.filters.gaussian_filter(surface_z_bg, sigma)
        with open("smooth_control.pickle", "wb") as f:
            pickle.dump(smooth_control, f)
        with open("smooth_gabazine.pickle", "wb") as f:
            pickle.dump(smooth_gabazine, f)
        with open("smooth_bg.pickle", "wb") as f:
            pickle.dump(smooth_bg, f)
    else:
        with open("smooth_control.pickle", "rb") as f:
            smooth_control = pickle.load(f)
        with open("smooth_gabazine.pickle", "rb") as f:
            smooth_gabazine = pickle.load(f)
        with open("smooth_bg.pickle", "rb") as f:
            smooth_bg = pickle.load(f)

    figs = []
    for z in (smooth_control, smooth_gabazine, smooth_bg, (smooth_gabazine - smooth_bg)):
        fig = go.Figure(go.Surface(z=z, cmin=0, cmax=6,))
        fig.update_layout(
            scene=dict(
                zaxis_range=[0,6],
                xaxis_title="Y",
                yaxis_title="X",
                aspectratio=dict(x=2/3, y=1, z=1)
            )
        )
        figs.append(fig)
    # fig = go.Figure(go.Surface(z=(smooth_control - smooth_gabazine), cmin=-4, cmax=2,))
    # fig.update_layout(
    #     scene=dict(
    #         zaxis_range=[-4,2],
    #         xaxis_title="Y",
    #         yaxis_title="X",
    #         aspectratio=dict(x=2/3, y=1, z=1)
    #     )
    # )
    # figs.append(fig)
    return figs
