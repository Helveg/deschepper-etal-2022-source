import os, plotly.graph_objects as go
from bsb.core import from_hdf5
import numpy as np, h5py
from scipy import stats
import scipy.ndimage

colorbar_grc = ['rgb(158,188,218)', 'rgb(140,150,198)', 'rgb(140,107,177)', 'rgb(136,65,157)', 'rgb(129,15,124)', 'rgb(77,0,75)']
colorbar_pc = "thermal"


network_path = os.path.join(
    os.path.dirname(__file__), "..", "networks", "300x_200z.hdf5"
)

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

def plot():
    base_start, base_end = 400, 600
    stim_start, stim_end = 700, 800
    print("Loading network", " " * 30, end="\r")
    scaffold = from_hdf5(network_path)
    results = h5py.File("/home/claudia/deschepper-etal-2020/results/results_stim_on_MFs_noBackg_50HzPoiss2mfs100ms_forCS.hdf5", "r")

    resultsGABAZ = h5py.File("/home/claudia/deschepper-etal-2020/results/results_stim_on_MFs_noBackg_50HzPoiss2mfs100ms_forCS_GABAZINE.hdf5", "r")

    print("Loading granule", " " * 30, end="\r")
    ps_pc = scaffold.get_placement_set("granule_cell")
    pc_pos = ps_pc.positions
    border_pc = pc_pos[:, 2] > 500
    cut_off = []


    print("cutoff", cut_off)
    print(len(ps_pc.identifiers))

    g = results["recorders/soma_spikes/"]
    numSpikesStim = {int(id): 0 for id in ps_pc.identifiers}
    pc_indices = [ps_pc.identifiers.tolist().index(id) for id in numSpikesStim.keys() if id not in cut_off]
    dur = stim_end - stim_start


    #pc_freq = {int(id): 0 for id in ps_pc.identifiers}
    #pc_lo_freq = {int(id): 0 for id in ps_pc.identifiers}
    #print("Scanning granule spikes", " " * 30, end="\r")
    for key in g:
        f = g[key]
        spikes = crop(f[()], min=stim_start, max=stim_end, indices=True)
        spikesBase = crop(f[()], min=base_start, max=base_end, indices=True)
        if f.attrs["label"] != ps_pc.tag or len(f) == 0:
            continue
        id = int(f[0, 0])
        numSpikesStim[id] = len(spikes)#-len(spikesBase)

    maxSpikes=1 #max(numSpikesStim.values())
    #Activity = {id: v if len(v) else 0 for id, v in numSpikesStim.items() if id not in cut_off}
    ActivityNorm = normaliz(numSpikesStim, maxSpikes)



    gGABAZ = resultsGABAZ["recorders/soma_spikes/"]
    numSpikesStim = {int(id): 0 for id in ps_pc.identifiers}
    for key in gGABAZ:
        f = gGABAZ[key]
        spikes = crop(f[()], min=stim_start, max=stim_end, indices=True)
        spikesBase = crop(f[()], min=base_start, max=base_end, indices=True)
        if f.attrs["label"] != ps_pc.tag or len(f) == 0:
            continue
        id = int(f[0, 0])
        numSpikesStim[id] = len(spikes) #-len(spikesBase)

    maxSpikes=1 #max(numSpikesStim.values())
    #Activity = {id: v if len(v) else 0 for id, v in numSpikesStim.items() if id not in cut_off}
    ActivityGABAZNorm = normaliz(numSpikesStim, maxSpikes)

    CenterSurr = {id: ActivityNorm[id] for id, v in ActivityNorm.items() if id not in cut_off} #ActivityNorm[id]- ActivityGABAZNorm[id]
    # pixel  x z
    min_c = min([v for k, v in CenterSurr.items() if k not in cut_off])
    max_c = max([v for k, v in CenterSurr.items() if k not in cut_off])

    pc_activity = go.Scatter3d(
            name="cells",
            x=pc_pos[pc_indices, 0],
            y=pc_pos[pc_indices, 2],
            z=pc_pos[pc_indices, 1],
            text=["ID: " + str(int(i)) + "\nd: " + str(round(d, 2))  for i, d in CenterSurr.items()],
            mode="markers",
            marker=dict(
                colorscale=colorbar_pc,
                cmin=min_c,
                cmax=max_c,
                opacity=0.2,
                size=5,
                color=list(CenterSurr.values()),
                colorbar=dict(
                    len=0.8,
                    x=0.8,
                    title=dict(
                        text="Change in activity (#spikes)",
                        side="bottom"
                    )
                )
            )
        )

    #fig = go.Figure(pc_activity)
    # axis_labels = dict(xaxis_title="X", yaxis_title="Z", zaxis_title="Y")
    # fig.update_layout(scene=axis_labels, title_text="cell activity", title_x=0.5)
    #return fig

    points= ps_pc.positions[pc_indices, :][:, [0, 2]]
    grid_offset = np.array([0.0, 0.0])  # x z
    grid_spacing = np.array([5., 5.])  #um
    gpoints = grid_offset + np.round((points - grid_offset) / grid_spacing) * grid_spacing


    GridAvgCenterSurr = {}
    #for i, v in enumerate(CenterSurr.values()):
    for i, v in enumerate(CenterSurr.values()):
        coords = tuple(gpoints[i,:])
        if coords not in GridAvgCenterSurr:
            GridAvgCenterSurr[coords] = []
        GridAvgCenterSurr[coords].append(v)

    X = [k[0] for k in GridAvgCenterSurr.keys()]
    print(min(X), max(X))
    Y = [k[1] for k in GridAvgCenterSurr.keys()]
    print(min(Y), max(Y))
    Z = [avg(v) for v in GridAvgCenterSurr.values()]
    print(len(Z), len(X), len(Y))
    z = np.zeros((int(max(X) / grid_spacing[0]) + 1, int(max(Y) /grid_spacing[1]) + 1))
    for i, (x, y) in enumerate(zip(X, Y)):
        z[int(x/grid_spacing[0]), int(y/grid_spacing[1])] = Z[i]
    print(z.shape)
    print(z[0,0])
    print(z[30,20])
    # print(list(GridAvgCenterSurr.keys())[0] )
    # print(list(GridAvgCenterSurr.keys())[-1] )

    sigma = [0.8,0.8]
    smoothZ= scipy.ndimage.filters.gaussian_filter(z, sigma)

    fig = go.Figure(go.Surface(z=smoothZ))
    fig.update_layout(
        scene=dict(
            # xaxis=dict(title="X", range=[0, 40], autorange=False),
            # yaxis=dict(title="Z", range=[0, 60], autorange=False),
            aspectratio=dict(x = 2/3, y = 1)
        )
    )
    return fig


    # min_c = min([v for k, v in pc_delta_freq.items() if k not in cut_off])
    # max_c = max([v for k, v in pc_delta_freq.items() if k not in cut_off])
    #
    # with h5py.File(results_path("../selected_results.hdf5"), "r") as f:
    #     print("Plotting granule activity", " " * 30, end="\r")
    #     pc_all = go.Scatter3d(
    #         name="All granule cells",
    #         x=pc_pos[:, 0],
    #         y=pc_pos[:, 2],
    #         z=pc_pos[:, 1],
    #         mode="markers",
    #         marker=dict(
    #             color="grey",
    #             opacity=0.2
    #         )
    #     )
    #     pc_border = go.Scatter3d(
    #         name="Border-excluded granule cells",
    #         x=pc_pos[border_pc, 0],
    #         y=pc_pos[border_pc, 2],
    #         z=pc_pos[border_pc, 1],
    #         mode="markers",
    #         marker=dict(
    #             color="red"
    #         )
    #     )
    #     pc_activity = go.Scatter3d(
    #         name="granule cells",
    #         x=pc_pos[pc_indices, 0],
    #         y=pc_pos[pc_indices, 2],
    #         z=pc_pos[pc_indices, 1],
    #         text=["ID: " + str(int(i)) + "\ndf: " + str(round(d, 2)) + "Hz" for i, d in pc_delta_freq.items()],
    #         mode="markers",
    #         marker=dict(
    #             colorscale=colorbar_pc,
    #             cmin=min_c,
    #             cmax=max_c,
    #             opacity=0.2,
    #             size=4,
    #             color=list(pc_CenterSurr.values()),
    #             colorbar=dict(
    #                 len=0.8,
    #                 x=0.8,
    #                 title=dict(
    #                     text="Change in activity (Hz)",
    #                     side="bottom"
    #                 )
    #             )
    #         )
    #     )
    #     # pc_activities = [pc_activity]
    #     # for h in f["selections"]:
    #     #     if "granule" in h:
    #     #         pcs = np.array([id for id in f["selections"][h][()] if id not in cut_off])
    #     #         pi = get_parallel(pcs, ps_pc.identifiers)
    #     #         delta = [pc_delta_freq[id] for id in pcs]
    #     #         kwargs = dict(zip(["x","z","y"], pc_pos[pi].T))
    #     #         kwargs["text"] = ["ID: {}\n".format(pcs[i]) + str(round(delta[i], 2)) + "Hz" for i in range(len(pcs))]
    #     #         scatter = go.Scatter3d(
    #     #             name="Subset " + h, **kwargs,
    #     #             mode="markers",
    #     #             marker=dict(
    #     #                 colorscale=colorbar_pc,
    #     #                 cmin=min_c,
    #     #                 cmax=max_c,
    #     #                 color=delta,
    #     #             )
    #     #         )
    #             # pc_activities.append(scatter)
    #
    # print(" " * 80, end="\r")
    # fig = go.Figure(pc_activity)
    # axis_labels = dict(xaxis_title="X", yaxis_title="Z", zaxis_title="Y")
    # fig.update_layout(scene=axis_labels, title_text="granule cell activity", title_x=0.5)
    # return fig
