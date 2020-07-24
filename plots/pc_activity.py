import os, plotly.graph_objects as go
from scaffold.core import from_hdf5
import numpy as np, h5py
from scipy import stats

colorbar_grc = ['rgb(158,188,218)', 'rgb(140,150,198)', 'rgb(140,107,177)', 'rgb(136,65,157)', 'rgb(129,15,124)', 'rgb(77,0,75)']
colorbar_pc = "thermal"


network_path = os.path.join(
    os.path.dirname(__file__), "..", "networks", "results.hdf5"
)

def results_path(*args):
    return os.path.join(
        os.path.dirname(__file__), "..", "results", *args
    )

def crop(data, min, max, indices=False):
    c = data[:, 0]
    if indices:
        return np.where((c > min) & (c < max))[0]
    return c[(c > min) & (c < max)]

def get_isis(spikes, selected):
    return [spikes[i + 1] - spikes[i] for i in selected]

inv = lambda x: [1000 / y for y in x]
avg = lambda x: sum(x) / len(x)


def plot():
    base_start, base_end = 400, 600
    stim_start, stim_end = 700, 800
    print("Loading network", " " * 30, end="\r")
    scaffold = from_hdf5(network_path)
    results = h5py.File(results_path("first_run_300_200.hdf5"), "r")
    ps = scaffold.get_placement_set("granule_cell")
    ids = ps.identifiers
    spikes_per_dict = {id: [] for id in ids}
    print("Scanning granule spikes", " " * 30, end="\r")
    g = results["recorders/soma_spikes/"]
    for key in g:
        f = g[key]
        if f.attrs["label"] != ps.tag or len(f) == 0:
            continue
        id = int(f[0, 1])
        spikes_per_dict[id].extend(crop(f[()], min=stim_start, max=stim_end))

    spikes_per_grc = [spikes_per_dict[x] for x in sorted(spikes_per_dict)]
    n_spikes_per_grc = map(len, spikes_per_grc)
    np_spikes = np.array(list(map(len, spikes_per_grc)))
    norm_spikes = np_spikes / np.max(np_spikes)
    pos = np.array(ps.positions)
    pos_arr = []
    kde_counter = 0
    kde_roi = []
    pos_roi = []
    firing_cells = 0
    print("Selecting granule Region of Interest", " " * 30, end="\r")
    for i, n_spikes in enumerate(n_spikes_per_grc):
        if n_spikes != 0:
            firing_cells += 1
            kde_roi.append(kde_counter)
            pos_roi.append(i)
        kde_counter += n_spikes
        p = pos[i]
        pos_arr.extend(p for _ in range(n_spikes))
    spikes_in_space = np.array(pos_arr)
    print("Performing Kernel Density Estimation", " " * 30, end="\r")
    values = spikes_in_space.T
    kde = stats.gaussian_kde(values)
    density = kde(values)
    grc_densities = density[kde_roi]
    norm = grc_densities / np.max(grc_densities)
    _fullpos = pos
    pos = pos[pos_roi]
    print("Plotting granule cloud", " " * 30, end="\r")
    granule_cloud = go.Scatter3d(x=pos[:,0],y=pos[:,2],z=pos[:,1],
        name="Active granule cells",
        mode="markers",
        marker=dict(
            colorscale=colorbar_grc, cmin=0, cmax=1,
            opacity=0.05,
            color=norm,
            colorbar=dict(
                len=0.8,
                xanchor="right",
                x=0.1,
                title=dict(
                    text="Activity density",
                    side="bottom"
                )
            )
        )
    )

    print("Loading purkinje", " " * 30, end="\r")
    ps_pc = scaffold.get_placement_set("purkinje_cell")
    pc_pos = ps_pc.positions
    cs_aa = scaffold.get_connectivity_set("ascending_axon_to_purkinje")
    cs_pf = scaffold.get_connectivity_set("parallel_fiber_to_purkinje")
    cs_bc = scaffold.get_connectivity_set("basket_to_purkinje")
    cs_sc = scaffold.get_connectivity_set("stellate_to_purkinje")
    cut_off = list(ps_pc.identifiers[ps_pc.positions[:, 2] > 150])

    g = results["recorders/soma_spikes/"]
    pc_isis = {int(id): [] for id in ps_pc.identifiers}
    pc_lo_isis = {int(id): [] for id in ps_pc.identifiers}
    dur = stim_end - stim_start
    pc_freq = {int(id): [] for id in ps_pc.identifiers}
    print("Scanning purkinje spikes", " " * 30, end="\r")
    for key in g:
        f = g[key]
        lospikes = crop(f[()], min=base_start, max=base_end, indices=True)
        spikes = crop(f[()], min=stim_start, max=stim_end, indices=True)
        if f.attrs["label"] != ps_pc.tag or len(f) == 0:
            continue
        id = int(f[0, 1])
        pc_lo_isis[id].extend(get_isis(f[:, 0], lospikes))
        pc_isis[id].extend(get_isis(f[:, 0], spikes))
    pc_indices = [ps_pc.identifiers.tolist().index(id) for id in pc_isis.keys() if id not in cut_off]
    pc_isi_freq = {id: avg(inv(v)) for id, v in pc_isis.items() if id not in cut_off}
    pc_delta_freq = {id: avg(inv(v)) - avg(inv(pc_lo_isis[id])) for id, v in pc_isis.items() if id not in cut_off}

    print("Plotting purkinje activity", " " * 30, end="\r")
    pc_activity = go.Scatter3d(
        name="Purkinje cells",
        x=pc_pos[pc_indices, 0],
        y=pc_pos[pc_indices, 2],
        z=pc_pos[pc_indices, 1],
        text=[str(round(d, 2)) + "Hz" for d in pc_delta_freq.values()],
        mode="markers",
        marker=dict(
            colorscale=colorbar_pc,
            cmin=min(pc_delta_freq.values()),
            cmax=max(pc_delta_freq.values()),
            color=list(pc_delta_freq.values()),
            colorbar=dict(
                len=0.8,
                title=dict(
                    text="Change in activity (Hz)",
                    side="bottom"
                )
            )
        )
    )
    print(" " * 80, end="\r")
    fig = go.Figure([granule_cloud, pc_activity])
    axis_labels = dict(xaxis_title="X", yaxis_title="Z", zaxis_title="Y")
    fig.update_layout(scene=axis_labels, title_text="Purkinje cell activity", title_x=0.5)
    return fig
