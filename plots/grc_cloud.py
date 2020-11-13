import os, plotly.graph_objects as go
from bsb.core import from_hdf5
import numpy as np, h5py
from scipy import stats

colorbar_grc = ['rgb(158,188,218)', 'rgb(140,150,198)', 'rgb(140,107,177)', 'rgb(136,65,157)', 'rgb(129,15,124)', 'rgb(77,0,75)']
colorbar_pc = "thermal"

def granule_kde(network_file, results_file, base_start=400, base_end=600, stim_start=700, stim_end=800):

    def crop(data, min, max, indices=False):
        c = data[:, 1]
        if indices:
            return np.where((c > min) & (c < max))[0]
        return c[(c > min) & (c < max)]

    def get_isis(spikes, selected):
        return [spikes[i + 1] - spikes[i] for i in selected]

    def get_parallel(subset, set):
        return np.where(np.isin(set, subset))[0]

    inv = lambda x: [1000 / y for y in x]
    avg = lambda x: sum(x) / len(x)

    print("Loading network", " " * 30, end="\r")
    scaffold = from_hdf5(network_file)
    print(" " * 30, end="\r")
    with h5py.File(results_file, "r") as results:
        ps = scaffold.get_placement_set("granule_cell")
        ids = ps.identifiers
        spikes_per_dict = {id: [] for id in ids}
        print("Scanning granule spikes", " " * 30, end="\r")
        g = results["recorders/soma_spikes/"]
        for key in g:
            f = g[key]
            if f.attrs["label"] != ps.tag or len(f) == 0:
                continue
            id = int(f[0, 0])
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
        return ids[pos_roi], pos[pos_roi], norm

def granule_cloud(network_file, results_file, base_start=400, base_end=600, stim_start=700, stim_end=800):
    ids, pos, norm = granule_kde(network_file, results_file, base_start, base_end, stim_start, stim_end)
    print("Plotting granule cloud", " " * 30, end="\r")
    return go.Scatter3d(x=pos[:,0],y=pos[:,2],z=pos[:,1],
        name="Active granule cells",
        text=["ID: " + str(int(id)) for id in ids],
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

def granule_disc(network_file, results_file, base_start=400, base_end=600, stim_start=700, stim_end=800, bar_l=0.8):
    ids, pos, norm = granule_kde(network_file, results_file, base_start, base_end, stim_start, stim_end)
    print("Plotting granule disc", " " * 30, end="\r")
    return go.Scatter(x=pos[:,0],y=pos[:,2],
        name="Active granule cells",
        text=["ID: " + str(int(id)) for id in ids],
        mode="markers",
        marker=dict(
            colorscale=colorbar_grc, cmin=0, cmax=1,
            opacity=0.3,
            color=norm,
            size=9,
            colorbar=dict(
                len=bar_l,
                xanchor="left",
                title=dict(
                    text="Activity density",
                    side="bottom"
                )
            )
        )
    )
