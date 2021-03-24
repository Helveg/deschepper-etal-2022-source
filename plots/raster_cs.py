import os, plotly.graph_objects as go
from plotly.subplots import make_subplots
from bsb.core import from_hdf5
from bsb.plotting import hdf5_plot_psth, hdf5_plot_spike_raster
import numpy as np, h5py
from scipy import stats
import selection
import itertools, functools
from ._paths import *

def sorter(grc_pos_map):
    def sort(kv):
        _, id = kv
        return -sum((np.array([150, 100]) - grc_pos_map[id.attrs["cell_id"]][[0,2]]) ** 2) ** (1/2)

    return sort

def select_groups(kv):
    name, group = kv
    return group.attrs["label"] == "granule_cell"

@functools.cache
def nth_after(n, after):
    def nafter(ds):
        times = ds[:, 1]
        x = np.nonzero(times > after)[0][n:(n+1)]
        if not len(x):
            return []
        return times[x[0]:(x[0]+1)]

    return nafter

def plot(path_control=None, path_gaba=None, net_path=None):
    if path_control is None or path_gaba is None:
        ValueError("Give control and gaba paths")
    if net_path is None:
        net_path = network_path(selection.network)
    netw = from_hdf5(net_path)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
    grc_pos_map = dict(zip((grc_ps := netw.get_placement_set("granule_cell")).identifiers, grc_ps.positions))
    plot_condition(fig, 1, path_control, grc_pos_map)
    plot_condition(fig, 2, path_gaba, grc_pos_map)
    return fig

def plot_condition(fig, row, path, grc_pos_map):
    colors = ("blue", "red", "purple", "lime")
    with h5py.File(path, "r") as f:
        groups = {k: v for k, v in sorted(filter(select_groups, f["/recorders/soma_spikes"].items()), key=sorter(grc_pos_map))}
        ys = list(itertools.chain(*(np.ones(len(y)) * i for i, y in enumerate(groups.values()))))
        xs = list(itertools.chain(*map(lambda x: x[:,1], groups.values())))
        fig.add_trace(go.Scatter(name="all spikes", x=xs, y=ys, mode="markers", marker=dict(size=1, color=colors[0])), row=row, col=1)
        for n in range(0, 3):
            fig.add_trace(
                go.Scatter(
                    name=f"{n+1}th spikes",
                    y=list(itertools.chain(*(np.ones(len(nth_after(n, 1000)(y))) * i for i, y in enumerate(groups.values())))),
                    x=list(itertools.chain(*map(nth_after(n, 1000), groups.values()))),
                    mode="markers",
                    marker=dict(size=(n + 1), color=colors[n+1])
                ),
                row=row,
                col=1
            )
    return fig
