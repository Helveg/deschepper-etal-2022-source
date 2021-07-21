from bsb.core import from_hdf5
from scipy.spatial import KDTree
from plotly import graph_objs as go
from scipy import stats
from plotly.subplots import make_subplots
import numpy as np
from ._paths import *
from glob import glob
import selection

def plot(net_path=None):
    if net_path is None:
        net_path = network_path(selection.network)
    network = from_hdf5(net_path)
    select = ["glomerulus", "granule_cell", "golgi_cell", "purkinje_cell", "basket_cell", "stellate_cell"]
    fig = make_subplots(cols=6, rows=1, x_title="Distance [μm]")
    fig.update_layout(title_text="Nearest neighbour")
    for i, ct in enumerate(sorted(network.get_cell_types(entities=False), key=lambda c: select.index(c.name))):
        ps = network.get_placement_set(ct)
        print("Looking into", ps.tag)
        tree = KDTree(ps.positions)
        matches, ind = tree.query(ps.positions, k=2)
        kde = stats.gaussian_kde(matches[:, 1])
        m = np.max(matches[:, 1]) + np.min(matches[:, 1])
        x = np.linspace(0, m, 1000)
        y = np.where(x < np.min(matches[:, 1]), 0, kde(x))
        fig.add_trace(
            go.Scatter(x=x, y=y, line_color=ct.plotting.color, name=ct.plotting.label),
            row=1,
            col=i + 1,
        )
    return fig

def meta():
    return {"height": 250, "width": 850}
