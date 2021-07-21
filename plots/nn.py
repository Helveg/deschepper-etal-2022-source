from bsb.core import from_hdf5
from scipy.spatial import KDTree
from plotly import graph_objs as go
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
    fig.update_layout(bargap=0, bargroupgap=0, title_text="Nearest neighbour")
    for i, ct in enumerate(sorted(network.get_cell_types(entities=False), key=lambda c: select.index(c.name))):
        ps = network.get_placement_set(ct)
        print("Looking into", ps.tag)
        tree = KDTree(ps.positions)
        matches, ind = tree.query(ps.positions, k=2)
        fig.add_trace(
            go.Histogram(x=matches[:, 1], marker_line_width=0, marker_color=ct.plotting.color, name=ct.plotting.label),
            row=1,
            col=i + 1,
        )
        fig.update_xaxes(row=1, col=i + 1, range=[0, np.max(matches[:, 1]) + np.min(matches[:, 1])])
    fig.update_yaxes(visible=False)
    return fig

def meta():
    return {"height": 250, "width": 850}
