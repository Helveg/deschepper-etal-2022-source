from bsb.core import from_hdf5
from scipy.spatial import KDTree
from plotly import graph_objs as go
import numpy as np, pickle
from radialdf import inner_rdf
from plotly.subplots import make_subplots

frozen = False
from ._paths import *
from glob import glob
import selection

def plot(net_path=None):
    if net_path is None:
        net_path = network_path(selection.network)
    network = from_hdf5(net_path)
    select = ["glomerulus", "granule_cell", "golgi_cell", "purkinje_cell", "basket_cell", "stellate_cell"]
    fig = make_subplots(cols=6, rows=1, x_title="Radial distance [μm]", y_title="Density per average density")
    fig.update_layout(bargap=0, bargroupgap=0, title_text="Radial distribution function")
    for i, ct in enumerate(sorted(network.get_cell_types(entities=False), key=lambda c: select.index(c.name))):
        ps = network.get_placement_set(ct)
        dr = 0.5
        if not frozen:
            if ps.tag in ["granule_cell", "glomerulus", "golgi_cell"]:
                gr = inner_rdf(np.array([[0, 300], [0, 130], [0, 200]]), ps.positions, 30, dr)
            elif ps.tag in ["basket_cell", "stellate_cell"]:
                gr = inner_rdf(np.array([[0, 300], [145, 295], [0, 200]]), ps.positions, 30, dr)
            elif ps.tag == "purkinje_cell":
                gr = inner_rdf(np.array([[0, 300], [0, 200]]), ps.positions[:, [0, 2]], 30, dr)
            with open(f"gr_{ps.tag}.pickle", "wb") as f:
                pickle.dump(gr, f)
        else:
            with open(f"gr_{ps.tag}.pickle", "rb") as f:
                gr = pickle.load(f)
        fig.add_trace(
            go.Bar(x=np.arange(0, len(gr) * dr, dr), y=gr, marker_line_width=0, marker_color=ct.plotting.color, name=ct.plotting.label),
            row=1,
            col=i + 1,
        )
    return fig

def meta():
    return {"height": 400, "width": 850}
