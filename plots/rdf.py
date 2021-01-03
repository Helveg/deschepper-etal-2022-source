from bsb.core import from_hdf5
from scipy.spatial import KDTree
from plotly import graph_objs as go
import numpy as np, pickle
from radialdf import rdf

frozen = True

def plot():
    network = from_hdf5("networks/300x_200z.hdf5")

    figs = {}
    for ct in network.get_cell_types(entities=False):
        ps = network.get_placement_set(ct)
        dr = 0.5
        if not frozen:
            if ps.tag in ["granule_cell", "glomerulus", "golgi_cell"]:
                gr = rdf(np.array([[0, 300], [0, 130], [0, 200]]), ps.positions, 30, dr)
            elif ps.tag in ["basket_cell", "stellate_cell"]:
                gr = rdf(np.array([[0, 300], [145, 295], [0, 200]]), ps.positions, 30, dr)
            elif ps.tag == "purkinje_cell":
                gr = rdf(np.array([[0, 300], [0, 200]]), ps.positions[:, [0, 2]], 30, dr)
            with open(f"gr_{ps.tag}.pickle", "wb") as f:
                pickle.dump(gr, f)
        else:
            with open(f"gr_{ps.tag}.pickle", "rb") as f:
                gr = pickle.load(f)
        figs[ps.tag] = go.Figure(go.Scatter(x=np.arange(0, len(gr) * dr, dr), y=gr, mode="lines"), layout=dict(title_text=ps.tag))


    return figs
