import os, sys, h5py, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "plots"))
import plotly.graph_objs as go, scipy.stats
from plotly.subplots import make_subplots
import pickle
from _paths import *
from glob import glob

with open("pkl_ca/norms.pickle", "rb") as p:
    norm_maps = pickle.load(p)

surfaces = []
id = 0
try:
    while True:
        with open(f"pkl_ca/sum_maps/calcium_rawsum_{id}.pickle", "rb") as p:
            surfaces.append(pickle.load(p))
            id += 1
except:
    pass

carries = {}
keys = ("ltp_cont", "ltd_cont")
for key in keys:
    carries[key] = carry = None
    i = 0
    for id, (norm, sdict) in enumerate(zip(norm_maps.values(), surfaces)):
        i += 1
        surface = sdict[key]["surface"]
        sigma = [2, 2]
        surface = scipy.ndimage.filters.gaussian_filter(surface, sigma)
        # fig = make_subplots(
        #     rows=1,
        #     cols=3,
        #     specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]],
        # )
        # fig.update_layout(title_text=f"Surface {id}")
        # fig.add_trace(go.Surface(z=surface, name="surface"), row=1, col=1)
        # fig.add_trace(go.Surface(z=norm, name="norm"), row=1, col=2)
        # fig.add_trace(go.Surface(z=surface / norm, name="norm"), row=1, col=3)
        # fig.show()
        carry = surface / norm if carry is None else carry + surface / norm
    carry /= i
    carries[key] = carry

fig = make_subplots(
    rows=1,
    cols=3,
    specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]],
)
fig.add_trace(go.Surface(z=carries["ltp_cont"], name="avg ltp"), row=1, col=1)
fig.add_trace(go.Surface(z=carries["ltd_cont"], name="avg ltd"), row=1, col=2)
fig.add_trace(go.Surface(z=carries["ltp_cont"] - carries["ltd_cont"], name="avg"), row=1, col=3)
fig.show()
