import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import ttest_rel
import os, sys, h5py, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "plots"))
import plotly.graph_objs as go, scipy.stats
from plotly.subplots import make_subplots
import pickle
from _paths import *
from glob import glob

def gauss_nd(coords, O, A, μ, σ):
    return O + A / (np.sqrt(2 * np.pi) * np.prod(σ)) * np.exp(-np.sum(np.power((coords - μ) / σ, 2)) / 2)

def gauss_expand_args(coords, O, A, *args):
    # Assert even number of args
    assert not (len(args) % 2)
    return gauss_nd(coords, O, A, np.array(args[:len(args) // 2]), np.array(args[len(args) // 2:]))

def make_scipy_curve(f):
    def curve(M, *args):
        return [f(point, *args) for point in M]

    return curve

single_gaussian_model = make_scipy_curve(gauss_expand_args)

def predict_activity_kernel(map, return_map=True):
    shape = map.shape
    idx = np.indices(shape).reshape(2, -1).T / shape
    lower_bounds = np.zeros(6)
    upper_bounds = np.ones(6)
    upper_bounds[0:2] = np.inf
    initial = np.ones(6) / 2
    ret = curve_fit(single_gaussian_model, idx, map.ravel(), p0=initial, bounds=(lower_bounds, upper_bounds))
    if return_map:
        out = np.array([gauss_expand_args(pair, *ret[0]) for pair in idx]).reshape(shape)
        return (*ret, out)
    else:
        return ret

def make_gaussian(shape, O, A, *args):
    idx = np.indices(shape).reshape(2, -1).T / shape
    np.array([gauss_expand_args(pair, *ret[0]) for pair in idx]).reshape(shape)


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

frozen = True

carries = {}
keys = ("ltp_cont", "ltd_cont")
border = 10
og_shape = surfaces[0]["ltp_cont"]["surface"].shape
# IMPORTANT: hardcoded, bin size might be different for your data
og_bins = 5
scale = ((og_shape[0] - 1) * og_bins, (og_shape[1] - 1) * og_bins)
border_mask = np.ones(og_shape, dtype=bool)
border_mask[:, :border] = 0
border_mask[:, -border:] = 0
border_mask[:border, :] = 0
border_mask[-border:, :] = 0
border_scale = ((og_shape[0] - border * 2 - 1) * og_bins, (og_shape[1] - border * 2 - 1) * og_bins)
border_offset = border * og_bins

if not frozen:
    for key in keys:
        carries[key] = []
        i = 0
        for id, (norm, sdict) in enumerate(zip(norm_maps.values(), surfaces)):
            i += 1
            ub_shape = tuple(np.array(border_mask.shape) - border * 2)
            surface = sdict[key]["surface"]
            normed = b_normed = surface / norm
            normed = normed[border_mask].reshape(ub_shape)
            # sigma = [2, 2]
            # surface = scipy.ndimage.filters.gaussian_filter(surface, sigma)
            # normed = scipy.ndimage.filters.gaussian_filter(normed, sigma)
            g_par, cov, gaussian = predict_activity_kernel(normed)
            # offset, amplitude, μx, μy, σx, σy = g_par
            fig = make_subplots(
                rows=1,
                cols=3,
                specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]],
            )
            fig.update_layout(title_text=f"Surface {id}")
            fig.add_trace(go.Surface(z=normed, name="norm"), row=1, col=1)
            fig.add_trace(go.Surface(z=gaussian, name="predicted"), row=1, col=2)
            fig.add_trace(go.Surface(z=normed, name="norm", opacity=0.3), row=1, col=3)
            fig.add_trace(go.Surface(z=gaussian, name="norm", opacity=0.3), row=1, col=3)
            fig.show()
            μx, μz = g_par[2:4]
            carries[key].append({
                "params": g_par,
                "covariance": cov,
                "pos": (border_offset + μx * border_scale[0], border_offset + μz * border_scale[1]),
                "dims": (og_shape, og_bins, scale, border),
                "surface": b_normed,
                "gauss": gaussian
            })
            print(f"Surface {id}:")
            print(" μx:", μx)
            print(" μz:", μz)
            print(" pos:", (border_offset + μx * border_scale[0], border_offset + μz * border_scale[1]))

    with open("pkl_ca/gaussian_offsets.pickle", "wb") as f:
        pickle.dump(carries, f)
else:
    with open("pkl_ca/gaussian_offsets.pickle", "rb") as f:
        carries = pickle.load(f)

import itertools

def overlay_maps(maps, offsets):
    pass

# ltp_kernel_sizes = np.array([est["params"][4:6] for est in carries["ltp_cont"]])
# ltd_kernel_sizes = np.array([est["params"][4:6] for est in carries["ltd_cont"]])
# res = ttest_rel(ltp_kernel_sizes[:, 0], ltd_kernel_sizes[:, 0])
# print(res)
# res = ttest_rel(ltp_kernel_sizes[:, 1], ltd_kernel_sizes[:, 1])
# print(res)
#

centered = {}

for key, estimates in carries.items():
    kernel_positions = np.array(list(est["pos"] for est in estimates))
    kernel_sizes = np.array(list(est["params"][4:6] for est in estimates))
    offsets = np.round((kernel_positions - np.array(estimates[0]["dims"][2]) / 2) / estimates[0]["dims"][1])
    overlap = np.zeros(np.array(og_shape) * 2)
    zero = np.array(og_shape) / 2 + border
    for id, (est, offset) in enumerate(zip(estimates, offsets)):
        zx, zz = (zero - offset).astype(int)
        surface = est["surface"]
        shape = tuple(np.array(border_mask.shape) - border * 2)
        unbordered = surface[border_mask].reshape(shape)
        overlap[zx:(zx + shape[0]), zz:(zz + shape[1])] = unbordered
    overlap /= len(estimates)
    # sigma = [2, 2]
    # overlap = scipy.ndimage.filters.gaussian_filter(overlap, sigma)
    fig = go.Figure()
    fig.update_layout(title_text=f"Overlap {key}")
    fig.add_trace(go.Surface(z=overlap, name="overlap"))
    fig.show()
    centered[key] = overlap

fig = go.Figure()
fig.update_layout(title_text=f"Difference LTP - LTD")
fig.add_trace(go.Surface(z=centered["ltp_cont"] - centered["ltd_cont"], name="overlap"))
fig.show()
