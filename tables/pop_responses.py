from bsb.core import from_hdf5
from bsb.plotting import plot_morphology, hdf5_gather_voltage_traces, plot_traces
from plotly import graph_objs as go
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import plots.selection as selection, numpy as np, h5py
from colour import Color
from plots._paths import *
from glob import glob
import itertools, pickle
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

frozen = False

def table(path=None, net_path=None, bg_start=5700, bg_end=5900, stim_start=6000, stim_end=6040):
    if path is None:
        paths = glob(results_path("balanced_sensory", "*.hdf5"))
    else:
        paths = glob(path)
    if net_path is None:
        net_path = network_path(selection.network)
    stimulated_mfs = selection.stimulated_mf_poiss
    network = from_hdf5(net_path)
    def listgen(list_of_list_len=None):
        while True:
            if list_of_list_len is not None:
                yield [[]] * list_of_list_len
            yield []

    def dictgen():
        while True:
            yield dict()

    if not frozen:
        bg_spikes = dict(itertools.chain(*(zip(ct.get_placement_set().identifiers, listgen()) for ct in network.get_cell_types())))
        stim_spikes = dict(itertools.chain(*(zip(ct.get_placement_set().identifiers, listgen()) for ct in network.get_cell_types())))
        inc_spikes = dict(itertools.chain(*(zip(ct.get_placement_set().identifiers, dictgen()) for ct in network.get_cell_types())))
        out_map = dict(itertools.chain(*(zip(ct.get_placement_set().identifiers, dictgen()) for ct in network.get_cell_types(entities=True))))
        ct_map = {}
        for cs in network.get_connectivity_sets():
            ct_from = cs.get_presynaptic_types()[0]
            ct_to = cs.get_postsynaptic_types()[0]
            l = ct_map.setdefault(ct_to.name, [])
            l.append(cs.tag)
            for fid, tid in cs.get_dataset():
                l = out_map[int(fid)].setdefault(cs.tag, [])
                l.append(int(tid))
        for i, path in enumerate(paths):
            print("Processing", i, len(paths), path)
            with h5py.File(path, "r") as f:
                for id, ds in f["recorders/soma_spikes"].items():
                    t = ds[:, 1]
                    bg_spikes[(id := int(id))].append(sum((t >= bg_start) & (t <= bg_end)))
                    out_spikes = sum((t >= stim_start) & (t <= stim_end))
                    stim_spikes[id].append(out_spikes)
                    if (outgoing := out_map[id]):
                        for out_cat, out_ids in outgoing.items():
                            for out_id in out_ids:
                                l = inc_spikes[out_id].setdefault(out_cat, [0 for _ in range(len(paths))])
                                l[i] += out_spikes
                for id, ds in f["recorders/input/background"].items():
                    id = int(float(id))
                    t = ds[()]
                    out_spikes = sum((t >= stim_start) & (t <= stim_end))
                    if id in stimulated_mfs:
                        out_spikes += 5
                    for glom in out_map[id].get("mossy_to_glomerulus", []):
                        if (outgoing := out_map[glom]):
                            for out_cat, out_ids in outgoing.items():
                                for out_id in out_ids:
                                    l = inc_spikes[out_id].setdefault(out_cat, [0 for _ in range(len(paths))])
                                    l[i] += out_spikes

        with open("pop_responses.pkl", "wb") as f:
            pickle.dump((bg_spikes, stim_spikes, inc_spikes, out_map, ct_map), f)
    else:
        print("Using cached results.")
        with open("pop_responses.pkl", "rb") as f:
            bg_spikes, stim_spikes, inc_spikes, out_map, ct_map = pickle.load(f)
    bg_p = (bg_end - bg_start) / 1000
    stim_p = (stim_end - stim_start) / 1000
    inhibitory = {"basket_to_basket", "stellate_to_stellate", "golgi_to_golgi", "basket_to_purkinje", "stellate_to_purkinje", "golgi_to_granule"}
    tbl = [["connection_type", "coeff", "score"]]
    for ct in network.get_cell_types():
        if ct.name == "glomerulus" or ct.name == "mossy_fibers":
            continue
        ids = ct.get_placement_set().identifiers
        istim = np.array([np.mean(stim_spikes[id]) / stim_p for id in ids])
        ispikes = np.array([[np.mean(inc_spikes[id].get(cstag, [0])) for cstag in ct_map[ct.name]] for id in ids])
        regressor = LinearRegression().fit(ispikes, istim)
        nonlinear = RandomForestRegressor(max_depth=4).fit(ispikes, istim)
        print("--- ", ct.name)
        print("score", regressor.score(ispikes, istim))
        score = regressor.score(ispikes, istim)
        print("nonlin. score", nonlinear.score(ispikes, istim))
        print("coeff", dict(zip(ct_map[ct.name], regressor.coef_)))
        print("intercept", regressor.intercept_)
        tbl.extend([name, coef, score] for name, coef in zip(ct_map[ct.name], regressor.coef_))

    return tbl
