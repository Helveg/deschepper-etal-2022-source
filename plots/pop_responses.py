from bsb.core import from_hdf5
from bsb.plotting import plot_morphology, hdf5_gather_voltage_traces, plot_traces
from plotly import graph_objs as go
import selection, numpy as np, h5py
from colour import Color
from ._paths import *
from glob import glob
import selection, itertools, pickle
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

frozen = True

def plot(path=None, net_path=None, bg_start=5700, bg_end=5900, stim_start=6000, stim_end=6040):
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
            if cs.tag == "gap_goc": continue
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
                    for glom in out_map[id]["mossy_to_glomerulus"]:
                        if (outgoing := out_map[glom]):
                            for out_cat, out_ids in outgoing.items():
                                for out_id in out_ids:
                                    l = inc_spikes[out_id].setdefault(out_cat, [0 for _ in range(len(paths))])
                                    l[i] += out_spikes

        with open("pop_responses.pkl", "wb") as f:
            pickle.dump((bg_spikes, stim_spikes, inc_spikes, out_map, ct_map), f)
    else:
        with open("pop_responses.pkl", "rb") as f:
            bg_spikes, stim_spikes, inc_spikes, out_map, ct_map = pickle.load(f)
    figs = {}
    bg_p = (bg_end - bg_start) / 1000
    stim_p = (stim_end - stim_start) / 1000
    inhibitory = {"basket_to_basket", "stellate_to_stellate", "golgi_to_golgi", "basket_to_purkinje", "stellate_to_purkinje", "golgi_to_granule"}
    label_map = {
        "glomerulus_to_granule": "Glom-Grc spikes", "golgi_to_granule": "GoC-GrC spikes",
        "parallel_fiber_to_basket": "GrC (pf)-BC", "basket_to_basket": "BC-BC",
        "parallel_fiber_to_stellate": "GrC (pf)-SC", "stellate_to_stellate": "SC-SC",
    }
    default_cam = dict(up=dict(x=0,y=0,z=1),center=dict(x=0,y=0,z=0),eye=dict(x=-1.6178192877772763,y=1.420370800120278,z=0.679576755143629))
    cameras = {
        "granule_cell": default_cam,
        "basket_cell": dict(up=dict(x=0,y=0,z=1),center=dict(x=0,y=0,z=0),eye=dict(x=1.9905432220621866,y=1.738269564893918,z=0.3723380871153242)),
        "stellate_cell": dict(up=dict(x=0,y=0,z=1),center=dict(x=0,y=0,z=0),eye=dict(x=1.720353554421655,y=1.8523332530375434,z=0.5250651742433416)),
    }
    for ct in network.get_cell_types():
        if ct.name == "glomerulus" or ct.name == "mossy_fibers":
            continue
        ids = ct.get_placement_set().identifiers
        istim = np.array([np.mean(stim_spikes[id]) / stim_p for id in ids])
        ispikes = np.array([[np.mean(inc_spikes[id].get(cstag, [0])) for cstag in ct_map[ct.name]] for id in ids])
        regressor = LinearRegression().fit(ispikes, istim)
        print("--- ", ct.name)
        print("score", regressor.score(ispikes, istim))
        print("coeff", dict(zip(ct_map[ct.name], regressor.coef_)))
        print("intercept", regressor.intercept_)
        dims = [0, 1]
        _ctmc = ct_map[ct.name]
        _is_spoofed = ispikes.shape[1] != 2

        if ct.name == "golgi_cell":
            dims = [0, 3]
            ct_map[ct.name] = [_ctmc[dims[0]], _ctmc[dims[1]]]
            ispikes = ispikes[:, dims]
            # print(ct_map[ct.name])
            # ct_map[ct.name][2]  #glomerulus_to_golgi
            # y_axis = ct_map[ct.name][3]  #golgi_to_golgi

        elif ct.name == "purkinje_cell":
            dims = [2,1]
            ct_map[ct.name] = [_ctmc[dims[0]], _ctmc[dims[1]]]
            ispikes = ispikes[:, dims]

        x_axis = ct_map[ct.name][0]
        x_axis = label_map.get(x_axis, x_axis)
        y_axis = ct_map[ct.name][1]
        y_axis = label_map.get(y_axis, y_axis)

        x = [0, max(ispikes[:, 0])]
        y = [0, max(ispikes[:, 1])]
        surface = np.linspace((0, 0), (max(ispikes[:, 0]), max(ispikes[:, 1])), 100)
        x, y = surface[:, 0], surface[:, 1]
        z_ = np.dstack(np.meshgrid(surface[:, 0], surface[:, 1]))
        if ct.name == "granule_cell":
            mask = ispikes[:, 0] > 4
            ispikes = ispikes[mask]
            istim = istim[mask]
            regressor = LinearRegression().fit(ispikes, istim)

        scatters = [
            go.Scatter3d(x=ispikes[:, 0], y=ispikes[:, 1], z=istim, mode="markers", marker_color=ct.plotting.color),
        ]
        if not _is_spoofed:
            scatters.append(
                go.Surface(
                    x=x,
                    y=y,
                    z=regressor.predict(z_.reshape(10000, 2)).reshape((100, 100)),
                    surfacecolor=np.zeros((100,100)),
                    colorscale=[
                        [0, "#ccc"],
                        [1, "rgb(0, 0, 0)"],
                    ],
                    opacity=0.4,
                    showscale=False,
                )
            )
        figs[ct.name] = go.Figure(
            scatters,
            layout_scene=dict(
                xaxis_title=x_axis,
                yaxis_title=y_axis,
                zaxis_title="Response [Hz]",
                camera=cameras.get(ct.name, default_cam),
            ),
        )

    return figs

def meta(key):
    if key == "granule_cell":
        return {"width": 1920 * 0.55, "height": 1080 * 0.55}
