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
        paths = glob(results_path("sensory_burst", "*"))
    else:
        paths = glob(path)
    if net_path is None:
        net_path = network_path(selection.network)
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
            print("Processing", i, len(paths))
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
                    if id in (213, 214, 222, 223):
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
    tbl = [[""]]
    for ct in network.get_cell_types():
        if ct.name == "glomerulus" or ct.name == "mossy_fibers":
            continue
        ids = ct.get_placement_set().identifiers
        istim = np.array([np.mean(stim_spikes[id]) / stim_p for id in ids])
        ispikes = np.array([[np.mean(inc_spikes[id].get(cstag, [0])) for cstag in ct_map[ct.name]] for id in ids])
        # # Not averaged
        # istim = np.fromiter(itertools.chain(*(stim_spikes[id] for id in ids)), dtype=float) / stim_p
        # ispikes = np.zeros((len(ids) * len(paths), len(ct_map[ct.name])))
        # print("tot", ispikes.shape)
        # for i, id in enumerate(ids):
        #     block = [inc_spikes[id].get(cstag, [0] * len(paths)) for cstag in ct_map[ct.name]]
        #     ispikes[i * len(paths) : (i + 1) * len(paths)] = np.array(block).T
        regressor = LinearRegression().fit(ispikes, istim)
        nonlinear = RandomForestRegressor(max_depth=4).fit(ispikes, istim)
        print("--- ", ct.name)
        print("score", regressor.score(ispikes, istim))
        print("nonlin. score", nonlinear.score(ispikes, istim))
        print("coeff", dict(zip(ct_map[ct.name], regressor.coef_)))
        print("intercept", regressor.intercept_)
        #     figs[cstag] = go.Figure([go.Scatter(x=ispikes, y=istim, mode="markers"), go.Scatter(x=ispikes, y=istim_corr, mode="markers", name=f"corrected for {', '.join(cnames)}")], layout_title_text=f"{ct.name} {cstag}")
        # figs[ct.name] = fig = go.Figure()
        # for i, cstag in enumerate(ct_map[ct.name]):
        #     reduced = np.zeros((10000, ispikes.shape[1]))
        #     reduced[:, i] = np.linspace(0, max(ispikes[:, i]), 10000)
        #     norm = np.zeros((2, ispikes.shape[1]))
        #     norm[1, i] = 1
        #     # fig.add_trace(go.Scatter(x=[0, 1], y=regressor.predict(norm), mode="lines", name=f"{cstag} R^2={regressor.score(ispikes, istim)}"))
        #     figs[cstag] = go.Figure(
        #         [
        #             go.Scatter(x=ispikes[:, i], y=istim, mode="markers", name=f"data projection"),
        #             go.Scatter(x=reduced[:, i], y=regressor.predict(reduced), mode="lines", name=f"linear prediction {regressor.score(ispikes, istim)}"),
        #             go.Scatter(x=reduced[:, i], y=nonlinear.predict(reduced), mode="lines", name=f"nonlinear prediction {nonlinear.score(ispikes, istim)}"),
        #         ],
        #         layout_title_text=f"{ct.name} {cstag}",
        #     )
        if len(ct_map[ct.name]) == 2:
            x_axis = ct_map[ct.name][0]
            y_axis = ct_map[ct.name][1]
            x = [0, max(ispikes[:, 0])]
            y = [0, max(ispikes[:, 1])]
            surface = np.linspace((0, 0), (max(ispikes[:, 0]), max(ispikes[:, 1])), 100)
            x, y = surface[:, 0], surface[:, 1]
            z_ = np.dstack(np.meshgrid(surface[:, 0], surface[:, 1]))
            os = z_.shape
            if ct.name == "granule_cell":
                mask = ispikes[:, 0] > 4
                ispikes_masked = ispikes[mask]
                istim_masked = istim[mask]
                improved = LinearRegression().fit(ispikes_masked, istim_masked)
                figs[ct.name] = go.Figure(
                    [
                        go.Surface(x=x, y=y, z=regressor.predict(z_.reshape(10000, 2)).reshape((100, 100)), surfacecolor=np.zeros((100,100)), opacity=0.4, colorscale=[[0, "orange"], [1, "blue"]]),
                        go.Surface(x=x, y=y, z=improved.predict(z_.reshape(10000, 2)).reshape((100, 100)), surfacecolor=np.zeros((100,100)), opacity=0.4, colorscale=[[0, "purple"], [1, "blue"]]),
                        go.Scatter3d(x=ispikes[:, 0], y=ispikes[:, 1], z=istim, mode="markers", marker=dict(size=1)),
                        go.Scatter3d(x=ispikes_masked[:, 0], y=ispikes_masked[:, 1], z=istim_masked, mode="markers", marker=dict(size=1)),
                    ],
                    layout=dict(
                        scene=dict(xaxis_title=x_axis, yaxis_title=y_axis),
                        coloraxis_showscale=False,
                    ),
                )
            else:
                figs[ct.name] = go.Figure(
                    [
                        go.Surface(x=x, y=y, z=regressor.predict(z_.reshape(10000, 2)).reshape((100, 100)), surfacecolor=np.zeros((100,100)), opacity=0.4),
                        go.Scatter3d(x=ispikes[:, 0], y=ispikes[:, 1], z=istim, mode="markers", marker_color="blue"),
                    ],
                    layout_scene=dict(xaxis_title=x_axis, yaxis_title=y_axis)
                )

    return figs
