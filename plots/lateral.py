from bsb.core import from_hdf5
from h5py import File
from glob import glob
import os, numpy as np, itertools
import plotly.graph_objs as go
import grc_cloud, pickle
from ._paths import *
import selection

frozen = True

def find_nearest(array, value):
    array = np.asarray(array)
    return (np.abs(array - value)).argmin()

def plot(run_mli_path=None, run_nomli_path=None, net_path=None):
    if run_mli_path is None:
        run_mli_path = results_path("lateral", "mli")
    if run_nomli_path is None:
        run_nomli_path = results_path("lateral", "no_mli")
    if net_path is None:
        net_path = network_path(selection.network)
    network = from_hdf5(net_path)
    ps_pc = network.get_placement_set("purkinje_cell")
    ps_bc = network.get_placement_set("basket_cell")
    bc_pc = network.get_connectivity_set("basket_to_purkinje")
    bc_pos_map = dict(zip(ps_bc.identifiers, ps_bc.positions))
    pc_pos_map = dict(zip(ps_pc.identifiers, ps_pc.positions))
    cut_mask = (ps_pc.positions[:, 2] > 250) | (ps_pc.positions[:, 2] < 50)
    incl_mask = np.logical_not(cut_mask)
    cut_off_ids = ps_pc.identifiers[cut_mask]
    mli_files = glob(run_mli_path + "/*.hdf5")
    no_mli_files = glob(run_nomli_path + "/*.hdf5")
    pcs_considered = len([id for id in ps_pc.identifiers if id not in cut_off_ids])
    stim_mli = np.empty((pcs_considered, len(mli_files)))
    base_mli = np.empty((pcs_considered, len(mli_files)))
    stim_no_mli = np.empty((pcs_considered, len(no_mli_files)))
    base_no_mli = np.empty((pcs_considered, len(no_mli_files)))

    base_start = 5700
    base_end = 5900
    stim_start = 6000
    stim_end = 6100

    # Get mean ISI during baseline and stimulus windows for each PC for each run
    for i, fname in enumerate(mli_files):
        with File(fname, "r") as f:
            base_mli[:, i] = np.array([np.mean(np.diff((times := f[f"/recorders/soma_spikes/{id}"][:, 1])[(times > base_start) & (times < base_end)])) for id in ps_pc.identifiers if id not in cut_off_ids])
            # print("---", fname)
            # print(*((times := f[f"/recorders/soma_spikes/{id}"][:, 1]) for id in ps_pc.identifiers if id not in cut_off_ids))
            # print(base_mli[:, i])
            stim_mli[:, i] = np.array([np.mean(np.diff((times := f[f"/recorders/soma_spikes/{id}"][:, 1])[(times > stim_start) & (times < stim_end)])) for id in ps_pc.identifiers if id not in cut_off_ids])
            # print(stim_mli[:, i])

    for i, fname in enumerate(no_mli_files):
        with File(fname, "r") as f2:
            base_no_mli[:, i] = np.array([np.mean(np.diff((times := f2[f"/recorders/soma_spikes/{id}"][:, 1])[(times > base_start) & (times < base_end)])) for id in ps_pc.identifiers if id not in cut_off_ids])
            stim_no_mli[:, i] = np.array([np.mean(np.diff((times := f2[f"/recorders/soma_spikes/{id}"][:, 1])[(times > stim_start) & (times < stim_end)])) for id in ps_pc.identifiers if id not in cut_off_ids])
    # ISIs
    stim_mli = np.mean(stim_mli, axis=1)
    stim_no_mli = np.mean(stim_no_mli, axis=1)
    base_mli = np.mean(base_mli, axis=1)
    base_no_mli = np.mean(base_no_mli, axis=1)
    # Frequencies
    sf_mli = 1000 / stim_mli
    sf_no_mli = 1000 / stim_no_mli
    bf_mli = 1000 / base_mli
    bf_no_mli = 1000 / base_no_mli

    # Plot the reduction of ISI (+ on y axis = + firing rate)
    # -ΔISI = -(stim - base) = base - stim
    deltas_mli = base_mli - stim_mli
    deltas_no_mli = base_no_mli - stim_no_mli
    # For frequencies this +/+ relation is already the case so:
    # Δf = stim - base
    df_mli = sf_mli - bf_mli
    df_no_mli = sf_no_mli - bf_no_mli

    rISI_mli = deltas_mli / base_mli
    rISI_no_mli = deltas_no_mli / base_no_mli
    rf_mli = df_mli / bf_mli
    rf_no_mli = df_no_mli / bf_no_mli

    figs = {}

    # 4 options were compared and gave similar results. In 3 cases the predicted KRR
    # lines crossed, so the no MLI line had higher highest activity and lower lowest act.
    # In 1 case, namely the delta of the frequencies this was not observed.
    # Because both relative metrics agreed the non relative metrics were discarded and
    # only 1 was shown: the relative delta of ISIs:
    for name, cond1, cond2 in (
        # ("isi", deltas_mli, deltas_no_mli),
        # ("freq", df_mli, df_no_mli),
        ("rel_isi", rISI_mli, rISI_no_mli),
        # ("rel_freq", rf_mli, rf_no_mli),
    ):
        if not frozen:
            print("------------", name, "------------")
            from sklearn.model_selection import GridSearchCV
            from sklearn.kernel_ridge import KernelRidge

            X = ps_pc.positions[incl_mask, 0].reshape(-1, 1)
            y = np.array(cond1)
            y2 = np.array(cond2)
            X_plot = np.linspace(np.min(X), np.max(X), 100000)[:, None]
            kr = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1),
                              param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3],
                                          "gamma": np.logspace(-100, 20, 200)})
            kr.fit(X, y)
            inhib_score = kr.score(X, y)
            print("R^2 score inhibited: ", inhib_score)
            # Standard error of the estimate
            S = np.sqrt(np.sum((cond1 - kr.predict(X)) ** 2) / len(X))
            print("S score inhibited:", S)
            y_kr = kr.predict(X_plot)
            kr2 = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1),
                              param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3],
                                          "gamma": np.logspace(-100, 20, 200)})
            kr2.fit(X, y2)
            disinhib_score = kr.score(X, y2)
            print("R^2 score disinhibited: ", disinhib_score)
            # Standard error of the estimate
            S = np.sqrt(np.sum((cond2 - kr2.predict(X)) ** 2) / len(X))
            print("S score disinhibited:", S)
            y_kr2 = kr2.predict(X_plot)
            x = X.ravel()
            x_plot = X_plot.ravel()
            mi = np.argmax(y_kr)
            nmi = np.argmax(y_kr2)
            m = y_kr[mi]
            nm = y_kr2[nmi]
            print("Max red. with MLI:", round(m, 2), "at", round(x_plot[mi], 2))
            print("Max red. without MLI:", round(nm, 2), "at", round(x_plot[nmi], 2))
            hmi = find_nearest(y_kr, m / 2)
            hnmi = find_nearest(y_kr2, nm / 2)
            slope_m = np.diff(y_kr)[hmi] / np.diff(x_plot)[hmi]
            slope_nm = np.diff(y_kr2)[hnmi] / np.diff(x_plot)[hnmi]
            print("MLI Slope at half:", round(slope_m, 5))
            print("No MLI Slope at half:", round(slope_nm, 5))
            beam = grc_cloud.granule_beam(net_path, mli_files[0], base_start=base_start, base_end=base_end, stim_start=stim_start, stim_end=stim_end)
            with open(f"lateral_{name}.pkl", "wb") as f:
                pickle.dump((x, cond1, x_plot, y_kr, cond2, y_kr2, beam), f)
        else:
            with open(f"lateral_{name}.pkl", "rb") as f:
                x, cond1, x_plot, y_kr, cond2, y_kr2, beam = pickle.load(f)
        fig = go.Figure([
            go.Scatter(x=x, y=cond1, name="Control", mode="markers", marker_size=3, text=[str(id) for id in ps_pc.identifiers if id not in cut_off_ids], marker=dict(color=ps_pc.type.plotting.color)),
            go.Scatter(x=x_plot, y=y_kr, name="Control fit", showlegend=False, mode="lines", line=dict(color=ps_pc.type.plotting.color, width=1)),
            go.Scatter(x=x, y=cond2, name="No MLI-PC", mode="markers", marker_size=3,  marker=dict(color="grey")),
            go.Scatter(x=x_plot, y=y_kr2, name="No MLI-PC fit", showlegend=False, mode="lines", line=dict(color="grey", width=1)),
            go.Scatter(x=[0], y=[0], mode="markers", name="GrC activation", marker=dict(
                colorscale=grc_cloud.colorbar_grc, cmin=0, cmax=1,
                size=9,
                symbol="square",
                color=[1],
                opacity=0,
                colorbar=dict(
                    len=0.6,
                    xanchor="left",
                    title=dict(
                        text="Activity density",
                        side="bottom"
                    )
                )
            ))
        ])
        fig.layout.shapes = beam
        fig.update_layout(
            title_text=f"Lateral response of PC to activated GrC bundle",
            yaxis_title="Relative elongation of ISI",
            xaxis_title="Distance (µm)",
            xaxis_range=[300, 0],
            xaxis_tickmode="array",
            xaxis_tickvals=[i * 50 for i in range(7)],
            xaxis_ticktext=[300 - i * 50 for i in range(7)]
        )
        # fig.add_annotation(x=x_plot[10], y=y_kr[10], text=f"R\u00B2 = {round(inhib_score, 2)}", showarrow=True, arrowhead=False, ay=30, ax=-20, standoff=3)
        # fig.add_annotation(x=x_plot[10], y=y_kr2[10], text=f"R\u00B2 = {round(disinhib_score, 2)}", showarrow=True, arrowhead=False, ay=-30, ax=-20, standoff=3)

        figs[name] = fig

    return figs

def meta(key):
    return {"width": 550, "height": 400}
