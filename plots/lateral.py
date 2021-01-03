from bsb.core import from_hdf5
from h5py import File
from glob import glob
import os, numpy as np, itertools
import plotly.graph_objs as go
import grc_cloud

network_path = os.path.join(
    os.path.dirname(__file__), "..", "networks", "300x_200z.hdf5"
)
def results_path(*args):
    return os.path.join(
        os.path.dirname(__file__), "..", "results", *args
    )

def find_nearest(array, value):
    array = np.asarray(array)
    return (np.abs(array - value)).argmin()

def plot():
    network = from_hdf5(network_path)
    ps_pc = network.get_placement_set("purkinje_cell")
    ps_bc = network.get_placement_set("basket_cell")
    bc_pc = network.get_connectivity_set("basket_to_purkinje")
    bc_pos_map = dict(zip(ps_bc.identifiers, ps_bc.positions))
    pc_pos_map = dict(zip(ps_pc.identifiers, ps_pc.positions))
    cut_mask = (ps_pc.positions[:, 2] > 250) | (ps_pc.positions[:, 2] < 50)
    incl_mask = np.logical_not(cut_mask)
    cut_off_ids = ps_pc.identifiers[cut_mask]
    mli_files = glob(results_path("mli/lateral_MLI/*.hdf5"))
    no_mli_files = glob(results_path("mli/lateral_no_MLI/*.hdf5"))
    pcs_considered = len([id for id in ps_pc.identifiers if id not in cut_off_ids])
    deltas_mli = np.empty((pcs_considered, len(mli_files)))
    base_mli = np.empty((pcs_considered, len(mli_files)))
    deltas_no_mli = np.empty((pcs_considered, len(no_mli_files)))
    base_no_mli = np.empty((pcs_considered, len(no_mli_files)))

    base_start = 500
    base_end = 600
    stim_start = 700
    stim_end = 800

    # Get mean ISI during baseline and stimulus windows for each PC for each run
    for i, fname in enumerate(mli_files):
        with File(fname, "r") as f:
            base_mli[:, i] = np.array([np.mean(np.diff((times := f[f"/recorders/soma_spikes/{id}"][:, 1])[(times > 480) & (times < 600)])) for id in ps_pc.identifiers if id not in cut_off_ids])
            deltas_mli[:, i] = np.array([np.mean(np.diff((times := f[f"/recorders/soma_spikes/{id}"][:, 1])[(times > 676) & (times < 800)])) - np.mean(np.diff((times := f[f"/recorders/soma_spikes/{id}"][:, 1])[(times > 480) & (times < 600)])) for id in ps_pc.identifiers if id not in cut_off_ids])


    for i, fname in enumerate(no_mli_files):
        with File(fname, "r") as f2:
            base_no_mli[:, i] = np.array([np.mean(np.diff((times := f2[f"/recorders/soma_spikes/{id}"][:, 1])[(times > 480) & (times < 600)])) for id in ps_pc.identifiers if id not in cut_off_ids])
            deltas_no_mli[:, i] = np.array([np.mean(np.diff((times := f2[f"/recorders/soma_spikes/{id}"][:, 1])[(times > 676) & (times < 800)])) - np.mean(np.diff((times := f2[f"/recorders/soma_spikes/{id}"][:, 1])[(times > 480) & (times < 600)])) for id in ps_pc.identifiers if id not in cut_off_ids])

    # Average the values over the runs, use the opposite to get the reduction of ISI
    deltas_mli = -np.mean(deltas_mli, axis=1)
    deltas_no_mli = -np.mean(deltas_no_mli, axis=1)
    base_mli = np.mean(base_mli, axis=1)
    base_no_mli = np.mean(base_no_mli, axis=1)

    from sklearn.model_selection import GridSearchCV
    from sklearn.kernel_ridge import KernelRidge

    X = ps_pc.positions[incl_mask, 0].reshape(-1, 1)
    y = np.array(deltas_mli)
    y2 = np.array(deltas_no_mli)
    X_plot = np.linspace(0, 292, 100000)[:, None]
    kr = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1),
                      param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3],
                                  "gamma": np.logspace(-100, 20, 50)})
    kr.fit(X, y)
    inhib_score = kr.score(X, y)
    print("R^2 score inhibited: ", inhib_score)
    # Standard error of the estimate
    S = np.sqrt(np.sum((deltas_mli - kr.predict(X)) ** 2) / len(X))
    print("S score inhibited:", S)
    y_kr = kr.predict(X_plot)
    kr2 = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1),
                      param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3],
                                  "gamma": np.logspace(-100, 20, 200)})
    kr2.fit(X, y2)
    disinhib_score = kr.score(X, y2)
    print("R^2 score disinhibited: ", disinhib_score)
    # Standard error of the estimate
    S = np.sqrt(np.sum((deltas_no_mli - kr2.predict(X)) ** 2) / len(X))
    print("S score disinhibited:", S)
    y_kr2 = kr2.predict(X_plot)
    x = X.ravel()
    x_plot = X_plot.ravel()
    fig = go.Figure([
        go.Scatter(x=x, y=deltas_mli, name="PC response with MLI", mode="markers", marker=dict(color="blue")),
        go.Scatter(x=X_plot.ravel(), y=y_kr, name="KRR of response with MLI", mode="lines", line=dict(color="blue")),
        go.Scatter(x=x, y=deltas_no_mli, name="PC response without MLI", mode="markers", marker=dict(color="red")),
        go.Scatter(x=X_plot.ravel(), y=y_kr2, name="KRR of response without MLI", mode="lines", line=dict(color="red")),
        go.Scatter(x=[0], y=[0], mode="markers", name="Activated GrC", marker=dict(
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
    fig.layout.shapes = grc_cloud.granule_beam("networks/300x_200z.hdf5", mli_files[0], base_start=base_start, base_end=base_end, stim_start=stim_start, stim_end=stim_end)
    fig.update_layout(
        title_text="Lateral response of PC to activated GrC bundle",
        yaxis_title="Reduction in ISI (ms)",
        xaxis_title="X (µm)",
        xaxis_range=[300, 0],
    )
    fig.add_annotation(x=x_plot[10], y=y_kr[10], text=f"R\u00B2 = {round(inhib_score, 2)}", showarrow=True, arrowhead=False, ay=30, ax=-20, standoff=3)
    fig.add_annotation(x=x_plot[10], y=y_kr2[10], text=f"R\u00B2 = {round(disinhib_score, 2)}", showarrow=True, arrowhead=False, ay=-30, ax=-20, standoff=3)

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
    print("MLI Slope at half:", round(slope_m, 2))
    print("No MLI Slope at half:", round(slope_nm, 2))

    return fig
