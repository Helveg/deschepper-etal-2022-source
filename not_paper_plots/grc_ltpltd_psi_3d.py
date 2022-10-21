from bsb.core import from_hdf5
import os, sys, h5py, numpy as np
sys.path.insert(0, os.path.join("..", "plots"))
import selection, plotly.graph_objs as go, scipy.stats
import pickle
from ._paths import *
from glob import glob
from bsb.config import get_result_config

def make_psi():
    # Use traced data from "A Nonlinear Cable Framework for Bidirectional Synaptic Plasticity"
    # Appearantly the formulas were in Supplementary, lol.
    c = np.array([[0, 428],[179.00,428.00],[192.41,428.44],[209.00,430.00],[226.50,434.50],[241.00,444.50],[253.00,456.00],[268.00,473.50],[282.00,491.50],[295.00,504.00],[304.50,510.00],[310.50,511.50],[315.50,511.50],[324.00,510.00],[330.50,505.00],[339.50,496.00],[346.50,484.50],[354.00,466.50],[360.00,447.50],[367.00,425.50],[384.00,344.50],[402.00,250.00],[417.50,188.00],[431.00,151.50],[444.00,133.50],[459.50,119.50],[471.50,114.50],[499.50,109.50],[529.00,108.00],[595.00,108.50]])
    # norm and milli to micro
    c /= 600
    c[:, 1] = (1 - c[:, 1]) * 1.4 - 0.4014
    c[:, 0] /= 1000
    f_intrp = scipy.interpolate.interp1d(c[:, 0], c[:, 1], kind="linear", bounds_error=False, fill_value=0.745)
    return f_intrp

psi = make_psi()

def plot(net_path=None, stim_start=6000, stim_end=6020):
    if net_path is None:
        net_path = network_path("batch_1", "*.hdf5")
    paths = glob(net_path)
    carry_x = np.empty((0,))
    carry_y = np.empty((0,))
    for path in paths:
        print(paths)
        id = int(path.split("_")[-1].split(".")[0])
        #print("Analyzing net", id)
        id=8
        with open(results_path("..", "pkl_ca", "calcium_data", f"calcium_{id}.pickle"), "rb") as f:
            result = pickle.load(f)["calcium"]["_data"]

        network = from_hdf5(path)
        #network=from_hdf5('/home/claudia/deschepper-etal-2020/networks/balanced.hdf5')
        MFs = selection.mf_batch_1[id]
        #MFs = selection.stimulated_mf_poiss
        ps = network.get_placement_set("granule_cell")
        pos = ps.positions
        #border = (pos[:, 0] > 290) | (pos[:, 0] < 10) | (pos[:, 2] > 190) | (pos[:, 2] < 10)
        border = (pos[:, 0] > 300) | (pos[:, 0] < 0) | (pos[:, 2] > 200) | (pos[:, 2] < 0)
        ids = ps.identifiers[~border]
        mf_glom = network.get_connectivity_set("mossy_to_glomerulus").get_dataset()
        glom_grc = network.get_connectivity_set("glomerulus_to_granule").get_dataset()
        active_glom = mf_glom[np.isin(mf_glom[:, 0], MFs), 1]
        active_dendrites = glom_grc[np.isin(glom_grc[:, 0], active_glom), 1]
        d = dict(zip(*np.unique(active_dendrites, return_counts=True)))
        grc_to_dend = np.vectorize(lambda x: d.get(x, 0))
        grc_to_calc = np.vectorize(result.get)
        x = grc_to_dend(ids)
        y = psi(grc_to_calc(ids))

        print(y)
        carry_x = np.concatenate((carry_x, x))
        carry_y = np.concatenate((carry_y, y))

        grc_plasticity_LTP = go.Scatter3d(
            name="GrCs LTP",
            x=pos[:, 0],
            y=pos[:, 2],
            z=pos[:, 1],
            #text=["ID: " + str(int(i)) + "\ndf: " + str(round(d, 2)) + "Hz" for i in ps.identifiers],
            mode="markers",
            marker=dict(
                colorscale="thermal",
                # cmin=min_c,
                # cmax=max_c,
                size=[20*i if i>0 else 0 for i in y],
                opacity=0.7,
                color=list(y),
                colorbar=dict(
                    len=0.8,
                    x=0.8,
                    title=dict(
                        text="LTD LTP (Ca-control theory)",
                        side="bottom"
                    )
                )
            )
        )
        grc_plasticity_LTD = go.Scatter3d(
            name="GrCs LTD",
            x=pos[:, 0],
            y=pos[:, 2],
            z=pos[:, 1],
            #text=["ID: " + str(int(i)) + "\ndf: " + str(round(d, 2)) + "Hz" for i in ps.identifiers],
            mode="markers",
            marker=dict(
                colorscale="thermal",
                # cmin=min_c,
                # cmax=max_c,
                size=[-80*i if i<0 else 0 for i in y],
                opacity=0.3,
                color=list(y),
                colorbar=dict(
                    len=0.8,
                    x=0.8,
                    title=dict(
                        text="LTD LTP (Ca-control theory)",
                        side="bottom"
                    )
                )
            )
        )

    x = carry_x
    y = carry_y
    fig = go.Figure(data=[grc_plasticity_LTP,grc_plasticity_LTD])
    fig.update_layout(xaxis_title="Granule cells", yaxis_title="Synaptic weight change")
    return fig
