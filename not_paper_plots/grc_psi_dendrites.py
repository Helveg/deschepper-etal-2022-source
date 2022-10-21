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
        id = int(path.split("_")[-1].split(".")[0])
        print("Analyzing net", id)
        with open(results_path("..", "pkl_ca", "calcium_data", f"calcium_{id}.pickle"), "rb") as f:
            result = pickle.load(f)["calcium"]["_data"]
        network = from_hdf5(path)
        MFs = selection.mf_batch_1[id]
        ps = network.get_placement_set("granule_cell")
        pos = ps.positions
        border = (pos[:, 0] > 290) | (pos[:, 0] < 10) | (pos[:, 2] > 190) | (pos[:, 2] < 10)
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
        carry_x = np.concatenate((carry_x, x))
        carry_y = np.concatenate((carry_y, y))


    x = carry_x
    y = carry_y
    fig = go.Figure([go.Violin(y=(z := y[x == d]), name=f"{d} active dendrites <br>n={len(z)}", span=[min(z), max(z)], spanmode="manual") for d in range(5)])
    fig.update_layout(xaxis_title="Granule cells", yaxis_title="Synaptic weight change")
    return fig
