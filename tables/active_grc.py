import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "plots"))

from _paths import *
import h5py, numpy as np
from scipy.stats import ttest_ind
from glob import glob

def crop(data, min, max, indices=False, transpose=False):
    if transpose:
        data = data[()].T
    c = data[:, 1]
    if indices:
        return np.where((c > min) & (c < max))[0]
    return c[(c > min) & (c < max)]

def table(control_path=None, gaba_path=None):
    if control_path is None:
        control_path = results_path("balanced_sensory", "*.hdf5")
    if gaba_path is None:
        gaba_path = results_path("balanced_sensory", "gabazine", "*.hdf5")
    table = [["control", "gabazine", "std_c", "std_g", "p"]]
    carry_control, carry_gaba = [], []
    for paths, carry, T in zip((control_path, gaba_path), (carry_control, carry_gaba), (False, True)):
        for path in glob(paths):
            with h5py.File(path, "r") as f:
                print("Analyzing", path)
                carry.append(sum(bool(len(crop(ds, 6000, 6040, transpose=T))) for ds in f["recorders/soma_spikes"].values() if ds.attrs["label"] == "granule_cell"))

    p = ttest_ind(carry_control, carry_gaba)[1]
    table.append([np.mean(x) for x in (carry_control, carry_gaba)] + [np.std(x) for x in (carry_control, carry_gaba)] + [p])


    return table
