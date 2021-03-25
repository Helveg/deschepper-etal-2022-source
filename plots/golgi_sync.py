import os, sys
from bsb.core import from_hdf5
import numpy as np
import plotly.graph_objects as go
import h5py
from random import randrange, uniform
import plotly.express as px
from scipy import signal
import collections
from collections import defaultdict
from ._paths import *
import selection

def matrix_synchronization_index(matrix):
    rows, cols = matrix.shape[0], matrix.shape[1]
    print(rows, cols)
    syncIndex = np.ones(shape=(rows, rows))
    for i in range(rows):
        for j in range(i+1, rows):
            corr = signal.correlate(matrix[i], matrix[j])
            if np.max(corr) != 0:
                corr = corr / np.max(corr)
            syncIndex [i, j] = syncIndex [j, i] = corr[int(len(corr)/2)]
    return syncIndex

def smudge(s1, s2, smear=1):
    corr = 0
    for shift in range(-smear, 0):
        corr += np.sum(s1[:shift] * s2[-shift:])
    corr += np.sum(s1 * s2)
    for shift in range(1, smear + 1):
        corr += np.sum(s1[shift:] * s2[:-shift])
    return corr


def matrix_smudge_index(counts, smear=1):
    matrix = np.empty((len(counts), len(counts)))
    for i in range(matrix.shape[0]):
        for j in range(i, matrix.shape[1]):
            matrix[i, j] = matrix[j, i] = smudge(counts[i], counts[j], smear)
        # Normalize by autocorrelation, unless the cell didn't fire
        if matrix[i, i] != 0:
            matrix[:, i] /= matrix[i, i]
        else:
            # Set the diagonal value to 1, as this is required for a correct
            # and fast calculation of the means below where ones are expected
            # and counteracted on the diagonal.
            matrix[i, i] = 1
    # Calculate the mean for each row, ignoring the autocorrelation of 1 on the
    # diagonal.
    return (np.sum(matrix, axis=1) - 1) / (matrix.shape[1] - 1), matrix


def plot(path=None, net_path=None, bin_width=2, cutoff=4000, duration=8000):
    if path is None:
        raise ValueError("Missing path to result file.")
    if net_path is None:
        net_path = network_path(selection.network)
    network = from_hdf5(net_path)
    with h5py.File(path, "r") as file:
        spikes = dict()
        for g in file["/recorders/soma_spikes"].values():
            if g.attrs["label"] != "golgi_cell":
                continue
            spike_times = g[:, 1]
            g_sel = spike_times[(spike_times > cutoff) & (spike_times < duration)]
            # Get or create the cell's group
            group = spikes.setdefault(g.attrs["label"], list())
            # Add the cell's spikes to the group
            group.append(g_sel - cutoff)

    bins = np.arange(0, duration - cutoff, bin_width)
    n_bins = int(np.ceil((duration - cutoff) / bin_width))
    golgi_spikes = spikes.get("golgi_cell", list())
    print("n. golgi cells:", len(golgi_spikes))
    counts_goc = np.zeros((len(golgi_spikes), n_bins))

    for l, g in enumerate(golgi_spikes):
        binned_spikes = (g // bin_width).astype(int)
        bin_counts = np.bincount(binned_spikes, minlength=n_bins)
        assert len(bin_counts) == n_bins, "Incorrect binning"
        counts_goc[l] = bin_counts


    syncIndex = matrix_synchronization_index(counts_goc)
    smudge_vector, smudge_matrix = matrix_smudge_index(counts_goc, smear=1)
    meanSyncIndex=syncIndex[np.nonzero(syncIndex)].mean()
    print("lag0 mean:", np.mean(meanSyncIndex))
    print("smudge mean:", np.mean(smudge_vector))
