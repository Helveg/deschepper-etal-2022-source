import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from bsb.core import Scaffold
from bsb.config import JSONConfig
from bsb.output import MorphologyRepository

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from bsb.plotting import *
import scipy.spatial.distance as dist
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import h5py
from random import randrange, uniform
import plotly.express as px
from scipy import signal
import collections
from collections import defaultdict

import shelve

filename = '/home/claudia/deschepper-etal-2020/networks/300x_200z.hdf5'
f = h5py.File(filename,'r')


cutoff=1000
duration=4000
with h5py.File("/home/claudia/deschepper-etal-2020/results/resultsPart1.hdf5", "a") as f:
    #order=dict(mossy_fiber=0, granule_cell=1, golgi_cell=2, purkinje_cell=3, stellate_cell=4, basket_cell=5)
    #freq_map = {"mossy_fiber": [], "granule_cell": [], "golgi_cell": [], "purkinje_cell": [], "stellate_cell": [], "basket_cell":[]}
    spikes = {"mossy_fibers": [], "granule_cell": [], "golgi_cell": [], "purkinje_cell": [], "stellate_cell": [], "basket_cell":[]}

    for g in f["/all"].values():
        #spikes[g.attrs["label"]].append(np.array(g[g[:,1]>cutoff,1])-cutoff)
        g1=np.array(g)
        gSel= g1[np.logical_and(g1[:,1]>cutoff, g1[:,1]<duration),1]
        spikes[g.attrs["label"]].append(gSel-cutoff)

binWidth=5
fig=go.Figure()
for l,k in spikes.items():
    v=spikes[l]
    counts, bins = np.histogram(np.concatenate(v), bins=np.arange(0, duration-cutoff, binWidth))
    fig.add_trace(go.Scatter(y=counts, name=l)) #, marker=dict(color=color["AMPA"]),showlegend=True))
fig.show()
counts_grc, bins = np.histogram(np.concatenate(spikes["granule_cell"]), bins=np.arange(0, duration-cutoff, binWidth))
counts_goc, bins = np.histogram(np.concatenate(spikes["golgi_cell"]), bins=np.arange(0, duration-cutoff, binWidth))
bincenters = np.mean(np.vstack([bins[0:-1],bins[1:]]), axis=0)

def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result #[result.size/2:]

def crosscorr(x,y):
    result = np.correlate(x, y, mode='full')
    return result #[result.size/2:]

# autocorr_grc=autocorr(counts_grc)
# autocorr_goc=autocorr(counts_goc)
#
#crosscorr_grcgoc=crosscorr(counts_grc, counts_goc)
#
# lags = np.arange(-bincenters[-1],bincenters[-1],binWidth) #adding 0.1 to include the last instant of time also
# plt.figure()
# plt.plot(lags,autocorr_goc)
# plt.xlabel('Lag')
# plt.ylabel('crosscorrelation')
# plt.show()

#
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import ccf
from statsmodels.graphics import tsaplots
from scipy import signal

# Display the autocorrelation plot of your time series
lag_acf = acf(counts_goc, nlags=int(counts_goc.size/2), fft=True)
plt.plot(lag_acf)
plt.title('Autocorrelation Function')
plt.show()
fig = tsaplots.plot_acf(counts_goc, lags=int(counts_goc.size/2))
plt.show()

# Display the crosscorrelation plot of your time series
# corr = signal.correlate(counts_goc, counts_grc, mode="full")
# lags = signal.correlation_lags(len(counts_goc), len(counts_grc),mode="full")
# corr /= np.max(corr)
# plt.figure()
# plt.plot(lags, corr)
# plt.show()
# lag_ccf = ccf(counts_goc, counts_grc)
# plt.plot(lag_ccf)
# plt.title('crosscorr Function')
# plt.show()
# fig = tsaplots.plot_ccf(counts_goc, counts_grc, lags=int(counts_goc.size/2))
# plt.show()


# def cross_corr(y1, y2):
#   """Calculates the cross correlation and lags without normalization.
#
#   The definition of the discrete cross-correlation is in:
#   https://www.mathworks.com/help/matlab/ref/xcorr.html
#
#   Args:
#     y1, y2: Should have the same length.
#
#   Returns:
#     max_corr: Maximum correlation without normalization.
#     lag: The lag in terms of the index.
#   """
#   if len(y1) != len(y2):
#     raise ValueError('The lengths of the inputs should be the same.')
#
#   y1_auto_corr = np.dot(y1, y1) / len(y1)
#   y2_auto_corr = np.dot(y2, y2) / len(y1)
#   corr = np.correlate(y1, y2, mode='same')
#   # The unbiased sample size is N - lag.
#   unbiased_sample_size = np.correlate(
#       np.ones(len(y1)), np.ones(len(y1)), mode='same')
#   corr = corr / unbiased_sample_size / np.sqrt(y1_auto_corr * y2_auto_corr)
#   shift = len(y1) // 2
#
#   max_corr = np.max(corr)
#   argmax_corr = np.argmax(corr)
#   return max_corr, argmax_corr - shift


# def autocorr(x):
#     n = x.size
#     norm = (x - np.mean(x))
#     result = np.correlate(norm, norm, mode='same')
#     acorr = result[n//2 + 1:] / (x.var() * np.arange(n-1, n//2, -1))
#     lag = np.abs(acorr).argmax() + 1
#     print(n)
#     r = acorr[lag-1]
#     if np.abs(r) > 0.5:
#       print('Appears to be autocorrelated with r = {}, lag = {}'. format(r, lag))
#     else:
#       print('Appears to be not autocorrelated')
#     return r, lag
#
#
# r, lag= autocorr(counts_goc)



# fig, [ax1, ax2] = plt.subplots(2, 1, sharex=True)
# ax1.xcorr(signal.detrend(counts_goc), signal.detrend(counts_grc), usevlines=True, normed=True)
# ax1.grid(True)
# ax2.acorr(signal.detrend(counts_goc), usevlines=True, normed=True)
# ax2.grid(True)
# plt.show()

# plt.acorr(counts_grc)
# plt.show()
# plt.acorr(counts_goc)
# plt.show()
    #print("mean freq goc: ", np.mean(freq_map["golgi_cell"][:][0]))
        #
        #
        #
        # l = g.attrs.get("label", "unlabelled")
        # if l not in row_map:
        #     color = g.attrs.get("color", None)
        #     order =  order.get(g.attrs["label"], 0)
        #     row_map[l] = row = PSTHRow(l, color, order=order)
        # else:
        #     row = row_map[l]
        # run_id = g.attrs.get("run_id", 0)
        # adjusted = g[()]
        # adjusted[:, 1] = adjusted[:, 1] - cutoff
