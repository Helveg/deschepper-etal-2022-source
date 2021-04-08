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

import collections
from collections import defaultdict

import shelve

filename = '/home/claudia/deschepper-etal-2020/networks/balanced.hdf5'
f = h5py.File(filename,'r')

duration=8000
cutoff=4000  #ms
with h5py.File("/home/claudia/deschepper-etal-2020/results/results_20HzAllMfs_from4to8sec.hdf5", "a") as f:
    #order=dict(mossy_fiber=0, granule_cell=1, golgi_cell=2, purkinje_cell=3, stellate_cell=4, basket_cell=5)
    freq_map = {"mossy_fibers": [], "granule_cell": [], "golgi_cell": [], "purkinje_cell": [], "stellate_cell": [], "basket_cell":[]}
    #freq_map = {"granule_cell": [], "golgi_cell": [], "purkinje_cell": [], "stellate_cell": [], "basket_cell":[]}

    for g in f["/all"].values():
        #print(g[:,1])
    #     isi = np.diff(np.array(g[g[:,1]>cutoff,1]))
    #     if len(isi)==0:
    #         mean_freq=0.
    #         cv=0.
    #     else:
    #         mean_isi=np.mean(isi)
    #         mean_freq=1/mean_isi*1000
    #         cv=np.std(isi)/mean_isi
    #     #print(g.attrs["label"], mean_isi, cv)
    #     del isi
    #     freq_map[g.attrs["label"]].append([mean_freq, cv]) # Hz, cv:[0-1]
    # print(freq_map["mossy_fibers"])
    # print(freq_map["golgi_cell"])

        NumbSpikes = len(np.array(g[np.logical_and  (g[:,1]>cutoff, g[:,1]<duration),1]))
        mean_freq= NumbSpikes / (duration-cutoff)*1000
        freq_map[g.attrs["label"]].append(mean_freq) # Hz

for l,k in freq_map.items():
    y=[]
    #y1=[]  #cv
    for v in freq_map[l]:
        #y.append(v[0])
        #y1.append(v[1])
        y.append(v)
    m= np.mean(np.array(y))
    std=np.std(np.array(y))
    # m1= np.mean(np.array(y1))
    # std1=np.std(np.array(y1))
    #print(l + " freq(Hz)= ", m, "+-", std, "; CV= ", m1, "+-", std1)
    print(l + " freq(Hz)= ", m, "+-", std)



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
        # row.extend(adjusted, stack=g.attrs.get("stack", None), run=run_id)
