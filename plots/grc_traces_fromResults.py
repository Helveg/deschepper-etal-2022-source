def plot():
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

    duration=6500 #ms
    timeRes=0.025  #ms
    cutoff=5500
    timeVect=np.linspace(cutoff, duration, int((duration-cutoff)/timeRes))

    pattern=[6000.0, 6004.0, 6008.0, 6014.0, 6020.0]

    filename = '/home/claudia/deschepper-etal-2020/networks/balanced.hdf5'
    f = h5py.File(filename,'r')

    #IDs=[5987,  6002,  9272, 17285, 17372, 24861, 3764,  3851,  4075,3083,  3114,  3192,3074, 3076, 3109]
    IDs=[ 6002,  3851,   3114,  3074]
    fig = make_subplots(rows=len(IDs), cols=1,shared_xaxes=True)

    with h5py.File("/home/claudia/deschepper-etal-2020/results/sensory_burst/sensory_burst_gabazine.hdf5", "a") as f:
        # Collect traces from cells across multiple recording groups.
        for n, g in f["/recorders/granules"].items():
            if g.attrs["cell_id"] in IDs:
                order = IDs.index(g.attrs["cell_id"])+1
                fig.add_scatter(x=timeVect, y=g[int(timeVect[0]/timeRes):-1], name=str(g.attrs["cell_id"]),mode='lines',line={'dash': 'solid','color': 'grey'},row=order, col=1)
                for j in range(0, len(pattern)):
                    fig.add_shape(type="line", x0=pattern[j], y0=min(g[int(timeVect[0]/timeRes):-1]), x1=pattern[j],
                        y1=max(g[int(timeVect[0]/timeRes):-1]),
                        line=dict(color="black", width=1, dash="dot"),row=order, col=1 )
    with h5py.File("/home/claudia/deschepper-etal-2020/results/sensory_burst/results_sensory_burst_16172284097089670601430045.hdf5", "a") as f:
        # Collect traces from cells across multiple recording groups.
        for n, g in f["/recorders/granules"].items():
            if g.attrs["cell_id"] in IDs:
                order = IDs.index(g.attrs["cell_id"])+1
                fig.add_scatter(x=timeVect, y=g[int(timeVect[0]/timeRes):-1], name=str(g.attrs["cell_id"]),mode='lines',line={'dash': 'solid','color': 'red'},row=order, col=1)

    fig.update_xaxes(range=[cutoff, duration])
    return fig