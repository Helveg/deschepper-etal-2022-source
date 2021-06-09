import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from bsb.core import Scaffold
from bsb.config import JSONConfig
from bsb.output import MorphologyRepository

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
from ._paths import *

def plot():
    duration=6500 #ms
    timeRes=0.025  #ms
    cutoff=5500
    timeVect=np.linspace(cutoff, duration, int((duration-cutoff)/timeRes))
    displayTime = timeVect - cutoff

    pattern=[500.0, 504.0, 508.0, 514.0, 520.0]

    filename = network_path('balanced.hdf5')
    f = h5py.File(filename,'r')

    #IDs=[5987,  6002,  9272, 17285, 17372, 24861, 3764,  3851,  4075,3083,  3114,  3192,3074, 3076, 3109]
    IDs=[ 6002,  3851,   3114,  3074]
    fig = make_subplots(
        rows=len(IDs),
        cols=1,
        x_title="Time (ms)",
        y_title="Membrane potential (mV)",
        subplot_titles=tuple(f"{i} active dendrite" + "s" * (i > 1) for i in range(4, 0, -1)),
        shared_xaxes=True
    )

    # Share xaxes but show all ticks, see: https://github.com/plotly/plotly.js/issues/2539#issuecomment-522917132
    for i in range(len(IDs)):
        xaxis_name = 'xaxis' if i == 0 else f'xaxis{i + 1}'
        getattr(fig.layout, xaxis_name).showticklabels = True

    with h5py.File(results_path("sensory_gabazine", "sensory_burst_gabazine.hdf5"), "a") as f:
        # Collect traces from cells across multiple recording groups.
        for n, g in f["/recorders/granules"].items():
            if g.attrs["cell_id"] in IDs:
                order = IDs.index(g.attrs["cell_id"]) + 1
                fig.add_scatter(
                    x=displayTime,
                    y=g[int(timeVect[0]/timeRes):-1],
                    name="Gabazine",
                    showlegend=not (order - 1),
                    legendgroup="gabazine",
                    mode='lines',
                    line=dict(
                        dash='solid',
                        color='grey'
                    ),
                    row=order,
                    col=1
                )
                for j in range(0, len(pattern)):
                    fig.add_shape(
                        type="line",
                        x0=pattern[j],
                        x1=pattern[j],
                        y0=min(g[int(displayTime[0]/timeRes):-1]),
                        y1=max(g[int(displayTime[0]/timeRes):-1]),
                        line=dict(
                            color="black",
                            width=1,
                            dash="dot",
                        ),
                        row=order,
                        col=1,
                    )
    with h5py.File(results_path("sensory_gabazine", "sensory_burst_control.hdf5"), "a") as f:
        # Collect traces from cells across multiple recording groups.
        for n, g in f["/recorders/granules"].items():
            if g.attrs["cell_id"] in IDs:
                order = IDs.index(g.attrs["cell_id"])+1
                fig.add_scatter(
                    x=displayTime,
                    y=g[int(timeVect[0]/timeRes):-1],
                    name="Control",
                    showlegend=not (order - 1),
                    legendgroup="control",
                    mode='lines',
                    line={'dash': 'solid','color': 'red'},
                    row=order,
                    col=1
                )

    fig.update_xaxes(range=[0, duration - cutoff])
    fig.update_yaxes(range=[-70, 50])
    fig.update_layout(title_text="Influence of gabazine on granule cell responses")
    return fig
