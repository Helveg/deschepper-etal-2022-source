import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from bsb.core import from_hdf5

import numpy as np
from bsb.plotting import *
import scipy.spatial.distance as dist
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import h5py
from random import randrange, uniform
import plotly.express as px
from scipy import signal
import collections, selection
from collections import defaultdict
from ._paths import *

def plot():
    duration=6500 #ms
    timeRes=0.025  #ms
    cutoff=5500
    timeVect=np.linspace(cutoff, duration, int((duration-cutoff)/timeRes))
    displayTime = timeVect - cutoff

    pattern=[500.0, 504.0, 508.0, 514.0, 520.0]

    network = from_hdf5(network_path(selection.network))

    #IDs=[5987,  6002,  9272, 17285, 17372, 24861, 3764,  3851,  4075,3083,  3114,  3192,3074, 3076, 3109]
    id = selection.golgi_cells["High activity"]
    fig = go.Figure(
        layout=dict(
            xaxis_title="Time [ms]",
            yaxis_title="Membrane potential [mV]"
        )
    )

    with h5py.File(results_path("sensory_gabazine", "sensory_burst_gabazine_nogap.hdf5"), "a") as f:
        # Collect traces from cells across multiple recording groups.
        g = f[f"/recorders/soma_voltages/{id}"][()]
        fig.add_scatter(
            x=displayTime,
            y=g[int(timeVect[0]/timeRes):-1],
            name="Gabazine, no gap junctions",
            legendgroup="gabazine",
            mode='lines',
            line=dict(
                dash='solid',
                color='grey'
            ),
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
                )
            )
    with h5py.File(results_path("sensory_gabazine", "sensory_burst_control.hdf5"), "a") as f:
        # Collect traces from cells across multiple recording groups.
        g = f[f"/recorders/soma_voltages/{id}"][()]
        fig.add_scatter(
            x=displayTime,
            y=g[int(timeVect[0]/timeRes):-1],
            name="Control",
            legendgroup="control",
            mode='lines',
            line={'dash': 'solid','color': network.configuration.cell_types["golgi_cell"].plotting.color}
        )

    fig.update_layout(xaxis_range=[0, duration - cutoff], yaxis_range=[-70, 50])
    fig.update_layout(title_text="Influence of gabazine on granule cell responses")
    return fig
