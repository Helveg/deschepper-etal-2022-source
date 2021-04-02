from bsb.plotting import *
from ._paths import *
from glob import glob
import selection, h5py

def plot(net_path=None):
    if net_path is None:
        net_path = network_path(selection.network)
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from bsb.core import Scaffold, from_hdf5
    from bsb.config import JSONConfig
    from bsb.output import MorphologyRepository

    import numpy as np
    import plotly.graph_objects as go
    f = h5py.File(net_path,'r')

    scaffoldInstance = from_hdf5(net_path)
    config = scaffoldInstance.configuration
    key = 'golgi_to_granule'

    p = np.array(f['/cells/connections/' + key])
    pairs, counts = np.unique(p, return_counts=True, axis=0)
    distr = [sum(counts == i) for i in range(5)]

    return go.Figure(go.Bar(x=list(range(5)), y=distr))
