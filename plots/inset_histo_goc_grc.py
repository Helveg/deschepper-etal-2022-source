from bsb.plotting import *

def plot():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from bsb.core import Scaffold, from_hdf5
    from bsb.config import JSONConfig
    from bsb.output import MorphologyRepository

    import numpy as np
    import plotly.graph_objects as go
    import h5py

    import collections

    filename = 'networks/300x_200z.hdf5'
    f = h5py.File(filename,'r')

    scaffoldInstance = from_hdf5(filename)
    config = scaffoldInstance.configuration
    key = 'golgi_to_granule'

    p = np.array(f['/cells/connections/' + key])
    pairs, counts = np.unique(p, return_counts=True, axis=0)
    print(pairs.shape)
    print(pairs, counts)
    distr = [sum(counts == i) for i in range(5)]

    return go.Figure(go.Bar(x=list(range(5)), y=distr))
