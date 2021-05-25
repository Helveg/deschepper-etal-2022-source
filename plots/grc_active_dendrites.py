import os, sys
import numpy as np
from bsb.plotting import *
from bsb.core import from_hdf5
import plotly.graph_objects as go
import h5py, selection, _layouts
from ._paths import *
from glob import glob
import selection

import collections
from collections import defaultdict

def plot(net_path=None,batch_id=None, lookup_mf=False):
    if net_path is None:
        net_path = network_path(selection.network)
    f = h5py.File(net_path,'r')
    scaffoldInstance = from_hdf5(net_path)

    if not lookup_mf:
        sim = next(iter(scaffoldInstance.configuration.simulations.values()))
        excitedMFs = sim.devices["mossy_fiber_sensory_burst"].targets
    elif batch_id is None:
        excitedMFs = selection.stimulated_mf_poiss
    else:
        excitedMFs = selection.mf_batch_1[int(batch_id)]

    connMfGlom = np.array(f['/cells/connections/mossy_to_glomerulus'])
    excitedGlom=[]
    for i in range(0,len(excitedMFs)):
        excitedGlom.append(connMfGlom[connMfGlom[:,0]==int(excitedMFs[i]),1])

    excitedGlomA = np.concatenate( (np.array(excitedGlom)[:]))
    connGlomGrC = np.array(f['/cells/connections/glomerulus_to_granule'])
    excitedGrC=[]
    for i in range(0,len(excitedGlomA)):
        excitedGrC.append(connGlomGrC[connGlomGrC[:,0]==int(excitedGlomA[i]),1])

    excitedGrCA = np.concatenate( (np.array(excitedGrC)[:]))
    grIDs= scaffoldInstance.get_placement_set("granule_cell").identifiers
    NOexcitedGrC = grIDs[np.logical_not(np.isin(grIDs, excitedGrCA))]

    counterExcGrC=collections.Counter(excitedGrCA)
    counterSynInGrC=collections.Counter(counterExcGrC.values())

    pos=np.array(f['/cells/positions'])
    InGrC = defaultdict(list)
    for key, val in sorted(counterExcGrC.items()):
        InGrC[val].append(key)

    pos_coll = []
    pos0=np.zeros((len(NOexcitedGrC),3))
    pos1=np.zeros((len(InGrC[1]),3))
    pos2=np.zeros((len(InGrC[2]),3))
    pos3=np.zeros((len(InGrC[3]),3))
    pos4=np.zeros((len(InGrC[4]),3))
    for i in range(0,len(NOexcitedGrC)):
        pos0[i,:]= pos[pos[:,0]==int(NOexcitedGrC[i]),2:5][0]
    for i in range(0,len(InGrC[1])):
        pos1[i,:]= pos[pos[:,0]==int(InGrC[1][i]),2:5][0]
    for i in range(0,len(InGrC[2])):
        pos2[i,:]= pos[pos[:,0]==int(InGrC[2][i]),2:5][0]
    for i in range(0,len(InGrC[3])):
        pos3[i,:]= pos[pos[:,0]==int(InGrC[3][i]),2:5][0]
    for i in range(0,len(InGrC[4])):
        pos4[i,:]= pos[pos[:,0]==int(InGrC[4][i]),2:5][0]

    In0= go.Scatter3d(
        x= pos0[:,0],
        y= pos0[:,2],
        z= pos0[:,1],
        name='0 input',
        mode='markers',
        marker=dict(
            size=1,
            color="red"
        ),
        opacity=0.1
    )

    In1= go.Scatter3d(
        x= pos1[:,0],
        y= pos1[:,2],
        z= pos1[:,1],
        name='1 input',
        mode='markers',
        marker=dict(
            size=2,
            color="red"
        ),
        opacity=0.1
    )
    In2= go.Scatter3d(
        x= pos2[:,0],
        y= pos2[:,2],
        z= pos2[:,1],
        name='2 inputs',
        mode='markers',
        marker=dict(
            size=3,
            color="red"
        ),
        opacity=0.3
    )
    In3= go.Scatter3d(
        x= pos3[:,0],
        y= pos3[:,2],
        z= pos3[:,1],
        name='3 inputs',
        mode='markers',
        marker=dict(
            size=4,
            color="red"
        ),
        opacity=0.5
    )
    In4= go.Scatter3d(
        x= pos4[:,0],
        y= pos4[:,2],
        z= pos4[:,1],
        name='4 inputs',
        mode='markers',
        marker=dict(
            size=5,
            color="red",
            line=dict(width=2, color='darkred')
        )
    )

    fig = go.Figure(data=[In0, In1, In2, In3, In4])

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="X", range=[0, 300], autorange=False),
            yaxis=dict(title="Z", range=[0, 200], autorange=False),
            zaxis=dict(title="Y", range=[0, 130], autorange=False),
            aspectratio=dict(x=1, y=2/3, z=13/30 ),
            camera=_layouts.struct_activation_cam
        )
    )
    return fig
