from bsb.core import from_hdf5
import selection, h5py, numpy as np
from scipy import sparse
from plotly import graph_objs as go

def crop(data, min, max, indices=False):
    if len(data.shape) > 1:
        c = data[:, 1]
    else:
        c = data
    if indices:
        return np.where((c > min) & (c < max))[0]
    return c[(c > min) & (c < max)]

stimStart=1100
stimEnd=1150
PC_IDS = np.array([126,105])
pf_pc_sel=[], aa_pc_sel=[], sc_pc_sel=[], bc_pc_sel=[]

def plot():
    network = from_hdf5("networks/300x_200z.hdf5")

    pf_pc = network.get_connectivity_set("parallel_fiber_to_purkinje").get_dataset().astype(int)
    aa_pc = network.get_connectivity_set("ascending_axon_to_purkinje").get_dataset().astype(int)
    sc_pc = network.get_connectivity_set("stellate_to_purkinje").get_dataset().astype(int)
    bc_pc = network.get_connectivity_set("basket_to_purkinje").get_dataset().astype(int)

    for i,pc in enumerate(PC_IDS):
        #print(i, pc)
        a=pf_pc[pf_pc[:,1]==pc]
        pf_pc_sel[i,0].append(a)
        a=aa_pc[aa_pc[:,1]==pc]
        aa_pc_sel[i,0].append(a)
        a=sc_pc[sc_pc[:,1]==pc]
        sc_pc_sel[i,0].append(a)
        a=bc_pc[bc_pc[:,1]==pc]
        bc_pc_sel[i,0].append(a)


    with h5py.File("results/results_stim_on_MFs_4syncImp.hdf5", "r") as f:
        g = f["recorders/soma_spikes"]
        spikes_inc_sc = {id: [] for id in sc_ps.identifiers}
        for id, d in g.items():
            if d.attrs["label"] != "granule_cell":
                continue
            spikes = crop(d[()], stimStart, stimEnd)
            targets = sc_m.getrow(id).nonzero()[1]
            for t in targets:
                spikes_inc_sc[t].extend(spikes)
        spikes_inc_sc = {id: np.array(v) for id, v in spikes_inc_sc.items()}
        sc_spikes = np.array([np.bincount(np.ceil((v - stimStart) / 5).astype(int), minlength=21) for v in spikes_inc_sc.values()])
        fig = go.Figure()
        for sc in sc_spikes:
            fig.add_trace(go.Scatter(x=np.arange(stimStart, stimEnd, 5), y=sc, mode="lines", line=dict(color="blue"), opacity=0.1))
        figs["skyscrapers"] = fig
        fig = go.Figure()
        for sc in sc_spikes:
            fig.add_trace(go.Scatter(x=np.arange(stimStart, stimEnd, 5) + np.random.rand(20) * 5 - 2.5, y=sc, mode="markers", marker=dict(color="blue", size=2)))
        figs["manhattan"] = fig
        fig = go.Figure()
        for i, bin in enumerate(sc_spikes.T):
            print(i, len(bin))
            fig.add_trace(go.Violin(x0=stimStart + i * 5, y=bin, showlegend=False, fillcolor="aquamarine", line_color="black"))
        figs["violin"] = fig
        fig.show()
        spikes_per_impulse = np.array([[len(crop(v, stimStart + i * 25, stimStart + (i + 1) * 25)) for i in range(4)] for v in spikes_inc_sc.values()])
        for i in range(4):
            figs[f"impulse{i+1}"] = go.Figure(go.Histogram(x=spikes_per_impulse[:, i]))
    return figs
