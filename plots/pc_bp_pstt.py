from bsb.core import from_hdf5
from bsb.config import get_result_config
from bsb.plotting import plot_morphology, hdf5_gather_voltage_traces, plot_traces
from scipy import sparse
from plotly import graph_objs as go
import selection, numpy as np, h5py
from colour import Color
from ._paths import *

def crop(data, min, max, indices=False):
    if len(data.shape) > 1:
        c = data[:, 1]
    else:
        c = data
    if indices:
        return np.where((c > min) & (c < max))[0]
    return c[(c > min) & (c < max)]

def get_spike_zero(spikes, stimulus):
    return np.where(spikes > stimulus)[0][0]

def plot(path=None, net_path=None, input_device="mossy_fiber_sensory_burst", buffer=200, cutoff=5000, bin_width=5):
    if path is None:
        raise ValueError("Give a path")
    if net_path is None:
        net_path = network_path(selection.network)
    network = from_hdf5(net_path)
    config = get_result_config(path)
    with h5py.File(path, "r") as f:
        ps_pc = network.get_placement_set("purkinje_cell")
        traces = hdf5_gather_voltage_traces(f, "recorders/soma_voltages/", map(str, map(int, ps_pc.identifiers)))
        pc_act = {id: f[f"/recorders/soma_spikes/{int(id)}"][:, 1] for id in ps_pc.identifiers}
        traces.set_legends(["Membrane potential"])
        traces.set_colors([config.cell_types["purkinje_cell"].plotting.color])
        device = next(iter(config.simulations.values())).devices[input_device]
        if hasattr(device, "spike_times"):
            start = min(map(float, device.spike_times))
            stop = max(map(float, device.spike_times))
        else:
            params = device.parameters
            start = float(params["start"])
            stop = start + (float(params["interval"]) * float(params["number"]))
        input_region = [start, stop]
        t = f["time"][()]
        tmask = (t > start - buffer) & (t < stop + 800 + buffer)
        t_masked = t[tmask]
        fig = go.Figure()
        b_factors = {id: 1 for id in ps_pc.identifiers}
        p_factors = {id: 1 for id in ps_pc.identifiers}
        bp_factors = {id: 1 for id in ps_pc.identifiers}
        windows = {id: stop + 50 for id in ps_pc.identifiers}
        for cell_trace in traces:
            id = cell_trace.cell_id
            for trace in cell_trace:
                spikes = pc_act[int(id)]
                breakpoint = np.mean(np.diff(spikes[(spikes > start - 500) & (spikes < start)]))
                sp_zero = get_spike_zero(spikes, start)
                t_zero = spikes[sp_zero]
                over_under = np.diff(spikes[sp_zero:]) > breakpoint
                sooner = np.nonzero(np.diff(spikes[(sp_zero + 1):]) > 2 * np.diff(spikes[sp_zero:-1]))[0]
                crossover_point = sp_zero + np.nonzero(over_under)[0][0]
                if len(sooner):
                    crossover_point = min(crossover_point, sp_zero + sooner[0] + 1)
                burst_isis = np.diff(spikes[sp_zero:(crossover_point + 1)])
                burst = breakpoint / np.mean(burst_isis)
                # print("burst_isis:", burst_isis)
                pause_isi = np.diff(spikes[crossover_point:(crossover_point + 2)])
                pause = breakpoint / np.mean(pause_isi)
                # print("pause_isi:", pause_isi)
                # print("B:", burst, "P:", pause, "BP:", burst / pause)
                BP = burst / pause
                # print("last time point:", spikes[crossover_point + 1])
                # fig.add_scatter(y=trace.data[tmask],x=t_masked, text=str(id), name=f"bp: {BP}; id: {id}")
                b_factors[id] = burst
                p_factors[id] = pause
                bp_factors[id] = BP
                windows[id] = spikes[crossover_point + 1]

        grc_ps = network.get_placement_set("granule_cell")
        pf_pc_cs = network.get_connectivity_set("parallel_fiber_to_purkinje").get_dataset().astype(int)
        m_pf_pc = sparse.coo_matrix((np.ones(len(pf_pc_cs)), (pf_pc_cs[:, 0], pf_pc_cs[:, 1])), shape=(max(grc_ps.identifiers) + 1, max(ps_pc.identifiers) + 1)).tocsr()
        aa_pc_cs = network.get_connectivity_set("ascending_axon_to_purkinje").get_dataset().astype(int)
        m_aa_pc = sparse.coo_matrix((np.ones(len(aa_pc_cs)), (aa_pc_cs[:, 0], aa_pc_cs[:, 1])), shape=(max(grc_ps.identifiers) + 1, max(ps_pc.identifiers) + 1)).tocsr()

        g = f["recorders/soma_spikes"]
        spikes_inc_pf = {id: [] for id in ps_pc.identifiers}
        spikes_inc_aa = {id: [] for id in ps_pc.identifiers}
        for id, d in g.items():
            if d.attrs["label"] != "granule_cell":
                continue
            spikes = crop(d[()], start, 15000)
            for t in  m_pf_pc.getrow(id).nonzero()[1]:
                spikes_inc_pf[t].extend(spikes[spikes < windows[t]])
            for t in m_aa_pc.getrow(id).nonzero()[1]:
                spikes_inc_aa[t].extend(spikes[spikes < windows[t]])
        pc_mask = (ps_pc.positions[:, 2] < 250) & (ps_pc.positions[:, 2] > 50)
        pf_spikes = [np.bincount(np.ceil((v - start) / bin_width).astype(int)) for v in map(np.array, spikes_inc_pf.values())]
        aa_spikes = [np.bincount(np.ceil((v - start) / bin_width).astype(int)) for v in map(np.array, spikes_inc_aa.values())]
        n_burst_bins = int(40 // bin_width)
        print("burst bins:", n_burst_bins)
        print("lens", [len(v) for v in pf_spikes])
        burst_spikes_pf = [np.sum(v[: n_burst_bins + 1], initial=0) for v in pf_spikes]
        burst_spikes_aa = [np.sum(v[: n_burst_bins + 1], initial=0) for v in aa_spikes]
        # go.Figure([
        #     go.Scatter(name=f"PF spikes on {id}", x=np.arange(start, windows[id], bin_width), y=y)
        #     for id, y in zip(windows.keys(), pf_spikes)
        # ]).show()
        b = list(b_factors.values())
        fig = go.Figure([
            go.Scatter3d(z=b, x=burst_spikes_pf, y=burst_spikes_aa, name="by sum", mode="markers"),
        ], layout=dict(
            scene=dict(
                xaxis=dict(title="# of burst-driving PF spikes"),
                yaxis=dict(title="# of burst-driving AA spikes"),
                zaxis=dict(title="Burst coefficient"),
            )
        ))

    return fig

def meta():
    return {"width": 1920 / 3 * 2}

if __name__ == "__main__":
    plot().show()
