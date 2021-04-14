from bsb.core import from_hdf5
from bsb.config import get_result_config
from bsb.plotting import plot_morphology, hdf5_gather_voltage_traces, plot_traces
from scipy import sparse
from plotly import graph_objs as go
import selection, numpy as np, h5py
from colour import Color
from ._paths import *
from glob import glob

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
        path = glob(results_path("sensory_burst", "*"))[0]
    if net_path is None:
        net_path = network_path(selection.network)
    network = from_hdf5(net_path)
    config = get_result_config(path)
    figs = {}
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
        nb = {id: 0 for id in ps_pc.identifiers}
        b_factors = {id: 1 for id in ps_pc.identifiers}
        p_factors = {id: 1 for id in ps_pc.identifiers}
        bp_factors = {id: 1 for id in ps_pc.identifiers}
        windows = {id: stop + 50 for id in ps_pc.identifiers}
        p_windows = {id: stop + 50 for id in ps_pc.identifiers}
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
                pause_isi = np.diff(spikes[crossover_point:(crossover_point + 2)])
                pause_end = spikes[crossover_point + 1]
                pause = breakpoint / np.mean(pause_isi)
                if len(sooner):
                    crossover_point = min(crossover_point, sp_zero + sooner[0] + 1)
                burst_isis = np.diff(spikes[sp_zero:(crossover_point + 1)])
                burst = breakpoint / np.mean(burst_isis)
                BP = burst / pause
                b_factors[id] = burst
                p_factors[id] = 1 / pause
                bp_factors[id] = BP
                nb[id] = len(burst_isis)
                p_windows[id] = pause_end - start

        figs["bp"] = go.Figure(go.Scatter(x=list(p_factors.values()), y=list(b_factors.values()), mode="markers"), layout=dict(title_text="B/P relationship"))

        grc_ps = network.get_placement_set("granule_cell")
        pf_pc_cs = network.get_connectivity_set("parallel_fiber_to_purkinje").get_dataset().astype(int)
        m_pf_pc = sparse.coo_matrix((np.ones(len(pf_pc_cs)), (pf_pc_cs[:, 0], pf_pc_cs[:, 1])), shape=(max(grc_ps.identifiers) + 1, max(ps_pc.identifiers) + 1)).tocsr()
        aa_pc_cs = network.get_connectivity_set("ascending_axon_to_purkinje").get_dataset().astype(int)
        m_aa_pc = sparse.coo_matrix((np.ones(len(aa_pc_cs)), (aa_pc_cs[:, 0], aa_pc_cs[:, 1])), shape=(max(grc_ps.identifiers) + 1, max(ps_pc.identifiers) + 1)).tocsr()
        bc_ps = network.get_placement_set("basket_cell")
        bc_pc_cs = network.get_connectivity_set("basket_to_purkinje").get_dataset().astype(int)
        m_bc_pc = sparse.coo_matrix((np.ones(len(bc_pc_cs)), (bc_pc_cs[:, 0], bc_pc_cs[:, 1])), shape=(max(bc_ps.identifiers) + 1, max(ps_pc.identifiers) + 1)).tocsr()
        sc_ps = network.get_placement_set("stellate_cell")
        sc_pc_cs = network.get_connectivity_set("stellate_to_purkinje").get_dataset().astype(int)
        m_sc_pc = sparse.coo_matrix((np.ones(len(sc_pc_cs)), (sc_pc_cs[:, 0], sc_pc_cs[:, 1])), shape=(max(sc_ps.identifiers) + 1, max(ps_pc.identifiers) + 1)).tocsr()

        g = f["recorders/soma_spikes"]
        spikes_inc_pf = {id: [] for id in ps_pc.identifiers}
        spikes_inc_aa = {id: [] for id in ps_pc.identifiers}
        spikes_inc_bc = {id: [] for id in ps_pc.identifiers}
        spikes_inc_sc = {id: [] for id in ps_pc.identifiers}
        for id, d in g.items():
            if d.attrs["label"] == "granule_cell":
                spikes = crop(d[()], start, 15000)
                for t in  m_pf_pc.getrow(id).nonzero()[1]:
                    spikes_inc_pf[t].extend(spikes[spikes < windows[t]])
                for t in m_aa_pc.getrow(id).nonzero()[1]:
                    spikes_inc_aa[t].extend(spikes[spikes < windows[t]])
            elif d.attrs["label"] == "basket_cell":
                spikes = crop(d[()], start, 15000)
                for t in  m_bc_pc.getrow(id).nonzero()[1]:
                    spikes_inc_bc[t].extend(spikes[spikes < windows[t]])
            elif d.attrs["label"] == "stellate_cell":
                spikes = crop(d[()], start, 15000)
                for t in  m_sc_pc.getrow(id).nonzero()[1]:
                    spikes_inc_sc[t].extend(spikes[spikes < windows[t]])


        pf_spikes = [np.bincount(np.ceil((v - start) / bin_width).astype(int)) for v in map(np.array, spikes_inc_pf.values())]
        aa_spikes = [np.bincount(np.ceil((v - start) / bin_width).astype(int)) for v in map(np.array, spikes_inc_aa.values())]
        bc_spikes = [np.bincount(np.ceil((v - start) / bin_width).astype(int), minlength=40) for v in map(np.array, spikes_inc_bc.values())]
        sc_spikes = [np.bincount(np.ceil((v - start) / bin_width).astype(int), minlength=40) for v in map(np.array, spikes_inc_sc.values())]
        mli_spikes = list(map(sum, zip(bc_spikes, sc_spikes)))
        figs["mli_p"] = go.Figure(
            go.Scatter(
                y=[sum(mli_spikes[i][:int(p_windows[id] // bin_width + 2)]) for i, id in enumerate(ps_pc.identifiers)],
                x=list(p_factors.values()),
                mode="markers",
            ),
            layout=dict(
                title_text="MLI/P-coeff",
                xaxis=dict(title="Pause coefficient"),
                yaxis=dict(title="# of pause-driving MLI spikes"),
            ),
        )
        figs["bc_sc_p"] = go.Figure(
            go.Scatter3d(
                x=[sum(sc_spikes[i][:int(p_windows[id] // bin_width + 2)]) for i, id in enumerate(ps_pc.identifiers)],
                y=[sum(bc_spikes[i][:int(p_windows[id] // bin_width + 2)]) for i, id in enumerate(ps_pc.identifiers)],
                z=list(p_factors.values()),
                mode="markers",
            ),
            layout=dict(
                title_text="BC/SC/P-coeff",
                scene=dict(
                    xaxis=dict(title="# of pause-driving SC spikes"),
                    yaxis=dict(title="# of pause-driving BC spikes"),
                    zaxis=dict(title="Pause coefficient"),
                )
            ),
        )
        n_burst_bins = int(40 // bin_width)
        burst_spikes_pf = [np.sum(v[: n_burst_bins + 1], initial=0) for v in pf_spikes]
        burst_spikes_aa = [np.sum(v[: n_burst_bins + 1], initial=0) for v in aa_spikes]
        b = list(b_factors.values())
        figs["bcoeff_colored"] = go.Figure(
            [
                go.Scatter3d(
                    z=b,
                    x=burst_spikes_pf,
                    y=burst_spikes_aa,
                    marker=dict(
                        showscale=True,
                        colorscale="Viridis",
                        cmin=0,
                        cmax=max(nb.values()),
                        color=list(nb.values()),
                        colorbar=dict(
                            tickmode="linear",
                            tick0=1,
                            dtick=1,
                        ),
                    ),
                    name="by sum",
                    mode="markers"
                ),
            ],
            layout=dict(
                title_text="Burst coefficient analysis",
                scene=dict(
                    xaxis=dict(title="# of burst-driving PF spikes"),
                    yaxis=dict(title="# of burst-driving AA spikes"),
                    zaxis=dict(title="Burst coefficient"),
                )
            )
        )
        figs["bcoeff"] = go.Figure(
            [
                go.Scatter3d(
                    z=b,
                    x=burst_spikes_pf,
                    y=burst_spikes_aa,
                    name="by sum",
                    mode="markers"
                ),
            ],
            layout=dict(
                title_text="Burst coefficient analysis",
                scene=dict(
                    xaxis=dict(title="# of burst-driving PF spikes"),
                    yaxis=dict(title="# of burst-driving AA spikes"),
                    zaxis=dict(title="Burst coefficient"),
                )
            )
        )

    return figs

def meta(key):
    return {"width": 1920 / 3 * 2}

if __name__ == "__main__":
    plot().show()