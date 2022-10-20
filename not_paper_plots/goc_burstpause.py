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
    if len(np.where(spikes > stimulus))>0:
        a = np.where(spikes > stimulus)[0][0]
    else:
        a=0
    return a

def plot(path=None, net_path=None, input_device="mossy_fiber_sensory_burst", buffer=200, cutoff=5000, bin_width=5):
    if path is None:
        #path = glob(results_path("sensory_burst", "*"))[0]
        path=results_path("sensory_burst_noGoCGABA_noGoCgap.hdf5")
    if net_path is None:
        net_path = network_path(selection.network)
    network = from_hdf5(net_path)
    config = get_result_config(path)
    figs = {}
    with h5py.File(path, "r") as f:
        ps_goc = network.get_placement_set("golgi_cell")
        traces = hdf5_gather_voltage_traces(f, "recorders/soma_voltages/", map(str, map(int, ps_goc.identifiers)))
        goc_act = {id: f[f"/recorders/soma_spikes/{int(id)}"][:, 1] for id in ps_goc.identifiers}
        # traces.set_legends(["Membrane potential"])
        # traces.set_colors([config.cell_types["golgi_cell"].plotting.color])
        device = next(iter(config.simulations.values())).devices[input_device]
        if hasattr(device, "spike_times"):
            start = min(map(float, device.spike_times))
            stop = max(map(float, device.spike_times))
        else:
            params = device.parameters
            start = float(params["start"])
            stop = start + (float(params["interval"]) * float(params["number"]))

        input_region = [start, stop]
        print(start, stop)

        t = f["time"][()]
        tmask = (t > start - buffer) & (t < stop + 800 + buffer)
        t_masked = t[tmask]
        nb = {id: 0 for id in ps_goc.identifiers}
        b_factors = {id: 1 for id in ps_goc.identifiers}
        p_factors = {id: 1 for id in ps_goc.identifiers}
        bp_factors = {id: 1 for id in ps_goc.identifiers}
        windows = {id: stop + 50 for id in ps_goc.identifiers}
        p_windows = {id: stop + 50 for id in ps_goc.identifiers}

        for cell_trace in traces:
            id = cell_trace.cell_id
            for trace in cell_trace:
                spikes = goc_act[int(id)]
                print(spikes)
                breakpoint = np.mean(np.diff(spikes[(spikes > start - 500) & (spikes < start)]))
                numBurstSpikes = len(np.array(spikes[np.logical_and  (spikes>start, spikes<=stop+30)]))
                print(breakpoint, numBurstSpikes)
                if numBurstSpikes>1:
                    pauseDur = np.array(spikes[np.logical_and  (spikes>start,spikes>stop+30 )])[0]-np.array(spikes[np.logical_and  (spikes>start, spikes<=stop+30)])[-1]
                    print(pauseDur)



        #         sp_zero = get_spike_zero(spikes, start)
        #         if sp_zero !=0:
        #             t_zero = spikes[sp_zero]
        #             over_under = np.diff(spikes[sp_zero:]) > breakpoint
        #
        #             if len(np.nonzero(over_under)[0])==0:
        #                 continue
        #             sooner = np.nonzero(np.diff(spikes[(sp_zero + 1):]) > 2 * np.diff(spikes[sp_zero:-1]))[0]
        #             crossover_point = sp_zero + np.nonzero(over_under)[0][0]
        #             pause_isi = np.diff(spikes[crossover_point:(crossover_point + 2)])
        #             print("pause isi: ", pause_isi)
        #             pause_end = spikes[crossover_point + 1]
        #             pause = breakpoint / np.mean(pause_isi)
        #             if len(sooner):
        #                 crossover_point = min(crossover_point, sp_zero + sooner[0] + 1)
        #             burst_isis = np.diff(spikes[sp_zero:(crossover_point + 1)])
        #             print("burst isi: ", burst_isis)
        #             burst = breakpoint / np.mean(burst_isis)
        #             BP = burst / pause
        #             b_factors[id] = burst
        #             p_factors[id] = 1 / pause
        #             bp_factors[id] = BP
        #             nb[id] = len(burst_isis)
        #             p_windows[id] = pause_end - start
        #         else:
        #             b_factors[id] = "NaN"
        #             p_factors[id] = "NaN"
        #             bp_factors[id] = "NaN"
        #             nb[id] = "NaN"
        #             p_windows[id] = "NaN"
        #
        # figs["bp"] = go.Figure(go.Scatter(x=list(p_factors.values()), y=list(b_factors.values()), mode="markers"), layout=dict(title_text="B/P relationship"))


        # grc_ps = network.get_placement_set("granule_cell")
        # pf_goc_cs = network.get_connectivity_set("parallel_fiber_to_golgi").get_dataset().astype(int)
        # m_pf_goc = sparse.coo_matrix((np.ones(len(pf_goc_cs)), (pf_goc_cs[:, 0], pf_goc_cs[:, 1])), shape=(max(grc_ps.identifiers) + 1, max(ps_goc.identifiers) + 1)).tocsr()
        # aa_goc_cs = network.get_connectivity_set("ascending_axon_to_golgi").get_dataset().astype(int)
        # m_aa_goc = sparse.coo_matrix((np.ones(len(aa_goc_cs)), (aa_goc_cs[:, 0], aa_goc_cs[:, 1])), shape=(max(grc_ps.identifiers) + 1, max(ps_goc.identifiers) + 1)).tocsr()
        # goc_ps = network.get_placement_set("golgi_cell")
        # goc_goc_cs = network.get_connectivity_set("golgi_to_golgi").get_dataset().astype(int)
        # m_goc_goc = sparse.coo_matrix((np.ones(len(goc_goc_cs)), (bc_pc_cs[:, 0], goc_goc_cs[:, 1])), shape=(max(goc_ps.identifiers) + 1, max(ps_goc.identifiers) + 1)).tocsr()
        # gap_goc_cs = network.get_connectivity_set("gap_goc").get_dataset().astype(int)
        # m_gap_goc = sparse.coo_matrix((np.ones(len(gap_goc_cs)), (gap_goc_cs[:, 0], gap_goc_cs[:, 1])), shape=(max(goc_ps.identifiers) + 1, max(ps_goc.identifiers) + 1)).tocsr()
        # glom_ps = network.get_placement_set("glomerulus")
        # glom_goc_cs = network.get_connectivity_set("glomerulus_to_golgi").get_dataset().astype(int)
        # m_glom_goc = sparse.coo_matrix((np.ones(len(glom_goc_cs)), (glom_goc_cs[:, 0], glom_goc_cs[:, 1])), shape=(max(glom_ps.identifiers) + 1, max(ps_goc.identifiers) + 1)).tocsr()
        #
        # g = f["recorders/soma_spikes"]
        # spikes_inc_pf = {id: [] for id in ps_goc.identifiers}
        # spikes_inc_aa = {id: [] for id in ps_goc.identifiers}
        # spikes_inc_goc = {id: [] for id in ps_goc.identifiers}
        # spikes_inc_gap_goc = {id: [] for id in ps_goc.identifiers}
        # spikes_inc_glom = {id: [] for id in ps_goc.identifiers}
        # for id, d in g.items():
        #     if d.attrs["label"] == "granule_cell":
        #         spikes = crop(d[()], start, 15000)
        #         for t in  m_pf_goc.getrow(id).nonzero()[1]:
        #             spikes_inc_pf[t].extend(spikes[spikes < windows[t]])
        #         for t in m_aa_goc.getrow(id).nonzero()[1]:
        #             spikes_inc_aa[t].extend(spikes[spikes < windows[t]])
        #     elif d.attrs["label"] == "golgi_cell":
        #         spikes = crop(d[()], start, 15000)
        #         for t in  m_goc_goc.getrow(id).nonzero()[1]:
        #             spikes_inc_goc[t].extend(spikes[spikes < windows[t]])
        #         for t in m_gap_goc.getrow(id).nonzero()[1]:
        #             spikes_inc_gap_goc[t].extend(spikes[spikes < windows[t]])
        #     elif d.attrs["label"] == "glomerulus":
        #         spikes = crop(d[()], start, 15000)
        #         for t in  m_glom_goc.getrow(id).nonzero()[1]:
        #             spikes_inc_glom[t].extend(spikes[spikes < windows[t]])
        #
        #
        # pf_spikes = [np.bincount(np.ceil((v - start) / bin_width).astype(int)) for v in map(np.array, spikes_inc_pf.values())]
        # aa_spikes = [np.bincount(np.ceil((v - start) / bin_width).astype(int)) for v in map(np.array, spikes_inc_aa.values())]
        # goc_spikes = [np.bincount(np.ceil((v - start) / bin_width).astype(int), minlength=40) for v in map(np.array, spikes_inc_goc.values())]
        # gap_goc_spikes = [np.bincount(np.ceil((v - start) / bin_width).astype(int), minlength=40) for v in map(np.array, spikes_inc_gap_goc.values())]
        # glom_spikes = [np.bincount(np.ceil((v - start) / bin_width).astype(int), minlength=40) for v in map(np.array, spikes_inc_glom.values())]
        # #mli_spikes = list(map(sum, zip(bc_spikes, sc_spikes)))
        # # figs["mli_p"] = go.Figure(
        # #     go.Scatter(
        # #         y=[sum(mli_spikes[i][:int(p_windows[id] // bin_width + 2)]) for i, id in enumerate(ps_goc.identifiers)],
        # #         x=list(p_factors.values()),
        # #         mode="markers",
        # #     ),
        # #     layout=dict(
        # #         title_text="MLI/P-coeff",
        # #         xaxis=dict(title="Pause coefficient"),
        # #         yaxis=dict(title="# of pause-driving MLI spikes"),
        # #     ),
        # # )
        # # figs["bc_sc_p"] = go.Figure(
        # #     go.Scatter3d(
        # #         x=[sum(sc_spikes[i][:int(p_windows[id] // bin_width + 2)]) for i, id in enumerate(ps_goc.identifiers)],
        # #         y=[sum(bc_spikes[i][:int(p_windows[id] // bin_width + 2)]) for i, id in enumerate(ps_goc.identifiers)],
        # #         z=list(p_factors.values()),
        # #         mode="markers",
        # #     ),
        # #     layout=dict(
        # #         title_text="BC/SC/P-coeff",
        # #         scene=dict(
        # #             xaxis=dict(title="# of pause-driving SC spikes"),
        # #             yaxis=dict(title="# of pause-driving BC spikes"),
        # #             zaxis=dict(title="Pause coefficient"),
        # #         )
        # #     ),
        # # )
        # n_burst_bins = int(40 // bin_width)
        # burst_spikes_pf = [np.sum(v[: n_burst_bins + 1], initial=0) for v in pf_spikes]
        # burst_spikes_aa = [np.sum(v[: n_burst_bins + 1], initial=0) for v in aa_spikes]
        # b = list(b_factors.values())
        # figs["bcoeff_colored"] = go.Figure(
        #     [
        #         go.Scatter3d(
        #             z=b,
        #             x=burst_spikes_pf,
        #             y=burst_spikes_aa,
        #             marker=dict(
        #                 showscale=True,
        #                 colorscale="Viridis",
        #                 cmin=0,
        #                 cmax=max(nb.values()),
        #                 color=list(nb.values()),
        #                 colorbar=dict(
        #                     tickmode="linear",
        #                     tick0=1,
        #                     dtick=1,
        #                 ),
        #             ),
        #             name="by sum",
        #             mode="markers"
        #         ),
        #     ],
        #     layout=dict(
        #         title_text="Burst coefficient analysis",
        #         scene=dict(
        #             xaxis=dict(title="# of burst-driving PF spikes"),
        #             yaxis=dict(title="# of burst-driving AA spikes"),
        #             zaxis=dict(title="Burst coefficient"),
        #         )
        #     )
        # )
        # figs["bcoeff"] = go.Figure(
        #     [
        #         go.Scatter3d(
        #             z=b,
        #             x=burst_spikes_pf,
        #             y=burst_spikes_aa,
        #             name="by sum",
        #             mode="markers"
        #         ),
        #     ],
        #     layout=dict(
        #         title_text="Burst coefficient analysis",
        #         scene=dict(
        #             xaxis=dict(title="# of burst-driving PF spikes"),
        #             yaxis=dict(title="# of burst-driving AA spikes"),
        #             zaxis=dict(title="Burst coefficient"),
        #         )
        #     )
        # )

    return figs

def meta(key):
    return {"width": 1920 / 3 * 2}

if __name__ == "__main__":
    plot().show()
