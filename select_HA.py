from bsb.core import from_hdf5
import h5py, os, sys, numpy as np, plotly.graph_objs as go
from scipy import stats
import statistics
_bounds = dict(min=700, max=800)

def bounds(data, min=(-float("inf")), max=(+float("inf")), indices=False):
    c = data[:, 0]
    if indices:
        return np.where((c > min) & (c < max))[0]
    return c[(c > min) & (c < max)]

# scaffold = from_hdf5("C:/Users/robin/Dropbox/Scaffold_NEURON_paper/neuronFINAL_V5.hdf5")
scaffold = from_hdf5("networks/results.hdf5")
file = "combined_results.hdf5"
results = h5py.File(file, "r")
ps = scaffold.get_placement_set("granule_cell")
ids = ps.identifiers
spikes_per_dict = {id: [] for id in ids}
g = results["recorders/soma_spikes/"]
for key in g:
    f = g[key]
    if f.attrs["label"] != ps.tag or len(f) == 0:
        continue
    id = int(f[0, 1])
    spikes_per_dict[id].extend(bounds(f[()], **_bounds))

spikes_per_grc = [spikes_per_dict[x] for x in sorted(spikes_per_dict)]

n_spikes_per_grc = map(len, spikes_per_grc)
np_spikes = np.array(list(map(len, spikes_per_grc)))
norm_spikes = np_spikes / np.max(np_spikes)

pos = np.array(ps.positions)
pos_arr = []
kde_counter = 0
kde_roi = []
pos_roi = []
firing_cells = 0
for i, n_spikes in enumerate(n_spikes_per_grc):
    if n_spikes != 0:
        firing_cells += 1
        kde_roi.append(kde_counter)
        pos_roi.append(i)
    kde_counter += n_spikes
    p = pos[i]
    pos_arr.extend(p for _ in range(n_spikes))

spikes_in_space = np.array(pos_arr)

# PRODUCES THE NON COLORED 'STACKING OPACITY SPIKES IN SPACE' PLOT
# go.Figure(go.Scatter3d(x=spikes_in_space[:,0],y=spikes_in_space[:,2],z=spikes_in_space[:,1], mode="markers", marker=dict(opacity=0.05))).show()

values = spikes_in_space.T
kde = stats.gaussian_kde(values)
density = kde(values)
grc_densities = density[kde_roi]
norm = grc_densities / np.max(grc_densities)
_fullpos = pos
pos = pos[pos_roi]

def get_isis(spikes, selected):
    return [spikes[i + 1] - spikes[i] for i in selected]

high_activity_ids_red = ids[pos_roi][norm > 0.5]
high_activity_ids_green = ids[norm_spikes > 0.5]
high_activity_ids = high_activity_ids_red if "green" not in sys.argv else high_activity_ids_green

def complete_sets(ids, *sets, column=1, ignore=[]):
    carry = {id: 0 for id in ids}
    for unique_ids, unique_counts in (np.unique(set[:, column], return_counts=True) for set in sets):
        for id, count in zip(unique_ids, unique_counts):
            carry[id] += count
    return np.array([[id, count if id not in ignore else 0] for id, count in carry.items()])

def filter(set, column=0):
    return set[np.isin(set[:, column], high_activity_ids)]

ps_pc = scaffold.get_placement_set("granule_cell")
ps_pc = scaffold.get_placement_set("purkinje_cell")
ps_gc = scaffold.get_placement_set("golgi_cell")
ps_sc = scaffold.get_placement_set("stellate_cell")
ps_bc = scaffold.get_placement_set("basket_cell")
cs_mf_glom = scaffold.get_connectivity_set("mossy_to_glomerulus").get_dataset()
cs_glom_grc = scaffold.get_connectivity_set("glomerulus_to_granule").get_dataset()
cs_pc_pf = scaffold.get_connectivity_set("parallel_fiber_to_purkinje").get_dataset()
cs_pc_aa = scaffold.get_connectivity_set("ascending_axon_to_purkinje").get_dataset()
cs_gc_pf = scaffold.get_connectivity_set("parallel_fiber_to_golgi").get_dataset()
cs_gc_aa = scaffold.get_connectivity_set("ascending_axon_to_golgi").get_dataset()
cs_sc_pf = scaffold.get_connectivity_set("parallel_fiber_to_stellate").get_dataset()
cs_bc_pf = scaffold.get_connectivity_set("parallel_fiber_to_basket").get_dataset()
pc_pos = ps_pc.positions
gc_pos = ps_gc.positions
sc_pos = ps_sc.positions
bc_pos = ps_bc.positions

border_pc = pc_pos[:, 2] > 175
border_mask = np.logical_not(border_pc)
conn_pc = complete_sets(ps_pc.identifiers, filter(cs_pc_pf), filter(cs_pc_aa), ignore=ps_pc.identifiers[border_pc])
conn_aa_pc = complete_sets(ps_pc.identifiers, filter(cs_pc_aa), ignore=ps_pc.identifiers[border_pc])
conn_gc = complete_sets(ps_gc.identifiers, filter(cs_gc_pf), filter(cs_gc_aa))
conn_aa_gc = complete_sets(ps_gc.identifiers, filter(cs_gc_aa))
conn_sc = complete_sets(ps_sc.identifiers, filter(cs_sc_pf))
conn_bc = complete_sets(ps_bc.identifiers, filter(cs_bc_pf))

if "check" in sys.argv:
    out = {}
    for ct in [['pc', 'aa'], ['gc', 'aa'], ['sc'], ['bc']]:
        jobs = [''] + ['_' + t for t in ct[1:]]
        traces = []
        for job in jobs:
            name = 'conn' + job + '_' + ct[0]
            for div in [(5, '80', 'blue'), (10, '90', 'green')]:
                outname = ct[0] + 'b' + job[1:] + '_' + div[1]
                var = vars()[name]
                pos = vars()[ct[0] + '_pos']
                sorted_check = var[np.argsort(var[:,1])]
                threshold = np.sum(sorted_check[:, 1]) / div[0]
                tally = 0
                for id, count in enumerate(sorted_check[:, 1]):
                    cutoff_id = id
                    cutoff_count = count
                    tally += count
                    if tally > threshold:
                        break
                else:
                    raise RuntimeError("No data?")
                traces.extend([
                    go.Scatter(x=list(range(len(sorted_check))), y=sorted_check[:, 1]),
                    go.Scatter(x=[cutoff_id], y=[cutoff_count], mode="markers", marker=dict(color=div[2]))
                ])
                globals()[outname] = cutoff_count
                print(outname, '=', cutoff_count)
                mask = var[:, 1] >= cutoff_count
                act_check = var[mask]
                act_pos = pos[mask]
                fig = go.Figure([
                    go.Scatter3d(
                        x=pos[:, 0], y=pos[:, 2], z=pos[:, 1], mode="markers", name="Cut off cells",
                        marker=dict(
                            color="grey", opacity=0.2
                        )
                    ),
                    go.Scatter3d(
                        x=act_pos[:, 0], y=act_pos[:, 2], z=act_pos[:, 1], mode="markers", name="Selected cells",
                        marker=dict(
                            colorscale="Viridis", cmin=0, cmax=np.max(act_check[:, 1]), color=act_check[:, 1]
                        ),
                        text=act_check[:, 1]
                    )
                ])
                fig.update_layout(title_text="Inclusion " + name + div[1])
                # fig.show()

        fig = go.Figure(traces)
        fig.update_layout(title_text="Rank-size " + name + div[1])
        fig.show()
        print()
else:
    # selected_mf = [229,213, 221, 228]
    # gloms = cs_mf_glom[np.isin(cs_mf_glom[:, 0], selected_mf), 1]
    # ha_grc = cs_glom_grc[np.isin(cs_glom_grc[:, 0], gloms), 1]

    pcb_80 = 149
    pcb_90 = 110
    pcbaa_80 = 95
    pcbaa_90 = 73

    gcb_80 = 117
    gcb_90 = 85
    gcbaa_80 = 226
    gcbaa_90 = 70

    scb_80 = 50
    scb_90 = 30

    bcb_80 = 80
    bcb_90 = 49

# GrC
ha_grc = high_activity_ids_red
# PC
ha_pc_80 = conn_pc[(conn_pc[:, 1] >= pcb_80) & border_mask, 0]
ha_pc_90 = conn_pc[(conn_pc[:, 1] >= pcb_90) & border_mask, 0]
aa_pc_80 = conn_pc[(conn_aa_pc[:, 1] >= pcbaa_80)  & border_mask, 0]
aa_pc_90 = conn_pc[(conn_aa_pc[:, 1] >= pcbaa_90)  & border_mask, 0]
pf_pc_80_80 = conn_pc[(conn_pc[:, 1] >= pcb_80) & (conn_aa_pc[:, 1] < pcbaa_80) & border_mask, 0]
pf_pc_80_90 = conn_pc[(conn_pc[:, 1] >= pcb_80) & (conn_aa_pc[:, 1] < pcbaa_90) & border_mask, 0]
pf_pc_90_80 = conn_pc[(conn_pc[:, 1] >= pcb_90) & (conn_aa_pc[:, 1] < pcbaa_80) & border_mask, 0]
pf_pc_90_90 = conn_pc[(conn_pc[:, 1] >= pcb_90) & (conn_aa_pc[:, 1] < pcbaa_90) & border_mask, 0]
la_pc_80 = conn_pc[(conn_pc[:, 1] < pcb_80) & (conn_aa_pc[:, 1] < pcbaa_90) & border_mask, 0]
la_pc_90 = conn_pc[(conn_pc[:, 1] < pcb_90) & (conn_aa_pc[:, 1] < pcbaa_90) & border_mask, 0]
# GC
ha_gc_80 = conn_gc[(conn_gc[:, 1] >= gcb_80), 0]
ha_gc_90 = conn_gc[(conn_gc[:, 1] >= gcb_90), 0]
aa_gc_80 = conn_gc[(conn_aa_gc[:, 1] >= gcbaa_80) , 0]
aa_gc_90 = conn_gc[(conn_aa_gc[:, 1] >= gcbaa_90) , 0]
pf_gc_80_80 = conn_gc[(conn_gc[:, 1] >= gcb_80) & (conn_aa_gc[:, 1] < gcbaa_80), 0]
pf_gc_80_90 = conn_gc[(conn_gc[:, 1] >= gcb_80) & (conn_aa_gc[:, 1] < gcbaa_90), 0]
pf_gc_90_80 = conn_gc[(conn_gc[:, 1] >= gcb_90) & (conn_aa_gc[:, 1] < gcbaa_80), 0]
pf_gc_90_90 = conn_gc[(conn_gc[:, 1] >= gcb_90) & (conn_aa_gc[:, 1] < gcbaa_90), 0]
la_gc_80 = conn_gc[(conn_gc[:, 1] < gcb_80) & (conn_aa_gc[:, 1] < gcbaa_90), 0]
la_gc_90 = conn_gc[(conn_gc[:, 1] < gcb_90) & (conn_aa_gc[:, 1] < gcbaa_90), 0]
# SC
ha_sc_80 = conn_sc[conn_sc[:, 1] >= scb_80, 0]
ha_sc_90 = conn_sc[conn_sc[:, 1] >= scb_90, 0]
la_sc_80 = conn_sc[conn_sc[:, 1] < scb_80, 0]
la_sc_90 = conn_sc[conn_sc[:, 1] < scb_90, 0]
# BC
ha_bc_80 = conn_bc[conn_bc[:, 1] >= bcb_80, 0]
ha_bc_90 = conn_bc[conn_bc[:, 1] >= bcb_90, 0]
la_bc_80 = conn_bc[conn_bc[:, 1] < bcb_80, 0]
la_bc_90 = conn_bc[conn_bc[:, 1] < bcb_90, 0]


def chain(*iters):
    for iter in iters:
        while True:
            try:
                yield next(iter)
            except StopIteration:
                break

results.close()
if "select" in sys.argv:
    with h5py.File("selected_results.hdf5", "w") as x:
        x.create_group("selections")
        with h5py.File(file, "r") as f:
            transfer_map = {id: [] for id in np.unique(np.concatenate((ha_grc, ha_pc_80, ha_pc_90, aa_pc_80, aa_pc_90, pf_pc_80_80, pf_pc_80_90, pf_pc_90_80, pf_pc_90_90, la_pc_80, la_pc_90, ha_gc_80, ha_gc_90, aa_gc_80, aa_gc_90, pf_gc_80_80, pf_gc_80_90, pf_gc_90_80, pf_gc_90_90, la_gc_80, la_gc_90, ha_sc_80, ha_sc_90, la_sc_80, la_sc_90, ha_bc_80, ha_bc_90, la_bc_80, la_bc_90)))}
            for set, transfer, attrs in [
                (ha_pc_80, 'ha_pc_80', {"stack": "##1 Stimulated PC"}),
                (ha_pc_90, 'ha_pc_90', {"stack": "##1 Stimulated PC"}),
                (la_pc_80, 'la_pc_80', {"stack": "##0 Unstimulated PC"}),
                (la_pc_90, 'la_pc_90', {"stack": "##0 Unstimulated PC"}),
                (pf_pc_80_80, 'pf_pc_80_80', {"stack": "##1 PC stimulated by PF"}),
                (pf_pc_80_90, 'pf_pc_80_90', {"stack": "##1 PC stimulated by PF"}),
                (pf_pc_90_80, 'pf_pc_90_80', {"stack": "##1 PC stimulated by PF"}),
                (pf_pc_90_90, 'pf_pc_90_90', {"stack": "##1 PC stimulated by PF"}),
                (aa_pc_80, 'aa_pc_80', {"stack": "##2 PC stimulated by PF + AA"}),
                (aa_pc_90, 'aa_pc_90', {"stack": "##2 PC stimulated by PF + AA"}),
                (ha_gc_80, 'ha_gc_80', {"stack": "##1 Stimulated GC"}),
                (ha_gc_90, 'ha_gc_90', {"stack": "##1 Stimulated GC"}),
                (la_gc_80, 'la_gc_80', {"stack": "##0 Unstimulated GC"}),
                (la_gc_90, 'la_gc_90', {"stack": "##0 Unstimulated GC"}),
                (pf_gc_80_80, 'pf_gc_80_80', {"stack": "##1 GC stimulated by PF"}),
                (pf_gc_80_90, 'pf_gc_80_90', {"stack": "##1 GC stimulated by PF"}),
                (pf_gc_90_80, 'pf_gc_90_80', {"stack": "##1 GC stimulated by PF"}),
                (pf_gc_90_90, 'pf_gc_90_90', {"stack": "##1 GC stimulated by PF"}),
                (aa_gc_80, 'aa_gc_80', {"stack": "##2 GC stimulated by PF + AA"}),
                (aa_gc_90, 'aa_gc_90', {"stack": "##2 GC stimulated by PF + AA"}),
                (ha_sc_90, 'ha_sc_90', {"stack": "##1 Stimulated SC"}),
                (ha_sc_80, 'ha_sc_80', {"stack": "##1 Stimulated SC"}),
                (la_sc_90, 'la_sc_90', {"stack": "##0 Unstimulated SC"}),
                (la_sc_80, 'la_sc_80', {"stack": "##0 Unstimulated SC"}),
                (ha_bc_90, 'ha_bc_90', {"stack": "##1 Stimulated BC"}),
                (ha_bc_80, 'ha_bc_80', {"stack": "##1 Stimulated BC"}),
                (la_bc_90, 'la_bc_90', {"stack": "##0 Unstimulated BC"}),
                (la_bc_80, 'la_bc_80', {"stack": "##0 Unstimulated BC"}),
                (ha_grc, 'ha_granule', {"stack": "Stimulated GrC"})
            ]:
                x.create_dataset("selections/" + transfer, data=set)
                x.create_group(transfer)
                for id in set:
                    transfer_map[id].append((transfer, attrs))

            g = f["recorders/soma_spikes"]
            for h in g:
                if g[h].shape[0] == 0:
                    continue
                id = g[h][0, 1]
                if id in transfer_map:
                    for transfer, attrs in transfer_map[id]:
                        print("copying", h, "part of id", id, "to", transfer, " " * 10, end="\r")
                        d = x[transfer].create_dataset(h, data=g[h][()])
                        d_attrs_kv = iter(g[h].attrs.items())
                        if len({k: v for k, v in chain(iter(g[h].attrs.items()), iter(attrs.items()))}) < 4:
                            raise Exception(
                                "{} Not enough kv {}".format(h, {k: v for k, v in chain(iter(g[h].attrs.items()), iter(attrs.items()))}))
                        if attrs is not None:
                            d_attrs_kv = chain(d_attrs_kv, iter(attrs.items()))
                        for k, v in d_attrs_kv:
                            d.attrs[k] = v

if "group" in sys.argv:
    groups = {
        "stacked_psth": [
            'ha_granule',
            'la_pc_80', 'pf_pc_80_90', 'aa_pc_90',
            'la_gc_80', 'pf_gc_80_90', 'aa_gc_90',
            'la_sc_80', 'ha_sc_80',
            'la_bc_80', 'ha_bc_80',
        ],
    }
    with h5py.File("selected_results.hdf5", "a") as f:
        for n, g in groups.items():
            f.create_group(n)
            for ds in g:
                for s in f[ds]:
                    d = f[n].create_dataset(s, data=f[ds][s][()])
                    for k, v in f[ds][s].attrs.items():
                        d.attrs[k] = v
