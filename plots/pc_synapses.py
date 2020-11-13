from bsb.core import from_hdf5
from bsb.plotting import plot_morphology #, plot_synapses
from plotly import graph_objs as go
import selection, numpy as np
from colour import Color

def plot():
    network = from_hdf5("networks/300x_200z.hdf5")
    mr = network.morphology_repository
    m = mr.get_morphology("PurkinjeCell")
    ps = network.get_placement_set("purkinje_cell")
    ps_grc = network.get_placement_set("granule_cell")

    stim_mf = selection.stimulated_mf_poiss
    cs_mf_glom = network.get_connectivity_set("mossy_to_glomerulus")
    cs_glom_grc = network.get_connectivity_set("glomerulus_to_granule")
    aa_pc_conn = network.get_connectivity_set("ascending_axon_to_purkinje")
    pf_pc_conn = network.get_connectivity_set("parallel_fiber_to_purkinje")
    gaba_conn = network.get_connectivity_set("stellate_to_purkinje")
    # Mask the glomeruli not connected to active MF
    stim_glom_mask = np.isin(cs_mf_glom.get_dataset()[:, 0], stim_mf)
    stim_glom = cs_mf_glom.get_dataset()[stim_glom_mask]
    # Get the region of interest of the granule cells connected to active glom
    roi_glom_grc = np.isin(cs_glom_grc.get_dataset()[:, 0], stim_glom)
    # Count the occurences of grc connections with stimulated glomeruli
    stim_grc, counts = np.unique(cs_glom_grc.get_dataset()[roi_glom_grc, 1], return_counts=True)
    incomplete_grc_count_map = dict(zip(stim_grc, counts))
    # Complete the map with those grc that did not occur (0 connections).
    grc_count_map = {id: incomplete_grc_count_map.get(id, 0) for id in ps_grc.identifiers}

    marker_sizes = {0: 1.0, 1: 2.0, 2: 2.0, 3: 3.0, 4: 4.0}
    marker_opacity = {0: 0.3, 1: 0.5, 2: 1.0, 3: 1.0, 4: 1.0}
    markers = {"ascending_axon_to_purkinje": "circle", "parallel_fiber_to_purkinje": "diamond"}
    labels = {"ascending_axon_to_purkinje": "AA", "parallel_fiber_to_purkinje": "PF"}

    grc_color = network.configuration.cell_types["granule_cell"].plotting.color
    pc_color = network.configuration.cell_types["purkinje_cell"].plotting.color
    sc_color = network.configuration.cell_types["stellate_cell"].plotting.color
    pc_radius = network.configuration.cell_types["purkinje_cell"].placement.soma_radius
    pc_scale = list(map(str, Color(pc_color).range_to("white", 10)))
    pc_colors = {
        "soma": pc_scale[0],
        "basal_dendrites": pc_scale[0],
        "pf_targets": pc_scale[1],
        "aa_targets": pc_scale[2],
        "axon": pc_scale[3]
    }
    figs = {}
    for pc_label, pc_id in selection.purkinje_cells.items():
        pc_pos = ps.positions[ps.identifiers==pc_id][0]
        fig = plot_morphology(m, show=False, color=pc_colors, soma_radius=pc_radius, offset=pc_pos)
        fig.update_layout(title_text=f"{pc_label} purkinje cell")
        for set in (aa_pc_conn, pf_pc_conn):
            tag_label = labels[set.tag]
            positions = [[] for _ in range(5)]
            for intersection in set.intersections:
                if intersection.to_id == pc_id:
                    count = grc_count_map[intersection.from_id]
                    positions[count].append(intersection.to_compartment.midpoint)
            for count in range(5):
                pos = np.array(positions[count])
                if not len(pos):
                    pos = np.empty((0,3))
                t = go.Scatter3d(
                    x=pos[:,0] + pc_pos[0],
                    y=pos[:,2] + pc_pos[2],
                    z=pos[:,1] + pc_pos[1],
                    mode="markers",
                    opacity=marker_opacity[count],
                    marker=dict(
                        size=marker_sizes[count],
                        symbol=markers[set.tag],
                        color=grc_color,
                    ),
                    name=f"Granule cell {tag_label} synapses with {count} active dendrites"
                )
                fig.add_trace(t)
        gabas = np.array([i.to_compartment.midpoint for i in gaba_conn.intersections if i.to_id == pc_id])
        fig.add_trace(go.Scatter3d(
            x=gabas[:,0] + pc_pos[0],
            y=gabas[:,2] + pc_pos[2],
            z=gabas[:,1] + pc_pos[1],
            name="Stellate GABA synapses",
            mode="markers",
            marker=dict(
                size=2.0,
                color=sc_color,
                symbol="diamond",
                line=dict(
                    width=1,
                    color="black",
                )
            )
        ))
        cfg = selection.btn_config.copy()
        cfg["filename"] = pc_label[0] + "_purkinje_cell_synapses"
        figs[pc_label[0]] = fig
    return figs

if __name__ == "__main__":
    plot()
