from bsb.core import from_hdf5
from bsb.plotting import plot_morphology #, plot_synapses
from plotly import graph_objs as go
import selection, numpy as np
from colour import Color

def plot():
    network = from_hdf5("networks/300x_200z.hdf5")
    mr = network.morphology_repository
    mb = mr.get_morphology("BasketCell")
    ms = mr.get_morphology("StellateCell")
    ps_grc = network.get_placement_set("granule_cell")

    stim_mf = selection.stimulated_mf_poiss
    cs_mf_glom = network.get_connectivity_set("mossy_to_glomerulus")
    cs_glom_grc = network.get_connectivity_set("glomerulus_to_granule")
    pf_sc_conn = network.get_connectivity_set("parallel_fiber_to_stellate")
    pf_bc_conn = network.get_connectivity_set("parallel_fiber_to_basket")
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

    grc_color = network.configuration.cell_types["granule_cell"].plotting.color
    bc_color = network.configuration.cell_types["basket_cell"].plotting.color
    sc_color = network.configuration.cell_types["stellate_cell"].plotting.color
    bc_radius = network.configuration.cell_types["basket_cell"].placement.soma_radius
    sc_radius = network.configuration.cell_types["stellate_cell"].placement.soma_radius
    bc_scale = list(map(str, Color(bc_color).range_to("black", 10)))
    sc_scale = list(map(str, Color(sc_color).range_to("black", 10)))
    bc_colors = {
        "soma": bc_scale[0],
        "dendrites": bc_scale[0],
        "basal_dendrites": bc_scale[0],
        "apical_dendrites": bc_scale[1],
        "axon": bc_scale[3]
    }
    sc_colors = {
        "soma": sc_scale[0],
        "dendrites": bc_scale[0],
        "basal_dendrites": sc_scale[0],
        "apical_dendrites": sc_scale[1],
        "axon": sc_scale[3]
    }
    bc_info = ("basket_cell", mb, selection.basket_cells, pf_bc_conn, bc_colors, bc_radius)
    sc_info = ("stellate_cell", ms, selection.stellate_cells, pf_sc_conn, sc_colors, sc_radius)
    figs = {}
    for key, m, select, set, colors, radius in (bc_info, sc_info):
        ps = network.get_placement_set(key)
        name = key.split("_")[0]
        for label, id in select.items():
            offset = ps.positions[ps.identifiers == id][0]
            fig = plot_morphology(m, show=False, color=colors, soma_radius=radius, offset=offset)
            fig.update_layout(title_text=f"{label} {name} cell")
            positions = [[] for _ in range(5)]
            for intersection in set.intersections:
                if intersection.to_id == id:
                    count = grc_count_map[intersection.from_id]
                    positions[count].append(intersection.to_compartment.midpoint)
            for count in range(5):
                pos = np.array(positions[count])
                if not len(pos):
                    pos = np.empty((0,3))
                t = go.Scatter3d(
                    x=pos[:,0] + offset[0],
                    y=pos[:,2] + offset[2],
                    z=pos[:,1] + offset[1],
                    mode="markers",
                    opacity=marker_opacity[count],
                    marker=dict(
                        size=marker_sizes[count],
                        color=grc_color,
                    ),
                    name=f"Granule cell PF synapses with {count} active dendrites"
                )
                fig.add_trace(t)
            cfg = selection.btn_config.copy()
            cfg["filename"] = label[0] + "_" + key + "_synapses"
            figs[f"{key[:2]}_{label[0]}"] = fig
    return figs

if __name__ == "__main__":
    plot()
