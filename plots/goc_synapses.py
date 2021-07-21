from bsb.core import from_hdf5
from bsb.plotting import plot_morphology #, plot_synapses
from plotly import graph_objs as go
import selection, numpy as np
from colour import Color
from ._paths import *
from glob import glob
import selection

camera = dict(up=dict(x=0,y=0,z=1),center=dict(x=0,y=0,z=0),eye=dict(x=1.5569731921541285,y=1.4381184168348293,z=0.4417577368579583))

def plot(net_path=None):
    if net_path is None:
        net_path = network_path(selection.network)
    network = from_hdf5(net_path)
    mr = network.morphology_repository
    m = mr.get_morphology("GolgiCell")
    ps = network.get_placement_set("golgi_cell")
    ps_grc = network.get_placement_set("granule_cell")

    stim_mf = selection.stimulated_mf_poiss
    cs_mf_glom = network.get_connectivity_set("mossy_to_glomerulus")
    cs_glom_grc = network.get_connectivity_set("glomerulus_to_granule")
    aa_goc_conn = network.get_connectivity_set("ascending_axon_to_golgi")
    pf_goc_conn = network.get_connectivity_set("parallel_fiber_to_golgi")
    gaba = network.get_connectivity_set("golgi_to_golgi")
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
    markers = {"ascending_axon_to_golgi": "circle", "parallel_fiber_to_golgi": "diamond"}
    labels = {"ascending_axon_to_golgi": "AA", "parallel_fiber_to_golgi": "PF"}

    grc_color = network.configuration.cell_types["granule_cell"].plotting.color
    goc_color = network.configuration.cell_types["golgi_cell"].plotting.color
    goc_radius = network.configuration.cell_types["golgi_cell"].placement.soma_radius
    goc_scale = list(map(str, Color(goc_color).range_to("white", 10)))
    goc_colors = {
        "soma": goc_scale[0],
        "basal_dendrites": goc_scale[0],
        "apical_dendrites": goc_scale[1],
        "axon": goc_scale[3]
    }
    tag_colors = {
        "ascending_axon_to_golgi": "#EB52B5",
        "parallel_fiber_to_golgi": grc_color
    }

    figs = {}
    for goc_label, goc_id in selection.golgi_cells.items():
        goc_pos = ps.positions[ps.identifiers == goc_id][0]
        fig = plot_morphology(m, show=False, color=goc_colors, soma_radius=goc_radius, offset=goc_pos)
        fig.update_layout(title_text=f"{goc_label} Golgi cell", scene=dict(
            camera=camera,
            zaxis_dtick=50
        ))
        for set in (aa_goc_conn, pf_goc_conn):
            tag_label = labels[set.tag]
            positions = [[] for _ in range(5)]
            for intersection in set.intersections:
                if intersection.to_id == goc_id:
                    count = grc_count_map[intersection.from_id]
                    positions[count].append(intersection.to_compartment.midpoint)
            for count in range(5):
                pos = np.array(positions[count])
                if not len(pos):
                    pos = np.empty((0,3))
                t = go.Scatter3d(
                    x=pos[:,0] + goc_pos[0],
                    y=pos[:,2] + goc_pos[2],
                    z=pos[:,1] + goc_pos[1],
                    mode="markers",
                    opacity=marker_opacity[count],
                    marker=dict(
                        size=marker_sizes[count],
                        symbol=markers[set.tag],
                        color=tag_colors[set.tag],
                    ),
                    name=f"Granule cell {tag_label} synapses with {count} active dendrites"
                )
                fig.add_trace(t)
        gabas = np.array([i.to_compartment.midpoint for i in gaba.intersections if i.to_id == goc_id])
        fig.add_trace(go.Scatter3d(
            x=gabas[:,0] + goc_pos[0],
            y=gabas[:,2] + goc_pos[2],
            z=gabas[:,1] + goc_pos[1],
            name="Golgi-Golgi GABA synapses",
            mode="markers",
            marker=dict(
                size=1.0,
                color=goc_colors["soma"],
                symbol="diamond",
            )
        ))
        cfg = selection.btn_config.copy()
        cfg["filename"] = goc_label[0] + "_golgi_cell_synapses"
        figs[goc_label[0]] = fig
    return figs

def meta(key):
    return {"width": 1650, "height": 900}

if __name__ == "__main__":
    for p in plot():
        p.show()
