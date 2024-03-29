from bsb.core import from_hdf5
from bsb.plotting import plot_morphology, MorphologyScene, get_soma_trace
from plotly import graph_objs as go
import selection, numpy as np
from colour import Color
from ._paths import *
import selection

inset_camera = dict(up=dict(x=0,y=0,z=1),center=dict(x=-0.03919936633492447,y=0.009141280925267406,z=-0.10532701007298666),eye=dict(x=0.05258464298628989,y=3.7406970544611196,z=0.46626663726890044))
main_camera = dict(up=dict(x=0,y=0,z=1),center=dict(x=0.2774248385852698,y=-0.04292027273220392,z=-0.4552177352722382),eye=dict(x=0.14544812231944693,y=0.6122189062165164,z=-0.297573572462025))

def plot(net_path=None):
    if net_path is None:
        net_path = network_path(selection.network)
    network = from_hdf5(net_path)
    mr = network.morphology_repository
    grm = mr.get_morphology("GranuleCell")
    gom = mr.get_morphology("GolgiCell")
    ps_goc = network.get_placement_set("golgi_cell")
    ps_glom = network.get_placement_set("glomerulus")
    ps_grc = network.get_placement_set("granule_cell")

    stim_mf = selection.stimulated_mf_poiss
    cs_glom_grc = network.get_connectivity_set("glomerulus_to_granule")
    cs_glom_goc = network.get_connectivity_set("glomerulus_to_golgi")
    cs_goc_grc = network.get_connectivity_set("golgi_to_granule")

    grc_color = network.configuration.cell_types["granule_cell"].plotting.color
    goc_color = network.configuration.cell_types["golgi_cell"].plotting.color
    glom_radius = network.configuration.cell_types["glomerulus"].placement.soma_radius
    goc_radius = network.configuration.cell_types["golgi_cell"].placement.soma_radius
    grc_radius = network.configuration.cell_types["granule_cell"].placement.soma_radius
    goc_scale = list(map(str, Color(goc_color).range_to("white", 10)))
    goc_colors = {
        "soma": goc_scale[0],
        "basal_dendrites": goc_scale[0],
        "apical_dendrites": "gray",
        "axon": goc_scale[3]
    }

    # Create granule cell inset
    grc_id = ps_grc.identifiers[36]
    inset = plot_morphology(grm, set_range=False, show=False, soma_radius=grc_radius, color=grc_color, use_last_soma_comp=False)
    inset.layout.scene.xaxis.range = [-5, 5]
    inset.layout.scene.yaxis.range = [-5, 5]
    inset.layout.scene.zaxis.range = [-15, 5]
    inset.layout.scene.aspectratio = dict(x=1, y=1, z=2)
    glom_synapses = np.array([i.to_compartment.midpoint for i in cs_glom_grc.intersections if i.to_id == grc_id], dtype=float)
    print("Glom syns:", [i.to_compartment.midpoint for i in cs_glom_grc.intersections if i.to_id == grc_id])
    goc_synapses = np.array([i.to_compartment.midpoint for i in cs_goc_grc.intersections if i.to_id == grc_id], dtype=float)
    print("Goc syns:", [i.to_compartment.midpoint for i in cs_goc_grc.intersections if i.to_id == grc_id])
    print("syn check:", glom_synapses.shape, goc_synapses.shape)
    text_glom = ["Glom1", "Glom3", "Glom2", "Glom4"]
    text_goc = ["GoC1", "GoC2", "GoC1", "GoC3"]
    inset.add_trace(go.Scatter3d(x=goc_synapses[:,0], y=goc_synapses[:,2], z=goc_synapses[:,1], marker=dict(color=goc_color, size=4), mode="markers", showlegend=False, legendgroup="goc"))
    inset.update_layout(
        scene_annotations=[
            dict(
                x=p[0] - 1,
                y=p[2],
                z=p[1] - ay,
                text=t,
                showarrow=False,
                font=dict(family="Arial", size=10),
                textangle=90
            )
            for t, p, ay in zip(text_goc + text_glom, np.concatenate((goc_synapses, glom_synapses)), [-2.2] * len(goc_synapses) + [1.7] * len(glom_synapses))
        ],
        scene_xaxis_visible=False,
        scene_yaxis_visible=False,
        scene_zaxis_visible=False,
    )
    inset.add_trace(go.Scatter3d(x=glom_synapses[:,0], y=glom_synapses[:,2], z=glom_synapses[:,1], marker=dict(color="black", size=3), mode="markers", showlegend=False, legendgroup="goc"))
    inset.update_layout(scene_camera=inset_camera)

    # Create main figure
    glom_id = [i.from_id for i in cs_glom_grc.intersections if i.to_id == grc_id][2]
    goc_id = [i.from_id for i in cs_goc_grc.intersections if i.to_id == grc_id][2]
    other_grcs = np.array([i.to_id for i in cs_glom_grc.intersections if i.from_id == glom_id])
    sorted_grcs = np.sort(other_grcs)
    grc_pos = ps_grc.positions[np.isin(ps_grc.identifiers, other_grcs)]
    goc_pos = ps_goc.positions[ps_goc.identifiers == goc_id][0]
    glom_pos = ps_glom.positions[ps_glom.identifiers == glom_id][0]
    ms = MorphologyScene()
    glom_synapses = []
    active_claws = {}
    glom_comps = []
    for i in cs_glom_grc.intersections:
        if i.from_id == glom_id and i.to_id in other_grcs:
            glom_synapses.append(i.to_compartment.midpoint + grc_pos[np.nonzero(sorted_grcs == i.to_id)[0][0]])
            active_claws[i.to_id] = i.to_compartment.id
    glom_synapses = np.array(glom_synapses, dtype=float)
    goc_synapses = []
    active_goc_synapses = []
    for i in cs_goc_grc.intersections:
        if i.from_id == goc_id and i.to_id in other_grcs:
            pos = i.to_compartment.midpoint + grc_pos[np.nonzero(sorted_grcs == i.to_id)[0][0]]
            goc_synapses.append(pos)
            if abs(i.to_compartment.id - active_claws[i.to_id]) == 1:
                active_goc_synapses.append(pos)
    goc_synapses = np.array(goc_synapses, dtype=float)
    active_goc_synapses = np.array(active_goc_synapses, dtype=float)
    # ms.fig.add_trace(go.Scatter3d(x=goc_synapses[:,0], y=goc_synapses[:,2], z=goc_synapses[:,1], marker=dict(color="blue", size=3.5), opacity=0.5, mode="markers", legendgroup="goc", showlegend=False))
    ms.fig.add_trace(go.Scatter3d(x=active_goc_synapses[:,0], y=active_goc_synapses[:,2], z=active_goc_synapses[:,1], marker=dict(color="blue", size=3.5), mode="markers", name="GoC-GrC synapse", legendgroup="goc"))
    ms.fig.add_trace(go.Scatter3d(x=glom_synapses[:,0], y=glom_synapses[:,2], z=glom_synapses[:,1], marker=dict(color="black", size=3), mode="markers", name="Glom-GrC synapse", legendgroup="glom"))
    print("Depicted GrC:", len(grc_pos))
    for pos in grc_pos:
        ms.add_morphology(grm, offset=pos, soma_radius=grc_radius, color=grc_color, use_last_soma_comp=False, segment_radius={"soma": 1, "axon": 0.2, "dendrites": 2})
    ms.fig.add_trace(get_soma_trace(goc_radius, offset=goc_pos, color=goc_color, name="Golgi cell", showlegend=True))
    ms.fig.add_trace(get_soma_trace(glom_radius, offset=glom_pos, name="Glomerulus", showlegend=True))
    ms.prepare_plot()
    ms.fig.layout.scene.xaxis.range = [0, 230]
    ms.fig.layout.scene.yaxis.range = [20, 250]
    ms.fig.layout.scene.zaxis.range = [-20, 210]
    ms.fig.update_layout(scene_camera=main_camera)
    return {"grc_inset": inset, "main": ms.fig}

def meta(key):
    if key == "grc_inset":
        return {"width": 300, "height": 500}

if __name__ == "__main__":
    for p in plot().values():
        p.show()
