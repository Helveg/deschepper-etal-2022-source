from bsb.core import from_hdf5
import plotly.graph_objs as go
import os

network_path = os.path.join(
    os.path.dirname(__file__), "..", "networks", "300x_200z.hdf5"
)

def results_path(*args):
    return os.path.join(
        os.path.dirname(__file__), "..", "results", *args
    )

individually_labelled = False

def plot():
    scaffold = from_hdf5(network_path)
    selected_mf_4 = [213, 214, 222, 223]
    selected_mf_13 = [213, 214, 222, 223, 229, 221, 206, 230, 247, 239, 231, 238, 240]
    selected_mf_lateral = [275, 261, 260, 263, 273, 267, 279, 265, 268, 280, 270, 269, 272, 274, 282, 264, 271, 266, 283, 276, 278, 281, 277, 284]
    selected_mf = selected_mf_lateral
    cs = scaffold.get_connectivity_set("mossy_to_glomerulus")
    fig = go.Figure()
    data = {id: [] for id in cs.from_identifiers}
    for conn in cs.connections:
        data[conn.from_id].append(conn.to_id)

    ps = scaffold.get_placement_set("glomerulus")
    pos_map = {c.id: c.position for c in ps.cells}

    axis_labels = dict(xaxis_title="X", yaxis_title="Z", zaxis_title="Y")
    fig.update_layout(scene=axis_labels)

    show_first_legend = [True, True]
    for mf, gloms in data.items():
        extras = {}
        if mf not in selected_mf:
            extras["marker"] = dict(color="gray", opacity=0.2)
        if individually_labelled:
            extras["name"] = "MF " + str(mf)
        else:
            extras["name"] = f"{'S' if mf in selected_mf else 'Uns'}timulated glomeruli"
            extras["legendgroup"] = "active" if mf in selected_mf else "inactive"
            extras["showlegend"] = show_first_legend[mf in selected_mf]
            show_first_legend[mf in selected_mf] = False
        fig.add_trace(go.Scatter3d(
            x=[pos_map[id][0] for id in gloms],
            y=[pos_map[id][2] for id in gloms],
            z=[pos_map[id][1] for id in gloms],
            mode="markers",
            **extras,
        ))

    # for mf, gloms in [(k, v) for k, v in data.items() if k not in selected_mf]:
    #     kwargs = {}
    #     if individually_labelled:
    #         kwargs["name"] = "MF " + str(mf)
    #     else:
    #         kwargs["name"] = "Unstimulated glomeruli"
    #         kwargs["legendgroup"] = "inactive"
    #         kwargs["showlegend"] = show_first_legend
    #         show_first_legend = False
    #     fig.add_trace(go.Scatter3d(x=[pos_map[id][0] for id in gloms], y=[pos_map[id][2] for id in gloms], z=[pos_map[id][1] for id in gloms], mode="markers", marker=dict(color="gray", opacity=0.2), **kwargs))

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="X", range=[0, 300], autorange=False),
            yaxis=dict(title="Z", range=[0, 200], autorange=False),
            zaxis=dict(title="Y", range=[0, 130], autorange=False),
            aspectratio=dict(x=1, y=2/3, z=13/30 )
        )
    )
    return fig
