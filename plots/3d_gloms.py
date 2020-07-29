from bsb.core import from_hdf5
import plotly.graph_objs as go
import os

network_path = os.path.join(
    os.path.dirname(__file__), "..", "networks", "results.hdf5"
)

def results_path(*args):
    return os.path.join(
        os.path.dirname(__file__), "..", "results", *args
    )

def plot():
    scaffold = from_hdf5(network_path)
    selected_mf = [229,213, 221, 228]
    cs = scaffold.get_connectivity_set("mossy_to_glomerulus")
    fig = go.Figure()
    data = {id: [] for id in cs.from_identifiers}
    for conn in cs.connections:
        data[conn.from_id].append(conn.to_id)

    ps = scaffold.get_placement_set("glomerulus")
    pos_map = {c.id: c.position for c in ps.cells}

    out_x = []
    out_y = []
    out_z = []
    for mf, gloms in [(k, v) for k, v in data.items() if k not in selected_mf]:
        out_x.extend([pos_map[id][0] for id in gloms])
        out_y.extend([pos_map[id][2] for id in gloms])
        out_z.extend([pos_map[id][1] for id in gloms])

    fig.add_trace(go.Scatter3d(x=out_x, y=out_y, z=out_z, name="glomeruli", mode="markers", marker=dict(color="gray", opacity=0.2)))
    axis_labels = dict(xaxis_title="X", yaxis_title="Z", zaxis_title="Y")
    fig.update_layout(scene=axis_labels)

    for mf, gloms in [(k, v) for k, v in data.items() if k in selected_mf]:
        fig.add_trace(go.Scatter3d(x=[pos_map[id][0] for id in gloms], y=[pos_map[id][2] for id in gloms], z=[pos_map[id][1] for id in gloms], name="MF " + str(mf), mode="markers"))

    fig.layout.scene.xaxis.range = [0, 300]
    fig.layout.scene.yaxis.range = [0, 200]
    fig.layout.scene.zaxis.range = [0, 150]
    return fig
