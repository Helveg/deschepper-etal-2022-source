from bsb.core import from_hdf5
import plotly.graph_objs as go
import os, _layouts
import selection, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "plots"))
from _paths import *
import numpy as np

individually_labelled = True
skip_selected = False

def plot(net_path=None):
    selected_mf = {
        0: [192, 208, 215, 231],
        1: [212, 246, 171, 185],
        2: [231, 254, 272, 276],
        3: [195, 271, 213, 182],
        4: [268, 198, 184, 252],
        5: [195, 194, 170, 199],
        6: [192, 187, 175, 273],
        7: [185, 279, 206, 172, 173],
        8: [240, 241, 274, 186],
        9: [221, 198, 188, 249]
    }
    figs = {}
    for k, selection_mf in selected_mf.items():
        to_select = len(selection_mf) < 4
        if not to_select and skip_selected:
            continue
        net_path = f"networks/batch_1/network_{k}.hdf5"
        scaffold = from_hdf5(net_path)
        cs = scaffold.get_connectivity_set("mossy_to_glomerulus")
        data = {id: [] for id in cs.from_identifiers}
        for conn in cs.connections:
            data[conn.from_id].append(conn.to_id)

        ps = scaffold.get_placement_set("glomerulus")
        pos_map = {c.id: c.position for c in ps.cells}

        axis_labels = dict(xaxis_title="X", yaxis_title="Z", zaxis_title="Y")

        selection_name = k
        fig = go.Figure()
        fig.update_layout(scene=axis_labels)
        traces = {}

        show_first_legend = [True, True]
        get_glom_pos = np.vectorize(pos_map.get, signature='()->(3)')
        for mf, gloms in data.items():
            _mean = get_glom_pos(gloms).mean(axis=0)
            if to_select and _mean[0] > 135 and _mean[0] < 165 and _mean[2] > 70 and _mean[2] < 130:
                selection_mf.append(mf)
            extras = {}
            if mf not in selection_mf:
                extras["marker"] = dict(color="gray", opacity=0.2)
            if individually_labelled:
                extras["name"] = "MF " + str(mf)
            else:
                extras["name"] = f"{'S' if mf in selected_mf else 'Uns'}timulated glomeruli"
                extras["legendgroup"] = "active" if mf in selected_mf else "inactive"
                extras["showlegend"] = show_first_legend[mf in selected_mf]
                show_first_legend[mf in selected_mf] = False
            traces[mf] = go.Scatter3d(
                x=[pos_map[id][0] for id in gloms],
                y=[pos_map[id][2] for id in gloms],
                z=[pos_map[id][1] for id in gloms],
                mode="markers",
                **extras,
            )
        for k in sorted(traces.keys()):
            fig.add_trace(traces[k])
        fig.update_layout(
            scene=dict(
                xaxis=dict(title="X", range=[0, 300], autorange=False),
                yaxis=dict(title="Z", range=[0, 200], autorange=False),
                zaxis=dict(title="Y", range=[0, 130], autorange=False),
                aspectratio=dict(x=1, y=2/3, z=13/30 ),
                camera=_layouts.struct_activation_cam
            ),
            legend=dict(
                traceorder="normal",
            )
        )
        figs[selection_name] = fig
        print(f"network_{selection_name}:", selection_mf)

    return figs
