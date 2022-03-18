import networkx as nx
from bsb.core import from_hdf5
from scipy.sparse import coo_matrix
import numpy as np
import itertools
import plotly.graph_objs as go

def goc_graph(netw):
    G = nx.DiGraph()
    goc = netw.get_placement_set("golgi_cell")
    gap_goc = netw.get_connectivity_set("gap_goc")
    l = len(gap_goc)
    conn_m = coo_matrix((np.ones(l), (gap_goc.from_identifiers, gap_goc.to_identifiers)), shape=(l, l))
    conn_m.eliminate_zeros()
    G.add_nodes_from(goc.identifiers)
    itertools.consume(G.add_edges_from(zip(itertools.repeat(from_), row.indices, map(lambda a: dict([a]), map(tuple, zip(itertools.repeat("weight"), row.data))))) for from_, row in enumerate(map(conn_m.getrow, range(len(goc)))))
    return G

def graph_traces(G, pos):
    id_map = dict(zip(G.nodes(), itertools.count()))
    edge_x = []
    edge_y = []
    edge_traces = []
    for edge in G.edges():
        x0, y0 = pos[id_map[edge[0]]]
        x1, y1 = pos[id_map[edge[1]]]
        edge_traces.append(go.Scatter(
            x=[x0, x1], y=[y0, y1],
            line=dict(width=G.edges[edge[0], edge[1]]["weight"] / 5, color='#888'),
            hoverinfo='none',
            mode='lines'
        ))

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[id_map[node]]
        node_x.append(x)
        node_y.append(y)
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo="skip",
        text=list(G.nodes()),
        textposition="middle center",
        marker=dict(
            color="blue",
            size=20,
            line_width=2
        ),
        textfont=dict(
            size=12,
            color="white"
        )
    )

    return [*edge_traces, node_trace]

def plot():
    netw = from_hdf5("networks/balanced.hdf5")
    G = goc_graph(netw)
    pos_traces = graph_traces(G, netw.get_placement_set("golgi_cell").positions[:, [0, 2]])
    spring_traces = graph_traces(G, nx.spring_layout(G))
    return {
        "pos": go.Figure(
            data=pos_traces,
            layout=go.Layout(
                title='Golgi gap network - positions',
                titlefont_size=16,
                showlegend=False,
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
        ),
        "layout": go.Figure(
            data=spring_traces,
            layout=go.Layout(
                title='Golgi gap network - force directed',
                titlefont_size=16,
                showlegend=False,
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
        ),
    ]
