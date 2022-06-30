import os, sys

sys.path.insert(0, "plots")
import goc_network_graph, goc_nspos, goc_nsync, goc_sync_hist, goc_sync_dist
from plotly.subplots import make_subplots

def transfer(exfig, fig, row, col):
    for trace in exfig:
        fig.add_trace(trace, row=row, col=col)

def plot():
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("Electrotonic coupling", "Spike coincidence", "Temporal relation", "Spatial relation"),
        vertical_spacing=0.05,
        horizontal_spacing=0.05,
    )
    fig.update_layout(title_text="Golgi millisecond synchrony")

    sync_dist = goc_sync_dist.plot()
    npos = goc_network_graph.plot()["pos"]
    surf = goc_nspos.plot()["surf"]
    surf.data[0].colorbar = dict(len=0.5, y=0.25)
    sync_hist = goc_sync_hist.plot()
    nsync = goc_nsync.plot()
    transfer(sync_dist.data[5:10], fig, 1, 1)
    transfer(nsync.data, fig, 2, 1)
    fig.update_traces(showlegend=False, row=2, col=1)
    transfer(sync_hist.data, fig, 1, 2)
    transfer(surf.data, fig, 2, 2)
    return fig
