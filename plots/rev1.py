import os, sys

sys.path.insert(0, "plots")
import goc_network_graph, goc_nspos, goc_nsync, goc_sync_hist
from plotly.subplots import make_subplots

def transfer(exfig, fig, row, col):
    for trace in exfig.data:
        fig.add_trace(trace, row=row, col=col)

def plot():
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("Golgi network", "Spike coincidence", "Temporal relation", "Spatial relation"),
        vertical_spacing=0.05,
        horizontal_spacing=0.05,
    )
    fig.update_layout(title_text="Golgi millisecond synchrony")

    npos = goc_network_graph.plot()["pos"]
    surf = goc_nspos.plot()["surf"]
    surf.data[0].colorbar = dict(len=0.5, y=0.5)
    sync_hist = goc_sync_hist.plot()
    nsync = goc_nsync.plot()
    transfer(npos, fig, 1, 1)
    transfer(nsync, fig, 1, 2)
    transfer(sync_hist, fig, 2, 1)
    transfer(surf, fig, 2, 2)
    return fig
