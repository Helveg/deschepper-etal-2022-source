from .grc_oscillations import plot as plot_grc_osc
from .goc_oscillations import plot as plot_goc_osc
from plotly.subplots import make_subplots

def plot():
    fig_grc = plot_grc_osc()
    fig_goc = plot_goc_osc()
    traces = fig_grc.data + fig_goc.data
    fig = make_subplots(rows=len(traces), cols=1, y_title="Power [Hz<sup>2</sup>]", x_title="Frequency [Hz]", shared_xaxes=True)
    for i, t in enumerate(traces):
        fig.add_trace(t, row=i+1, col=1)
    fig.update_yaxes(range=[0, 30000], row=1, col=1)
    fig.update_yaxes(range=[0, 30000], row=2, col=1)
    fig.update_yaxes(range=[0, 25], row=3, col=1)
    fig.update_yaxes(range=[0, 25], row=4, col=1)
    fig.update_layout(showlegend=False)
    return fig

def meta():
    return {"width": 1920 / 4 * 0.85, "height": 1920 / 2 * 0.62}
