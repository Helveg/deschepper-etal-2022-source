import numpy as np, scipy.ndimage
import plotly.graph_objs as go
from plotly.subplots import make_subplots

def plot():
    figs = {}

    # Calculation methods
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True)
    fig.update_layout(title_text="Review of E/I balance methods in complete absence of center-surround inhibition")
    one_side = np.concatenate((np.zeros(10000), np.linspace(0, 4, 1000)))
    gabazine = np.concatenate((one_side, np.flip(one_side)))
    fig.add_scatter(y=gabazine, name="gabazine")
    one_side_inh = np.concatenate((np.zeros(9000), np.ones(2000) * 1))
    inh = np.concatenate((one_side_inh, np.flip(one_side_inh)))
    control = np.maximum(gabazine - inh, 0)
    fig.add_scatter(y=inh, row=2, col=1, name="true_inhibition")
    fig.add_scatter(y=control, name="control")
    fig.update_yaxes(range=[-4, 4])
    E = control
    nE = E / abs(E).max()
    I = gabazine - control
    nI = I / abs(I).max()
    pE = nE
    pI = (ng := gabazine / abs(gabazine).max()) - (nc := control / abs(control).max())

    fig.add_scatter(y=ng, row=2, col=1, name="gaba norm")
    fig.add_scatter(y=nc, row=2, col=1, name="c norm")

    # B = (E - I) / E
    fig.add_scatter(y=(E - I) / E, row=3, col=1, name="classical balance")
    # B = (E - I) / (E + 1)
    fig.add_scatter(y=(E - I) / (E + 1), row=3, col=1, name="divzero-corr balance")
    # B = (E - I)
    fig.add_scatter(y=(E - I), row=3, col=1, name="EI diff")
    # B = (nE - nI)
    fig.add_scatter(y=(nE - nI), row=3, col=1, name="afternorm. EI diff")
    fig.add_scatter(y=(pE - pI), row=3, col=1, name="prenorm. EI diff")

    fig.update_yaxes(range=[-10, 10], row=3, col=1)
    figs["bad_formulas"] = fig

    # Mapelli fraud plot

    from scipy.interpolate import interp1d
    i = [(0, 147), (18, 140), (32, 137), (51, 130), (59, 124), (65, 121), (74, 113), (82, 105), (100, 81), (117, 59), (131, 48), (141, 42), (154, 40), (160, 39), (168, 40), (184, 42), (195, 40), (207, 36), (225, 20), (232, 14), (244, 4), (258, 0), (268, 3), (277, 12), (291, 36), (315, 87), (323, 102), (334, 117), (351, 131), (363, 137), (388, 144), (398, 148)]
    ix = np.array([x[0] for x in i])
    iy = np.array([x[1] for x in i])
    e = [(5, 148), (11, 145), (26, 139), (42, 134), (55, 129), (72, 123), (99, 113), (115, 106), (130, 96), (138, 89), (151, 70), (162, 50), (168, 40), (178, 23), (187, 12), (195, 3), (206, 0), (218, 4), (226, 10), (232, 14), (250, 35), (269, 53), (280, 64), (293, 79), (310, 97), (322, 109), (334, 117), (349, 129), (363, 136), (376, 140), (389, 145), (395, 148)]
    ex = np.array([x[0] for x in e])
    ey = np.array([x[1] for x in e])

    def coord_x(inp):
        return inp * 0.75

    def coord_y(inp):
        return (150 - inp) / 150

    interpI = interp1d(coord_x(ix), coord_y(iy), kind="cubic")
    interpE = interp1d(coord_x(ex), coord_y(ey), kind="cubic")
    smooth_x = np.arange(5 * 0.75, 395 * 0.75, 0.1)

    fig = go.Figure([
        go.Scatter(x=smooth_x, y=interpI(smooth_x), name="inhibition", line=dict(width=4, color="red")),
        go.Scatter(x=smooth_x, y=interpE(smooth_x), name="excitation", line=dict(width=4, color="black")),
        go.Scatter(x=smooth_x, y=(interpE(smooth_x) - interpI(smooth_x)), name="errata", line=dict(width=4, color="black", dash='dash')),
        go.Scatter(x=smooth_x, y=(interpE(smooth_x) - interpI(smooth_x)) / interpE(smooth_x), name="real result", line=dict(width=4, color="black", dash='dash')),
    ])
    figs["mapelli_fraud"] = fig

    # Casali & Tognolina 2020
    surfaces = {}
    for f in ("G", "C"):
        s = np.loadtxt(f"results/cs_casali/{f}.txt", delimiter=",")
        s[np.isnan(s)] = -70
        sigma = [2, 2]
        s = scipy.ndimage.filters.gaussian_filter(s, sigma)
        surfaces[f] = s
        surfaces[f"n{f}"] = (s + 70) / np.max(s + 70)

    g, c = surfaces["G"], surfaces["C"]
    ng, nc = surfaces["nG"], surfaces["nC"]

    fig = go.Figure([
        go.Surface(z=c, name="C", showlegend=True, visible="legendonly"),
        go.Surface(z=g, name="G", showlegend=True, visible="legendonly"),
        go.Surface(z=nc, name="C norm", showlegend=True, visible="legendonly"),
        go.Surface(z=ng, name="G norm", showlegend=True, visible="legendonly"),
        go.Surface(z=g - c, name="inhibition", showlegend=True),
    ])
    fig.update_traces(showscale=False)
    figs["casali"] = fig

    return figs
