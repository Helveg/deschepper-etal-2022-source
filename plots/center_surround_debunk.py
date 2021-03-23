import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots

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

# Calculation methods

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
fig.write_html("cs.html")
