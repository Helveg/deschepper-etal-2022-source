import numpy as np
import plotly.graph_objs as go
from scipy.optimize import curve_fit

def gauss_nd(coords, A, μ, σ):
    return A / (np.sqrt(2 * np.pi) * np.prod(σ)) * np.exp(-np.sum(np.power((coords - μ) / σ, 2)) / 2)

def mf_model(coords, A, μ_x, μ_y, σ_x, σ_y):
    return gauss_nd(coords, A, np.array([μ_x, μ_y]), np.array([σ_x, σ_y]))

vgauss = np.vectorize(gauss_nd, excluded=set([1, 2, 3]))
vmf = np.vectorize(mf_model, excluded=set([1, 2, 3, 4, 5]))

shape = (100, 100)
idx = np.indices(shape).reshape(2, -1).T / 10
x_data = np.arange(0, 10, 0.01)
print(vgauss(x_data, 2, np.array([5]), np.array([1.2])))
params, cov = curve_fit(vmf, x_data, vmf(x_data, 2, 5, 5, 0.4, 1.2), p0=[5, 4, 6, 0.2, 1.9], bounds=([0, 0, 0, 0, 0], [10, 10, 100, 10, 10]))
print("Did we do it?", params, cov)
go.Figure(
    [
        go.Scatter(x=x_data, y=vgauss(x_data, 2, np.array([5, 5]), np.array([0.4, 1.2])), name="nd_gauss", mode="lines"),
        go.Scatter(x=x_data, y=vmf(x_data, 2, 5, 5, 0.4, 1.2), name="mf_model", mode="lines"),
        go.Scatter(x=x_data, y=vmf(x_data, *[5.60631159, 4.89837876, 6.98210583, 0.38907941, 1.71834128]), name="fitted", mode="lines"),
    ]
).show()
res = np.array([gauss_nd(pair, 4, np.array([5, 5]), np.array([1, 3])) for pair in idx]).reshape(shape)
print(res.shape)
go.Figure(go.Surface(z=res)).show()
