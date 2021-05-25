import numpy as np
import plotly.graph_objs as go
from scipy.optimize import curve_fit

def gauss_nd(coords, A, μ, σ):
    return A / (np.sqrt(2 * np.pi) * np.prod(σ)) * np.exp(-np.sum(np.power((coords - μ) / σ, 2)) / 2)

def gauss_expand_args(coords, A, *args):
    # Assert even number of args
    assert not (len(args) % 2)
    return gauss_nd(coords, A, np.array(args[:len(args) // 2]), np.array(args[len(args) // 2:]))

def make_scipy_curve(f):
    def curve(M, *args):
        return [f(point, *args) for point in M]

    return curve

shape = (100, 100)
scale = (200, 300)
idx = np.indices(shape).reshape(2, -1).T / shape
print(idx[-1])
single_gaussian_model = make_scipy_curve(gauss_expand_args)
z_data = np.array([gauss_nd(coords, 2, np.array([100, 150]) / scale, np.array([10, 10]) / scale) for coords in idx])
params, cov = curve_fit(single_gaussian_model, idx, z_data, p0=[3, 10, 15, 10, 10], bounds=([0, 0, 0, 0, 0], [10, *scale, 100, 100]))
print("Did we do it?", params[0], params[1], params[2], params[3], params[4])
res = np.array([gauss_expand_args(pair, *params) for pair in idx]).reshape(shape)
z_data.shape = shape
surface_kwargs = dict(x=np.linspace(0, scale[1], shape[1]), y=np.linspace(0, scale[0], shape[0]), showlegend=True)

def predict_activity_kernel(map):
    shape = map.shape
    idx = np.indices(shape).reshape(2, -1).T / shape
    lower_bounds = np.zeros(5)
    upper_bounds = np.ones(5)
    upper_bounds[0] = np.inf
    initial = np.ones(5) / 2
    return curve_fit(single_gaussian_model, idx, map.ravel(), p0=initial, bounds=(lower_bounds, upper_bounds))

params2, cov2 = predict_activity_kernel(z_data)
res_f = np.array([gauss_expand_args(pair, *params2) for pair in idx]).reshape(shape)
print("Confirmed?", params2)

go.Figure(
    [
        go.Surface(z=z_data, opacity=0.3, **surface_kwargs),
        go.Surface(z=res, opacity=0.3, **surface_kwargs),
        go.Surface(z=res_f, opacity=0.3, **surface_kwargs),
    ]
).show()
