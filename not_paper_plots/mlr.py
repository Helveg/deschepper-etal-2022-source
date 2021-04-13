# Multiple Linear Regression tests

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objs as go
X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y = np.dot(X, np.array([1, 2])) + 3
forest = RandomForestRegressor(200).fit(X, y)
reg = LinearRegression().fit(X, y)
print(reg.score(X, y))
print(forest.score(X, y))
print("linear", reg.coef_, reg.intercept_)
go.Figure([
    go.Scatter3d(x=X[:,0], y=X[:,1], z=y),
    go.Scatter3d(x=X[:,0], y=X[:,1], z=reg.predict(X)),
    go.Scatter3d(x=X[:,0], y=X[:,1], z=forest.predict(X)),
], layout_scene=dict(xaxis_range=[0,15], yaxis_range=[0, 15], zaxis_range=[0, 15])).show()
