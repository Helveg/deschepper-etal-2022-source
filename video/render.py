from neuro3d import set_backend
from neuro3d.render import render

set_backend("blender")
render("with_plots.blend", 4)
