from neuro3d.render import Renderer
from neuro3d._blender.render import BlenderRender

r = Renderer(BlenderRender, "linear_scale10.blend", 4)
r.render_portions()
