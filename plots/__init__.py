def make_3dsubplots(rows, cols):
    return {"specs": make_specs(rows, cols), "rows": rows, "cols": cols}


def make_specs(rows, cols):
    return [[{"type": "scene"} for _ in range(cols)] for _ in range(rows)]
