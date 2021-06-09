#!/usr/bin/env python3
import os, sys, warnings
from types import ModuleType
from importlib import import_module

# Sets the plotly default theme
import plotly.io as pio
pio.templates.default = "simple_white"

def mod_names():
    return [
        f[:-3]
        for f in os.listdir(os.path.join(path, "plots"))
        if f.endswith(".py") and f != "__init__.py"
    ]

def with_args(plots):
    return [(p.split(":")[0], p.split(":")[1:]) for p in plots]

sys.path.append(os.path.abspath("plots"))
path = os.path.abspath(os.path.dirname(__file__))
plots = mod_names() if len(sys.argv) == 1 else sys.argv[1:]
plots = with_args(plots)

def plot():
    t = len(plots)
    for i, (plot, arg) in enumerate(plots):
        print("Plotting", plot, "... ({}/{})".format(i + 1, t))
        plotting_module = import_module("." + plot, package="plots")
        show_figure(plotting_module, arg)
    print("Done", " " * 30)

def show_figure(plotting_module, arg):
    plt = plotting_module.plot(*arg)
    if not plt:
        warnings.warn(f"No figure returned from {plotting_module.__name__}.")
        return
    if isinstance(plt, list):
        for i, p in enumerate(plt):
            _show_figure(plotting_module, p, i)
    elif isinstance(plt, dict):
        for k, p in plt.items():
            _show_figure(plotting_module, p, k)
    else:
        _show_figure(plotting_module, plt)

def _show_figure(plotting_module, plt, suffix=None):
    plt.show(config={
        'toImageButtonOptions': {
            'format': 'svg',
            'filename': plotting_module.__name__ + (f"_{suffix}" if suffix is not None else ""),
            'height': 1920,
            'width': 1080,
            'scale': 1
        },
        'modeBarButtonsToAdd': [
            'drawline',
            'drawopenpath',
            'drawclosedpath',
            'drawcircle',
            'drawrect',
            'eraseshape'
        ]
    })

if __name__ == "__main__":
    plot()
