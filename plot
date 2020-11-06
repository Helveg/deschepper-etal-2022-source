#!/usr/bin/env python3
import os, sys, warnings
from types import ModuleType
from importlib import import_module

def mod_names():
    return [
        f[:-3]
        for f in os.listdir(os.path.join(path, "plots"))
        if f.endswith(".py") and f != "__init__.py"
    ]

sys.path.append(os.path.abspath("plots"))
path = os.path.abspath(os.path.dirname(__file__))
plots = mod_names() if len(sys.argv) == 1 else sys.argv[1:]

def plot():
    t = len(plots)
    for i, plot in enumerate(plots):
        print("Plotting", plot, "... ({}/{})".format(i + 1, t))
        plotting_module = import_module("." + plot, package="plots")
        show_figure(plotting_module)
    print("Done", " " * 30)

def show_figure(plotting_module):
    plt = plotting_module.plot()
    if plt:
        plt.show({
            'toImageButtonOptions': {
                'format': 'svg',
                'filename': plotting_module.__name__,
                'height': 1920,
                'width': 1080,
                'scale': 1
            }
        })
    else:
        warnings.warn(f"No figure returned from {plotting_module.__name__}.")

if __name__ == "__main__":
    plot()
