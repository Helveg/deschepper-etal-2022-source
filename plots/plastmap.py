import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, "tables")

import pop_freq
import glob
from plots._paths import *
import re
import pickle
import plotly.graph_objs as go
import numpy as np
import h5py
import selection
from collections import defaultdict
from bsb.core import from_hdf5

def find(name):
    def finder(a):
        return a[0] == name

    return finder

def get_freq(name, all_):
    return [*map(lambda x: x[1], filter(find(name), all_))][0]

def get_both(name, all_):
    return [*map(lambda x: x[1:], filter(find(name), all_))][0]

def parse_path(path):
    return ((re.search("pfPC_(\d+)", path) or [None])[0], (re.search("pfMLI_(\d+)", path) or [None])[0])

def cell_freq(path, cells):
    freq = []
    with h5py.File(path, "r") as h:
        for c in cells:
            t = h[f"recorders/soma_spikes/{c}"][:, 1]
            freq.append(np.sum((t > 6000) & (t < (6000 + 160))) * (1000 / 160))
    return (np.mean(freq), np.std(freq))

def plot():
    paths = glob.glob(results_path("plast", "*.hdf5"))
    base_path = next(path for path in paths if "base" in path)
    paths.remove(base_path)
    net = from_hdf5(f"networks/{selection.network}")
    pc = selection.onbeam["purkinje_cell"]
    pc = net.get_placement_set("purkinje_cell").identifiers
    sc = net.get_placement_set("stellate_cell").identifiers
    bc = net.get_placement_set("basket_cell").identifiers
    figs = {}
    for tag, cells in (("pc", pc), ("sc", sc), ("bc", bc)):
        base_freq = cell_freq(base_path, cells)[0]
        freqs = defaultdict(list)
        for path in paths:
            parsed = parse_path(path)
            print("Processing", parsed)
            freqs[parsed].append(cell_freq(path, cells))
        freqs = {((re.search("\d+", str(pc)) or [None])[0], (re.search("\d+", str(mli)) or [None])[0]): v for (pc, mli), v in freqs.items()}
        heatmap = np.full((6, 6), np.nan)
        for (pc, mli), data in freqs.items():
            freq = np.mean([f[0] for f in data])
            std = np.mean([f[1] for f in data])
            if mli is not None and pc is not None:
                print("PC: ", pc)
                print("MLI:", mli)
                print("F:  ", freq, "⨦", std)
                print("+%: ", round(freq / base_freq, 2))
                heatmap[int(mli), int(pc)] = freq / base_freq
        figs[tag] = go.Figure(
            go.Heatmap(z=heatmap, colorbar_title="Pop. freq delta"),
            layout=dict(
                title_text=f"Plastic changes in {tag} firing",
                xaxis=dict(
                    title="pfPC LTD (AMPA-U downreg.)",
                    tickmode="array",
                    tickvals=list(range(6)),
                    ticktext=[f"{15 * i}%" for i in range(6)],
                ),
                yaxis=dict(
                    title="pfMLI LTP (AMPA-U+NMDA-U upreg. -> GABA)",
                    tickmode="array",
                    tickvals=list(range(6)),
                    ticktext=[f"{15 * i}%" for i in range(6)],
                ),
            ),
        )

    return figs
