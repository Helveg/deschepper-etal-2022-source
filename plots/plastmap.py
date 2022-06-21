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

def plot():
    paths = glob.glob(results_path("plast", "*.hdf5"))
    base_path = next(path for path in paths if "base" in path)
    paths.remove(base_path)
    if not os.path.exists("plast_freq.pkl"):
        with open("plast_freq.pkl", "wb") as f:
            base_freqs = pop_freq.table(base_path, start=6000, end=6100)
            print(base_freqs)
            base_freq = get_freq("purkinje_cell", base_freqs)
            freqs = {parse_path(path): get_both("purkinje_cell", pop_freq.table(path, start=6000, end=6100)) for path in paths}
            pickle.dump((base_freq, freqs), f)
    else:
        with open("plast_freq.pkl", "rb+") as f:
            base_freq, freqs = pickle.load(f)
            freqs = {((re.search("\d+", str(pc)) or [None])[0], (re.search("\d+", str(mli)) or [None])[0]): v for (pc, mli), v in freqs.items()}
            # pickle.dump((base_freq, freqs), f)
    print(base_freq)
    mli_x = []
    mli_y = []
    pc_x = []
    pc_y = []
    for (pc, mli), (freq, std) in freqs.items():
        print("PC: ", pc)
        print("MLI:", mli)
        print("F:  ", freq, "⨦", std)
        print("+%: ", round(freq / base_freq, 2))
        if mli is not None:
            mli_x.append(int(mli))
            mli_y.append(freq / base_freq)
        else:
            pc_x.append(int(pc))
            pc_y.append(freq / base_freq)
    xs = np.argsort(mli_x)
    mli_x = np.array(mli_x)[xs]
    mli_y = np.array(mli_y)[xs]
    xs = np.argsort(pc_x)
    pc_x = np.array(pc_x)[xs]
    pc_y = np.array(pc_y)[xs]
    return go.Figure([go.Scatter(x=mli_x, y=mli_y, name="pfMLI"), go.Scatter(x=pc_x, y=pc_y, name="pfPC")])
