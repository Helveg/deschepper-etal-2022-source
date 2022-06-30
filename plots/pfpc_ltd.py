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
from plotly.subplots import make_subplots

start = 6000
dur = 160

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

def cell_data(path, cells):
    freq = []
    spike = []
    with h5py.File(path, "r") as h:
        for c in cells:
            t = h[f"recorders/soma_spikes/{c}"][:, 1]
            s = t[(t > start - dur * 2) & (t < (start + dur * 2))]
            spike.extend(s)
            freq_during = np.sum((t > start) & (t < (start + dur)))
            freq_base = np.sum((t > start - dur * 2) & (t < start))
            freq_during *=  (1000 / dur)
            freq_base *=  (500 / dur)
            freq.append(freq_during - freq_base)
    return freq, spike

def plot():
    paths = glob.glob(results_path("plast", "*.hdf5"))
    net = from_hdf5(f"networks/{selection.network}")
    pc = selection.onbeam["purkinje_cell"]
    pc = net.get_placement_set("purkinje_cell").identifiers
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("Conditioning protocol", "Purkinje cell modulation by pfPC LTD"),
        specs=[[{}, {"rowspan": 2}], [{}, None]]
    )
    for tag, cells in (("pc", pc),):
        freqs = defaultdict(list)
        spikes = defaultdict(list)
        nums = defaultdict(int)
        for path in paths:
            parsed = parse_path(path)
            parsed = (int((re.search("\d+", str(parsed[0])) or [None])[0]), int((re.search("\d+", str(parsed[1])) or [None])[0]))
            print("Processing", parsed)
            freq, spike = cell_data(path, cells)
            freqs[parsed].extend(freq)
            spikes[parsed].extend(spike)
            nums[parsed] += len(cells)
        trace = np.full((11, 2), np.nan)
        base_mean, base_var = np.mean(freqs[(0, 0)]), np.var(freqs[(0, 0)])
        print("F:  ", base_mean, "+-", np.sqrt(base_var))
        for (pc, mli), data in freqs.items():
            state_mean, state_var = np.mean(data), np.var(data)
            if mli is not None and pc is not None:
                print("PC: ", pc)
                print("MLI:", mli)
                print("F:  ", state_mean, "+-", np.sqrt(state_var))
                ratio = state_mean / base_mean
                # s² = µ²A/µ²B * (s²A/µ²A + s²B/µ²B)
                ratio_var = (state_mean ** 2 / base_mean ** 2) * (state_var ** 2 / state_mean ** 2 + base_var ** 2 / base_mean ** 2)
                ratio_sem = np.sqrt(ratio_var) / np.sqrt(len(data))
                print("+%: ", round(ratio, 2), "+-", ratio_sem)
                trace[pc] = ratio, ratio_sem
        fig.add_trace(
            go.Scatter(
                y=trace[:, 0],
                mode="markers",
                marker_size=15,
                line_color=net.configuration.cell_types["purkinje_cell"].plotting.color,
                error_y=dict(
                    type="data",
                    array=trace[:, 1],
                ),
            ),
            row=1,
            col=2,
        )
        fig.update_layout(
            title_text=f"pf-Purkinje plasticity",
        )
        fig.update_xaxes(
            title="pf-PC LTD (%)",
            tickmode="array",
            tickvals=list(range(11)),
            ticktext=[f"{round(i * 70 / 10)}%" for i in range(11)],
            row=1,
            col=2,
        )
        fig.update_yaxes(
            title="Purkinje firing rate modulation",
            row=1,
            col=2,
        )
        norm_counts_0 = np.random.random(len(spikes[(0, 0)])) < (nums[(5, 0)] / nums[(0, 0)])
        spikes_0 = np.array(spikes[(0, 0)])[norm_counts_0]
        norm_counts_9 = np.random.random(len(spikes[(5, 0)])) < (nums[(0, 0)] / nums[(5, 0)])
        spikes_9 = np.array(spikes[(5, 0)])[norm_counts_9]
        pre0 = np.sum(spikes_0 < start) / 2
        dur0 = np.sum((spikes_0 >= start) & (spikes_0 < start + dur + 20))
        post0 = np.sum(spikes_0 >= start + dur + 20)
        start2 = start + 1000
        pre9 = np.sum(spikes_9 < start) / 2
        dur9 = np.sum((spikes_9 >= start) & (spikes_9 < start + dur + 20))
        post9 = np.sum(spikes_9 >= start + dur + 20)
        for trace in [
            go.Scatter(mode="lines", line_color="black", x=[start - dur * 2, start, start, start + dur, start + dur, start + dur * 3], y=[0, 0, dur0 - pre0, dur0 - pre0, post0 - pre0, post0 - pre0]),
            go.Scatter(mode="lines", line_color="black", x=[start2 - dur * 2, start2, start2, start2 + dur, start2 + dur, start2 + dur * 3], y=[0, 0, dur9 - pre9, dur9 - pre9, post9 - pre9, post9 - pre9])
        ]:
            fig.add_trace(trace, row=2, col=1)
        fig.update_yaxes(zeroline=True, row=2, col=1)

    return fig
