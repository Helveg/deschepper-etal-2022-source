import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "plots"))
from plots.grc_activation_overlap import plot as act_nmi
from plots.grc_latency_overlap import plot as lat_nmi
from plots.grc_calcium_overlap import plot as ca_nmi


def table():
    table = [["experiment", "control", "gabazine"]]
    table.append(("activity", *act_nmi(ret_nmi=True)))
    table.append(("latency", *lat_nmi(ret_nmi=True)))
    table.append(("calcium", *ca_nmi(ret_nmi=True)))
    return table
