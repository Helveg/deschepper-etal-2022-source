import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "plots"))

from plots.pc_bp_pstt import plot as bp_calc

def table():
    table = [["bg_r2", "bg_coeff", "bp_r", "bp_p", "mli_r", "mli_p", "nmi_bp", "nmi_mli"]]
    table.append(bp_calc(ret_corr=True))

    return table
