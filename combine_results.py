import os, sys, h5py, glob

files = glob.glob(sys.argv[1])
total = len(files)
groups = sys.argv[2:] or ["recorders/soma_spikes"]
with h5py.File("combined_results.hdf5", "w") as cf:
    offset = 0
    for i, fn in enumerate(files):
        print("Combining {}/{} files".format(i + 1, total), end="\r" if i + 1 < total else "\n")
        with h5py.File(fn, "r") as f:
            for gn in groups:
                cg = cf.require_group(gn)
                g = f[gn]
                for k in g.keys():
                    g.copy(k, cg, name=str(offset))
                    cg[str(offset)].attrs["run_id"] = i
                    offset += 1
