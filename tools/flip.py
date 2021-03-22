import h5py, sys, itertools
from glob import glob

def flip(f):
    for g in f:
        if isinstance(f[g], h5py.Group):
            flip(f[g])
        else:
            ds = f[g][()]
            attrs = dict(f[g].attrs)
            del f[g]
            h = f.create_dataset(g, data=ds.T)
            for k, v in attrs.items():
                h.attrs[k] = v

for f in itertools.chain(*map(glob, sys.argv[1:])):
    print("Flipping", f, " " * 30, end="\r")
    flip(h5py.File(f, "a"))
