from bsb.core import from_hdf5
import h5py, os, sys, numpy as np, plotly.graph_objs as go
from scipy import stats
import statistics
_bounds = dict(min=700, max=800)

reps = [
    ('Contacted Granule cell', 120),
    # ('Purkinje cell PF + AA stimulation', 120),
    # ('Purkinje cell PF stimulation', 135),
    # ('Purkinje cell unstimulated', 121),
    # ('Golgi cell PF + AA stimulation', 37),
    # ('Golgi cell PF stimulation', 63),
    # ('Golgi cell unstimulated', 11),
    # ('Stellate cell PF stimulation', 507),
    # ('Stellate cell unstimulated', 656),
    # ('Basket cell PF stimulation', 289),
    # ('Basket cell unstimulated', 405),
]

cutoff = 400

def chain(*iters):
    for iter in iters:
        while True:
            try:
                yield next(iter)
            except StopIteration:
                break

def _copy(x, ds, transfer, **attrs):
    print("copying", ds.name, "to", transfer)
    data = np.array(ds[()])[ds[:, 0] - 400 > 0]
    data[:, 0] -= 400
    d = x[transfer].create_dataset(ds.name.split("/")[-1], data=data)
    d_attrs_kv = iter(ds.attrs.items())
    if attrs is not None:
        d_attrs_kv = chain(d_attrs_kv, iter(attrs.items()))
    for k, v in d_attrs_kv:
        d.attrs[k] = v

scaffold = from_hdf5("networks/results.hdf5")
file = "results/grc/results_poc_1596112751542.hdf5"
results = h5py.File(file, "r")
with h5py.File("selected_reps.hdf5", "a") as x:
    # x.create_group("/somas")
    # x.create_group("/dendrites")
    with h5py.File(file, "r") as f:
        for i, (rep_name, id) in enumerate(reps):
            find_tag = str(int(id))
            name = [key for key in f["/recorders/granules"].keys() if key.startswith(find_tag + ".")][0]
            ds = f["/recorders/granules"][name]
            transfer = "/somas"
            _copy(x, ds, transfer, label=rep_name, display_label=rep_name, order=i)
            # ds = f["/recorders/dendrites"][name]
            transfer = "/dendrites"
            _copy(x, ds, transfer, label=rep_name, display_label=rep_name, order=i)
