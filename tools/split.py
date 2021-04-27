# python split.py input.hdf5 out1.hdf 0 4000 out2.hdf5 4000 8000 0.025

import os, sys, h5py, traceback, numpy as np

def partial_copy_all(o, n, g, sets, l, trans_short, trans_long):
    print("Copying", g)
    ng = n.require_group(g)
    for k, v in o[g].attrs.items():
        ng.attrs[k] = v
    for i, (key, og) in enumerate(sets.items()):
        if i // 100 == i / 100:
            print("Copying", f"{i}/{len(sets)}", end="\r", flush=True)
        od = og[()]
        if abs(l - od.shape[0]) < 5:
            data = trans_long(od)
        else:
            data = trans_short(od)
        ds = ng.create_dataset(key, data=data)
        for k, v in og.attrs.items():
            ds.attrs[k] = v

def copy_all_recursive(o, n, r, *args, **kwargs):
    def visit_group(name, obj):
        if not isinstance(obj, h5py.Group):
            return None
        sets = {k.split("/")[-1]: v for k, v in obj.items() if isinstance(v, h5py.Dataset)}
        partial_copy_all(o, n, name, sets, *args, **kwargs)

    or_ = o[r]
    # Copy root attrs
    ng = n.require_group(or_.name)
    for k, v in or_.attrs.items():
        ng.attrs[k] = v

    # Copy children
    or_.visititems(visit_group)

if __name__ == "__main__":

    dt = None
    if len(sys.argv) > 1:
        print("Raw input:", sys.argv, flush=True)
        inp = sys.argv[1]
        outputs = sys.argv[2:-1:3]
        starts = [float(i) for i in sys.argv[3::3]]
        stops = [float(i) for i in sys.argv[4::3]]
        dt = float(sys.argv[-1])
    else:
        while True:
            try:
                inp = input("Input file? ")
                outputs, starts, stops = [], [], []
                while (outp := input("Output file (empty to skip): ")):
                    outputs.append(outp)
                    starts.append(float(input("Start time: ")))
                    stops.append(float(input("Stop time: ")))
            except Exception as e:
                traceback.print_exc()
            else:
                break
    print("-- overview --")
    print("Input file:", inp)
    print("Output files:", outputs)
    print("Start times:", starts)
    print("Stop times:", stops)
    print("-- file properties --")
    f = h5py.File(inp, "r")
    print("Has spikes:", (spikes := "/recorders/soma_spikes" in f))
    print("Has voltages:", (voltages := "/recorders/soma_voltages" in f))
    print("Has all:", (all := "/all" in f))
    print("--", flush=True)
    try:
        time = f["/time"][()]
        if not len(time) and (voltages or all):
            if dt is not None:
                print("No time vector found, dt given:", dt)
            else:
                dt = float(input("No time vector found, give dt: "))
            check = f["/all"] if all else f["/recorders/soma_voltages"]
            time = np.arange(0, len(next(iter(check.values()))[()])) * dt
        print("Starting copy operation", flush=True)
        for outp, start, stop in zip(outputs, starts, stops):
            path = os.path.abspath(os.path.dirname(outp))
            print("creating path:", path)
            os.makedirs(path, exist_ok=True)
            crop = (start <= time) & (time < stop)
            transformers = (lambda x: x[(start <= x[:, 1]) & (x[:, 1] <= stop), :] if len(x.shape) == 2 else x[(start <= x) & (x <= stop)], lambda x: x[crop])
            print("cropping time vector:", len(crop), sum(crop))
            with h5py.File(outp, "w") as nf:
                copy_all_recursive(f, nf, "/", len(time), *transformers)
    finally:
        f.close()
