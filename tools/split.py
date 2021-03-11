import os, sys, h5py, traceback, numpy as np

def partial_copy_all(o, n, g, transform):
    ng = n.require_group(g)
    t = len(o[g].keys())
    for i, (key, og) in enumerate(o[g].items()):
        if i // 100 == i / 100:
            print("Copying", f"{i}/{t}", end="\r", flush=True)
        data = transform(og[()])
        ds = ng.create_dataset(key, data=data)
        for k, v in og.attrs.items():
            ds.attrs[k] = v

if __name__ == "__main__":

    dt = None
    if len(sys.argv) > 1:
        print("Raw input:", sys.argv, flush=True)
        inp = sys.argv[1]
        outputs = sys.argv[2::3]
        starts = sys.argv[3::3]
        stops = sys.argv[4::3]
        dt = sys.argv[-1]
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
            os.makedirs(os.path.dirname(outp), exist_ok=True)
            crop = (start <= time) & (time < stop)
            print("cropping time vector:", len(crop), sum(crop))
            with h5py.File(outp, "w") as nf:
                if all:
                    print("Copying all")
                    partial_copy_all(f, nf, "/all", lambda x: x[crop])
                else:
                    if spikes:
                        print("Copying spikes", flush=True)
                        partial_copy_all(f, nf, "/recorders/soma_spikes", lambda x: x[(start <= x) & (x <= stop)])
                    if voltages:
                        print("Copying voltages", flush=True)
                        partial_copy_all(f, nf, "/recorders/soma_voltages", lambda x: x[crop])
    finally:
        f.close()
