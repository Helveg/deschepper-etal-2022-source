import sys, h5py, numpy as np

file = sys.argv[1]
duration = 8000

with h5py.File(file, "a") as f:
    # del f["/all"]
    dset = f.require_group("all")
    for k, s in f["/recorders/soma_spikes"].items():
        f.copy(s, "/all/" + k)
    for k, s in f["/recorders/input/background"].items():
        ntimes= np.array(s)
        ntimes= ntimes[ntimes<duration]
        npID = np.flip(np.ones(len(ntimes))*int(float(k)))
        data = np.column_stack((npID,ntimes))
        K=int(float(k))
        s=f.create_dataset("/all/" + str(K), data=data)
        s.attrs["label"] = "mossy_fiber"
        s.attrs["display_label"] = "mossy_fiber"
        s.attrs["cell_id"] = int(float(k))
        s.attrs["color"] = "black"
        s.attrs["order"] = 0
        #f.copy(s, f"/all/{int(float(k))}")
    for k, s in f["/recorders/input/mossy_fiber_input_Poiss"].items():
        if k in f["/all"]:
        #print(np.concatenate( (np.array(f["/all/" + k]), np.array(f["/recorders/input/mossy_fiber_input_Poiss/" + k]))))
            v = np.array(f["/all/" + k])
            del f["/all/" + k]
            ntimes=np.sort (np.concatenate( (v[:,1], np.array(f["/recorders/input/mossy_fiber_input_Poiss/" + k]))))
            ntimes= ntimes[ntimes<duration]
            npID = np.flip(np.ones(len(ntimes))*int(float(k)))
            data = np.column_stack((npID,ntimes))
            #print(s, npID, data)
            K=int(float(k))
            s=f.create_dataset("/all/" + k, data=data)
            s.attrs["label"] = "mossy_fiber"
            s.attrs["display_label"] = "mossy_fiber"
            s.attrs["cell_id"] = int(float(k))
            s.attrs["color"] = "black"
            s.attrs["order"] = 0
        else:
            ntimes= np.array(s)
            ntimes= ntimes[ntimes<duration]
            npID = np.flip(np.ones(len(ntimes))*int(float(k)))
            data = np.column_stack((npID,ntimes))
            K=int(float(k))
            s=f.create_dataset("/all/" + str(K), data=data)
            s.attrs["label"] = "mossy_fiber"
            s.attrs["display_label"] = "mossy_fiber"
            s.attrs["cell_id"] = int(float(k))
            s.attrs["color"] = "black"
            s.attrs["order"] = 0

    for k, s in f["/recorders/input/mossy_fiber_input_syncImp"].items():
        if k in f["/all"]:
        #print(np.concatenate( (np.array(f["/all/" + k]), np.array(f["/recorders/input/mossy_fiber_input_Poiss/" + k]))))
            v = np.array(f["/all/" + k])
            del f["/all/" + k]
            ntimes=np.sort (np.concatenate( (v[:,1], np.array(f["/recorders/input/mossy_fiber_input_syncImp/" + k]))))
            ntimes= ntimes[ntimes<duration]
            npID = np.flip(np.ones(len(ntimes))*int(float(k)))
            data = np.column_stack((npID,ntimes))
            #print(s, npID, data)
            K=int(float(k))
            s=f.create_dataset("/all/" + k, data=data)
            s.attrs["label"] = "mossy_fiber"
            s.attrs["display_label"] = "mossy_fiber"
            s.attrs["cell_id"] = int(float(k))
            s.attrs["color"] = "black"
            s.attrs["order"] = 0
        else:
            ntimes= np.array(s)
            ntimes= ntimes[ntimes<duration]
            npID = np.flip(np.ones(len(ntimes))*int(float(k)))
            data = np.column_stack((npID,ntimes))
            K=int(float(k))
            s=f.create_dataset("/all/" + str(K), data=data)
            s.attrs["label"] = "mossy_fiber"
            s.attrs["display_label"] = "mossy_fiber"
            s.attrs["cell_id"] = int(float(k))
            s.attrs["color"] = "black"
            s.attrs["order"] = 0
