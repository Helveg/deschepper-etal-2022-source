from bsb.config import JSONConfig
import plotly.graph_objs as go
import h5py, json, os, glob, itertools

def plot(*files):
    if not files:
        raise ValueError("Provide at least a single configuration or HDF5 file to plot device configuration of.")

    configs = []
    files = list(itertools.chain(*map(glob.glob, map(os.path.abspath, files))))
    print(len(files))
    for file in files:
        # Figure out whether it's an HDF5 or a config file and extract the cfg
        # All the try/except/finally's make sure that we close all files we open
        try:
            f = h5py.File(file, "r")
            try:
                configs.append(f.attrs["configuration_string"])
            except KeyError as e:
                raise IOError(f"Missing conf string in result file `{file}`")
            finally:
                f.close()
        except OSError as e:
            f = open(file, "r")
            try:
                configs.append(f.read())
            finally:
                f.close()
    print(len(configs))
    figs = {}
    for file, conf in zip(files, configs):
        figs[file] = fig = go.Figure(layout=dict(title_text=file))
        try:
            cfg_obj = JSONConfig(stream=conf)
        except Exception as e:
            raise IOError(f"Unable to make conf for `{file}`: {e}")
        for sim_name, sim in cfg_obj.simulations.items():
            plottable_devices = {k: v for k, v in sim.devices.items() if hasattr(v, "parameters")}
            t = len(plottable_devices)
            frac = 1 / t
            for i, (device_name, device) in enumerate(plottable_devices.items()):
                params = device.parameters
                if hasattr(params, "spike_times"):
                    start = min(map(float, params.spike_times))
                    stop = max(map(float, params.spike_times))
                else:
                    start = float(params["start"])
                    stop = start + (float(params["interval"]) * float(params["number"]))
                fig.add_trace(
                    go.Scatter(
                        x=[start, start, max(start + 3, stop), max(start + 3, stop)],
                        y=[i, i + 1, i + 1, i],
                        fill="toself",
                        line=dict(
                            width=2,
                        ),
                        text=f"<b><span style='font-size: 20px'>{device_name}</span></b><br /><br />" + json.dumps(cfg_obj._parsed_config["simulations"][sim_name]["devices"][device_name], indent=2).replace("\n", "<br />"),
                    )
                )
    return figs
