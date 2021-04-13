import dbbs_models, nrnsub
from multiprocessing.pool import Pool
import numpy as np, uuid, h5py
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from time import time, sleep


def rancz(trials, scale=1.0, delay=20):
    for _ in range(trials):
        spikes = list(np.random.randn(1) * 1.2)
        if np.random.rand() <= 12 / 14:
            spikes.extend(np.random.randn(1) * 1.2 + 8.5)
        if np.random.rand() <= 7 / 14:
            spikes.extend(np.random.randn(1) * 2.2 + 40)
        if np.random.rand() <= 13 / 14:
            spikes.extend(np.random.randn(1) * 1.8 + 80)
        if np.random.rand() <= 4 / 14:
            spikes.extend(np.random.randn(1) * 4.5 + 105)
        yield np.array(spikes) / scale + delay

def make_synapse(cell, section, type, stimuli):
    from patch import p

    syn = cell.create_synapse(section, type)
    # Connect the synapse to each given NetStim stimulus
    for ns in stimuli:
        p.NetCon(ns, syn._point_process)

def make_stim(seed, **kwargs):
    # print("Making a stim:", kwargs)
    from patch import p

    x = p.NetStim()
    for k,v in kwargs.items():
        setattr(x, k, v)
    x.seed(seed)
    return x


def model(rancz_scale, duration, dendrites, stimulus, goc_coherence, gabazine, rng_seed, mf_bg_rate=4, goc_bg_rate=18, goc_delay=4):
    from patch import p

    rng = np.random.default_rng(rng_seed)
    def seeder():
        loop_seed = rng_seed
        while True:
            if loop_seed - 1 < 0:
                loop_seed = np.iinfo(np.int_).max - 1
            else:
                loop_seed -= 1
            yield loop_seed

    seedtr = seeder()
    stimuli = [make_stim(next(seedtr), start=s, number=1, interval=1) for s in stimulus]
    # print("Stimulus:", stimulus)
    # print("Turned into:", stimuli)
    goc_stimuli = [
        [make_stim(next(seedtr), start=s + goc_delay, number=1, interval=1) for s in stimulus if rng.random() <= goc_coherence]
        for _ in range(4)
    ]
    mf_bg = [make_stim(next(seedtr), start=0, number=int(duration / 1000 * mf_bg_rate) * 3, interval=1000/mf_bg_rate, noise=True) for _ in range(4)]
    # print("MF BG", mf_bg)
    # print("Golgi coherence", goc_coherence)
    # print(goc_stimuli)
    goc_bg = [make_stim(next(seedtr), start=0, number=int(duration / 1000 * goc_bg_rate) * 3, interval=1000/goc_bg_rate, noise=True) for _ in range(4)]
    # print("Goc BG:", goc_bg)
    p.celsius = 32
    p.dt = 0.025
    if gabazine:
        dbbs_models.GranuleCell.section_types["dendrites"]["mechanisms"].remove(("Leak", "GABA"))
        # print("Removed GABA Leak")
    grc = dbbs_models.GranuleCell()
    t = p.time
    grc.record_soma()
    # print("This GrC has", dendrites, "active dendrites")
    for i in range(4):
        dend_stim = [mf_bg[i]]
        if i < dendrites:
            dend_stim += stimuli
        # print("Adding AMPA & NMDA to dend", i, dend_stim)
        dendrite = grc.dendrites[i]
        make_synapse(grc, dendrite, "AMPA", dend_stim)
        make_synapse(grc, dendrite, "NMDA", dend_stim)
    if not gabazine:
        # print("GABA'ing all dends")
        for dend, stim, bg in zip(grc.dendrites, goc_stimuli, goc_bg):
            make_synapse(grc, dend, "GABA", [bg] + stim)
    p.finitialize(-65)
    p.continuerun(duration)
    return np.column_stack((t, grc.Vm))

def jobgen(rancz_min, rancz_max, rancz_step, trials, warmup, cooldown):
    master_rng = np.random.default_rng()
    for rancz_scale in np.linspace(rancz_min, rancz_max, rancz_step):
        for gabazine in (True, False,):
            for dendrites in range(5):
                for coherence in (0.05, 0.25, 0.5, 0.75, 1):
                    for trial in rancz(trials, rancz_scale):
                        trial += warmup
                        duration = max(trial) + cooldown
                        yield (rancz_scale, duration, dendrites, trial, coherence, gabazine, master_rng.integers(0, np.iinfo(np.int_).max, endpoint=False))

def plot():
    figs = dict()
    figs["rancz_gen"] = go.Figure(
        [
            go.Scatter(x=spikes, y=np.ones(len(spikes)) * i, mode="markers", name=f"trial {i}")
            for i, spikes in enumerate(rancz(100))
        ]
    )
    rmin = 0.1
    rmax = 2
    rsteps = 20
    trials = 20
    warmup = 500
    cooldown = 100
    done = set()
    start_time = time()
    batchid = str(uuid.uuid4())
    i = 0
    with Pool(processes=8, maxtasksperchild=1) as pool:
        # Skip finished jobs 0-16581
        jobs = list(jobgen(rmin, rmax, rsteps, trials, warmup, cooldown))
        fs = [
            pool.apply_async(model, args)
            for args in jobs
        ]
        lookup = dict(zip(fs, jobs))
        print("Generated", len(fs), "jobs")
        while len(done) < len(jobs):
            sleep(1)
            all_done = set(f for f in fs if f.ready())
            newly_done = all_done - done
            for f in newly_done:
                job = lookup.get(f)
                arg_dict = dict(zip(("rancz", "duration", "act_dend", "stimulus", "coherence", "gabazine"), job))
                print("Job finished with args:", arg_dict)
                try:
                    result = f.get()
                    with h5py.File(f"/home/robin/rancor-{batchid}.hdf5", "a") as data:
                        ds = data.require_group(
                            str(arg_dict["rancz"])
                        ).require_group(
                            str(arg_dict["coherence"])
                        ).require_group(
                            str(arg_dict["act_dend"])
                        ).require_group(
                            str(arg_dict["gabazine"])
                        ).create_dataset(str(i), data=result)
                        for arg, val in arg_dict.items():
                            ds.attrs[arg] = val
                    print(f"Finished after {round(time() - start_time, 2)}s")
                    i += 1
                    done.add(f)
                    print(f"Finished {len(done)} out of {len(jobs)}")
                except Exception as e:
                    import traceback

                    traceback.print_exc()
    return figs
