# Reconstruction and simulation of an entire column of the cerebellar cortex

Required steps to reproduce De Schepper et al. 2022

## Installing the full software environment

This step is not always required, to simply reproduce the plots from the available datasets, this can be skipped.

```
pip install -r requirements.txt --only-binary=":all:"
```

## Usage

1. Plot all figures

```
python plot
```

2. Plot specific figures

```
python plot figure1 figure2
```

The figure names correspond to the filenames without the suffix

3. Build the static image files

```
python build [format]
```

The default format is EPS.

## Obtaining the data

You can obtain an already reconstructed version of the network model at https://doi.org/10.5281/zenodo.7230288 and save it as `/networks/balanced.hdf5`. Otherwise you can reconstruct it yourself by downloading the [morphologies](https://doi.org/10.5281/zenodo.7230455) as `morphologies.hdf5` running:

```
bsb -v=4 -c=configs/balanced.json compile
```

To replicate the figures, you require certain result files in exact locations inside of the `results` folder. These files are available as datasets on Zenodo, here is a list of URLs for each plot:

* `feedforward_jitter`: https://doi.org/10.5281/zenodo.7230239 (place in: `results/single_impulse/sensory_burst`)
* `feedforward`: https://doi.org/10.5281/zenodo.7230798, https://doi.org/10.5281/zenodo.7230830, https://doi.org/10.5281/zenodo.7230836 (place in: `results/clamp`)
* `goc_nspos`, `goc_sync*`: https://doi.org/10.5281/zenodo.7231068 (place in: `results`)
* `goc_nsync`, `goc_sync*`: https://doi.org/10.5281/zenodo.7231161 (place in: `results`)
* `goc_oscillations`: https://doi.org/10.5281/zenodo.7231187 (place in: `results/oscillations`)
* `grc_activation_overlap`: Calcium signal datasets too large to be uploaded (1.44TB). Please reproduce using `sensory_burst_calcium_dense.json`.

## Reproduce the results

In order to reproduce the results for yourself, install `bsb==3.10.3` and use the `balanced.hdf5` network to simulate the configuration files you can find in the `configs` folder. Place the result files for each config as dictated by this config-to-location list:

<placeholder>
