# Reconstruction and simulation of an entire column of the cerebellar cortex

Required steps to reproduce De Schepper et al. 2022

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

To replicate the figures, you require certain result files in exact locations inside of a `results` folder. These files are available as datasets on Zenodo, here is a list of URLs for each plot:

<placeholder>

## Reproduce the results

In order to reproduce the results for yourself, install `bsb==3.10.3` and use the `balanced.hdf5` network to simulate the configuration files you can find in the `configs` folder. Place the result files for each config as dictated by this config-to-location list:

<placeholder>
