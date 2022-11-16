# vhr-cnn-chm

Very high-resolution CNN-based CHM

![CI Workflow](https://github.com/nasa-nccs-hpda/vhr-cnn-chm/actions/workflows/ci.yml/badge.svg)
![CI to DockerHub ](https://github.com/nasa-nccs-hpda/vhr-cnn-chm/actions/workflows/dockerhub.yml/badge.svg)
![Code style: PEP8](https://github.com/nasa-nccs-hpda/vhr-cnn-chm/actions/workflows/lint.yml/badge.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Coverage Status](https://coveralls.io/repos/github/nasa-nccs-hpda/vhr-cnn-chm/badge.svg?branch=main)](https://coveralls.io/github/nasa-nccs-hpda/vhr-cnn-chm?branch=main)

## Objectives

- Library to match ATL08 and WorldView inputs.
- Machine Learning and Deep Learning regression to produce canopy height models.
- Postprocessing methods for regression output smoothing.
- Validation with GEDI and GLiHT data sources.

## Installation

The following library is intended to be used to accelerate the development of data science products
for remote sensing satellite imagery, or any other applications. tensorflow-caney can be installed
by itself, but instructions for installing the full environments are listed under the requirements
directory so projects, examples, and notebooks can be run.

Note: PIP installations do not include CUDA libraries for GPU support. Make sure NVIDIA libraries
are installed locally in the system if not using conda/mamba.

```bash
module load singularity
singularity build --sandbox tensorflow-caney docker://nasanccs/tensorflow-caney:latest
```

Example run (assuming you exposed your environment variables into the container):

```bash
singularity exec --nv -B /lscratch,/explore/nobackup/projects/ilab,/explore/nobackup/people /explore/nobackup/projects/ilab/containers/tensorflow-caney-22.11 python /explore/nobackup/people/jacaraba/development/vhr-cnn-chm/projects/chm/scripts/preprocess.py -c /explore/nobackup/people/jacaraba/development/vhr-cnn-chm/projects/chm/configs/tanana/cnn_tanana_v1.yaml 
```

## Configuration

This step will change in the future, but you will need to clone both the [tensorflow-caney](https://github.com/nasa-nccs-hpda/tensorflow-caney) and [vhr-cnn-chm](https://github.com/nasa-nccs-hpda/vhr-cnn-chm) repositories to use the methods included in them. These will be migrated to PIP files in the short future.

```bash
export PYTHONPATH="/adapt/nobackup/people/jacaraba/development/tensorflow-caney:/adapt/nobackup/people/jacaraba/development/vhr-cnn-chm"
```

## Why CNNs?

CNNs have additional spatial abilities that traditional algorithms are not capable of.

## Contributors

- Jordan Alexis Caraballo-Vega, jordan.a.caraballo-vega@nasa.gov
- Caleb Spradlin, caleb.s.spradlin@nasa.gov

## Contributing

Please see our [guide for contributing to tensorflow-caney](CONTRIBUTING.md).

## References

- [TensorFlow Advanced Segmentation Models](https://github.com/JanMarcelKezmann/TensorFlow-Advanced-Segmentation-Models)
