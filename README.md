# vhr-cnn-chm

Very high-resolution CNN-based CHM

![CI Workflow](https://github.com/nasa-nccs-hpda/vhr-cnn-chm/actions/workflows/ci.yml/badge.svg)
![CI to DockerHub ](https://github.com/nasa-nccs-hpda/vhr-cnn-chm/actions/workflows/dockerhub.yml/badge.svg)
![Code style: PEP8](https://github.com/nasa-nccs-hpda/vhr-cnn-chm/actions/workflows/lint.yml/badge.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Coverage Status](https://coveralls.io/repos/github/nasa-nccs-hpda/vhr-cnn-chm/badge.svg?branch=main)](https://coveralls.io/github/nasa-nccs-hpda/vhr-cnn-chm?branch=main)

## Configuration

```bash
export PYTHONPATH="/adapt/nobackup/people/jacaraba/development/tensorflow-caney:/adapt/nobackup/people/jacaraba/development/vhr-cnn-chm"
```

singularity exec --nv -B /lscratch,/explore/nobackup/projects/ilab,/explore/nobackup/people /explore/nobackup/projects/ilab/containers/tensorflow-caney python /explore/nobackup/people/jacaraba/development/vhr-cnn-chm/projects/chm/scripts/preprocess.py -c /explore/nobackup/people/jacaraba/development/vhr-cnn-chm/projects/chm/configs/tanana/cnn_tanana_v1.yaml 

singularity shell --nv -B /lscratch,/explore/nobackup/projects/ilab,/explore/nobackup/people /explore/nobackup/projects/ilab/containers/tensorflow-cane

## Experiments

### Experiment #1

CNN to output a single value per tile, this allows for a coarse result.

### Experiment #2

CNN to output a set of pixel-wise output. The input labels are the average h_can
value for that tile in locations where trees are present.

### Experiment #3

CNN to output a set of pixel-wise output. The input labels are the same h_can
value for that tile in every location, then a tree mask is multiplied to the tile
in the postprocessing.
