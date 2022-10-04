# vhr-cnn-chm

Very high-resolution CNN-based CHM

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
