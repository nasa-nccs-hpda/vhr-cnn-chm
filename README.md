# vhr-cnn-chm

Very high-resolution CNN-based CHM

## Configuration

```bash
export PYTHONPATH="/adapt/nobackup/people/jacaraba/development/tensorflow-caney:/adapt/nobackup/people/jacaraba/development/vhr-cnn-chm"
```

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
