import numpy  as np
from glob import glob
from pathlib import Path
import os

directory = glob('/adapt/nobackup/projects/ilab/projects/Senegal/CNN_CHM/v2/intersection_tiles_evhrtoa/CAS/label/*.npy')
output_dir = '/adapt/nobackup/projects/ilab/projects/Senegal/CNN_CHM/v3/intersection_tiles_evhrtoa/CAS/label'

for f in directory:

    h_can = float(Path(f).stem.split('_')[-1])
    x = np.load(f).shape
    y = np.full(x, h_can)

    output_filename = os.path.join(output_dir, f.split('/')[-1])

    np.save(output_filename, y)

