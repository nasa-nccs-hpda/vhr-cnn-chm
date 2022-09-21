import rioxarray as rxr
from glob import glob
import os

f = glob('/adapt/nobackup/projects/ilab/projects/Senegal/CNN_CHM/v1/intersection_tiles_evhrtoa/CAS/label/*.tif')
for xx in f:
    x = rxr.open_rasterio(xx)
    if x.shape[1] < 32 or x.shape[2] < 32:
        print(xx, x.shape)
        os.remove(xx)
    # else:
    #    print("NOOO")
