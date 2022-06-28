import os
from glob import glob
from pathlib import Path

intersection_wv = glob('/adapt/nobackup/projects/ilab/projects/Senegal/CNN_CHM/intersection_metadata_evhrtoa/*/*.gpkg')
data_directory = '/adapt/nobackup/projects/3sl/data/VHR'
landcover_directory = '/adapt/nobackup/projects/3sl/products/landcover/trees.v2'
output_directory = '/adapt/nobackup/projects/ilab/projects/Senegal/CNN_CHM/data/intersection_metadata_evhrtoa_trees.v2'

print("Intersection WV files", len(intersection_wv))

for filename in intersection_wv:

    location = filename.split('/')[-2]
    filename_stem = Path(filename).stem

    data_filename = os.path.join(
        data_directory, location, 'M1BS', f'{filename_stem}.tif')

    mask_filename = os.path.join(
        landcover_directory, location, f'{filename_stem}.trees.tif')        

    full_output_directory = os.path.join(output_directory, location)
    output_filename = os.path.join(full_output_directory, 

    print(data_filename, mask_filename, output_filename))

