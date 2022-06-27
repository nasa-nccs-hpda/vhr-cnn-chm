from glob import glob
from pathlib import Path

atl08_dir = '/adapt/nobackup/projects/ilab/projects/Senegal/CNN_CHM/metadata/*.gpkg'
atl08_files = glob(atl08_dir)

wv_data_dirs = [
        '/adapt/nobackup/people/mwooten3/Senegal_LCLUC/VHR/CAS/M1BS/*.tif',
        '/adapt/nobackup/people/mwooten3/Senegal_LCLUC/VHR/ETZ/M1BS/*.tif',
        '/adapt/nobackup/people/mwooten3/Senegal_LCLUC/VHR/SRV/M1BS/*.tif'
]

wvt = '/adapt/nobackup/people/mcarrol2/LCLUC_Senegal/ForKonrad/*_data.tif'
wv_data = glob(wvt)
wv_data_stem = []
for filename in wv_data:
    wv_data_stem.append(Path(filename).stem[9:22])
print(wv_data_stem)

#wv_data = glob(wv_data_dirs[0])
#wv_data_stem = []
#for filename in wv_data:
#    wv_data_stem.append(Path(filename).stem[:13])

atl08_stem = []
for filename in atl08_files:

    atl08_fn = Path(filename).stem[:13]
    if atl08_fn in wv_data_stem and atl08_fn[:2] == 'WV':
        atl08_stem.append(Path(filename).stem[:13])
print(len(atl08_stem))
print(atl08_stem)

