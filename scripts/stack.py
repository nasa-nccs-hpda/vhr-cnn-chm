import sys
from osgeo import gdal

# or use sorted(glob.glob('*.tif')) if input images are sortable
ImageList = [sys.argv[1], sys.argv[2]]
VRT = 'OutputImage.vrt'
gdal.BuildVRT(VRT, ImageList, separate=True, callback=gdal.TermProgress_nocb)

InputImage = gdal.Open(VRT, 0)  # open the VRT in read-only mode
gdal.Translate(
    'OutputImageName.tif',
    InputImage,
    format='GTiff',
    creationOptions=['COMPRESS:DEFLATE', 'TILED:YES'],
    callback=gdal.TermProgress_nocb
)
del InputImage  # close the VRT
