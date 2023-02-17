import os
import re
import time
import glob
import random
import logging
import warnings
import rasterio
import osgeo.gdal
import numpy as np
import pandas as pd
import geopandas as gpd
import rioxarray as rxr
from pathlib import Path
from itertools import repeat
from shapely.geometry import box
from multiprocessing import Pool, cpu_count
from omegaconf.listconfig import ListConfig

"""
from tensorflow_caney.utils.vector.extract import \
    convert_coords_to_pixel_location, extract_centered_window
from tensorflow_caney.utils.data import modify_bands
"""


def atl08_io(
    atl08_csv_output_dir,
    year_search, do_pickle=True,
    filename_regex='ATL08*.csv'
):
    """
    Read all ATL08 from CSVs of a given year after extract_filter_atl08.py
    Write to a pickle file by year
    Return a geodataframe
    Modified from https://github.com/pahbs/geoscitools/blob/master/atl08lib.py
    """
    dir_pickle = atl08_csv_output_dir
    filename_regex = os.path.join(
        atl08_csv_output_dir, year_search, filename_regex)

    all_atl08_csvs = glob.glob(filename_regex, recursive=True)
    if len(all_atl08_csvs) < 1:
        logging.info(f"No ATL08 CSVs were found under {filename_regex}")
        return
    logging.info(f"Processing ATL08 CSV: {filename_regex}")

    atl08_gdf = pd.concat(
        (pd.read_csv(f) for f in all_atl08_csvs),
        sort=False, ignore_index=True)  # <--generator is (), list is []

    atl08_gdf = gpd.GeoDataFrame(
        atl08_gdf, geometry=gpd.points_from_xy(
            atl08_gdf.lon, atl08_gdf.lat), crs='epsg:4326')

    if do_pickle:
        # Pickle the file
        if year_search == "**":
            year_search = 'allyears'
        cur_time = time.strftime("%Y%m%d%H%M")
        out_pickle_fn = os.path.join(
            dir_pickle, f"atl08_{year_search}_filt_gdf_{cur_time}.pkl")
        atl08_gdf.to_pickle(out_pickle_fn)
    return atl08_gdf


def filter_gdf_by_list(
    gdf,
    gdf_key: str = 'acq_year',
    isin_list: list = [],
    reset_index: bool = True
):
    """
    Filter GDF by year range.
    """
    return gdf[gdf[gdf_key].isin(isin_list)].reset_index(drop=True)


def get_atl08_gdf(
    data_dir: str,
    start_year: int = 2018,
    end_year: int = 2022,
    reset_index: bool = True,
    crs: str = None
):
    """
    Get ATL08 from extracted CSVs.
    """
    # --------------------------------
    # TODO, add GPU component here
    # --------------------------------
    # atl08_gdf = pd.concat([
    #    atl08_io(data_dir, str(year), do_pickle=False)
    #    for year in range(start_year, end_year)
    # ])
    dataframes = []
    for year in range(start_year, end_year):
        dataframe = atl08_io(data_dir, str(year), do_pickle=False)
        if dataframe is not None:
            dataframes.append(dataframe)
    atl08_gdf = pd.concat(dataframes)

    if crs is not None:
        atl08_gdf = atl08_gdf.to_crs(epsg=crs.split(':')[-1])
    return atl08_gdf.reset_index(drop=True)


def get_filenames(data_regex: str):
    """
    Get WorldView from local EVHR TIFs.
    Improvements:
        - this could be done way faster with multiprocessing
        - optional GPU or CPU versions, add GPU inside function
    """
    # get the paths/filenames of the world view imagery available
    filenames = []
    if isinstance(data_regex, list) or isinstance(data_regex, ListConfig):
        for regex in data_regex:
            filenames.extend(glob.glob(regex))
    else:
        filenames = glob.glob(data_regex)
    return sorted(filenames)


def filter_vhr_year_range(filenames, start_year, end_year):
    """
    Filter list by year.
    """
    new_list = []
    years = [str(year) for year in range(start_year, end_year+1)]
    for f in filenames:
        date_match = re.search(
            r'(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})',
            os.path.basename(f))
        if date_match['year'] in years:
            new_list.append(f)
    return sorted(new_list)


def filter_vhr_month_range(filenames, start_month, end_month):
    """
    Filter list by month
    """
    new_list = []
    months = ['{:02}'.format(month)
              for month in range(start_month, end_month+1)]
    print(months)
    for f in filenames:
        date_match = re.search(
            r'(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})',
            os.path.basename(f))
        print(date_match['month'])
        print(date_match)
        if date_match['month'] in months:
            new_list.append(f)
    return sorted(new_list)


def get_wv_vhr_gdf(data_regex: str, crs: str = None):
    """
    Get WorldView from local EVHR TIFs.
    Improvements:
        - this could be done way faster with multiprocessing
        - optional GPU or CPU versions, add GPU inside function
    """
    # get the paths/filenames of the world view imagery available
    filenames = []
    if isinstance(data_regex, list) or isinstance(data_regex, ListConfig):
        for regex in data_regex:
            filenames.extend(glob.glob(regex))
    else:
        filenames = glob.glob(data_regex)

    # define variables to store the output of the searches
    scene_ids_list, bounds_ids_list, years_ids_list, month_ids_list, \
        study_area_list = [], [], [], [], []

    if crs is None:
        crs = rasterio.open(filenames[0]).crs

    for filename in filenames:

        # append some metadata
        date_match = re.search(
            r'(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})',
            os.path.basename(filename))

        scene_ids_list.append(filename)
        years_ids_list.append(int(date_match['year']))
        month_ids_list.append(int(date_match['month']))
        bounds_ids_list.append(box(*rasterio.open(filename).bounds))

        # adding site name
        if os.path.basename(os.path.dirname(filename)) == 'M1BS':
            study_area_list.append(os.path.basename(
                os.path.dirname(os.path.dirname(filename))))
        else:
            study_area_list.append(os.path.basename(
                os.path.dirname(filename)))

    d = {
        'study_area': study_area_list,
        'scene_id': scene_ids_list,
        'wv_year': years_ids_list,
        'wv_month': month_ids_list,
        'geometry': bounds_ids_list
    }
    return gpd.GeoDataFrame(d, crs=crs)


def gen_random_tiles_from_swath(wv_vhr_gdf_iter,
                                lidar_swath_gdf,
                                tile_size: int,
                                input_bands: list,
                                output_bands: list,
                                images_output_dir: str,
                                labels_output_dir: str,
                                max_patches: int = None,
                                mask_path: str = None,
                                cloudmask_dir: str = None,
                                cloudcover_threshold: float = 0.9,
                                mask_preprocessing: bool = True,
                                augment: bool = False,
                                data_dir: str = '.'):
    """
    Given a single VHR scene extract sub-tiles.
    1. Extract the overlay geometry between
    the VHR scene and the LIDAR swath
    2. Clip (in mem) VHR scene, LIDAR swath, and any a priori masks
    3. Extract random subtiles, apply random augmentations
    """
    # Get geometry overlay between VHR and LIDAR swath
    row_id, row = wv_vhr_gdf_iter
    clip_geometry = [row.geometry]
    lidar_file_path = os.path.join(
        '/explore/nobackup/people/cssprad1/projects/serc-howland-chm/data/GLiHT/Serc/', row['location'])
    print(row['location'])
    print(row['study_area'])
    lidar_swath = rxr.open_rasterio(
        lidar_file_path)
    # Clip VHR scene to overlay geometry
    vhr_scene = rxr.open_rasterio(
        row['scene_id']).rio.reproject_match(lidar_swath)

    print('Clipping')
    vhr_scene_clipped = vhr_scene.rio.clip(clip_geometry)
    image = vhr_scene_clipped.data
    image = np.moveaxis(image, 0, -1)
    print(image.shape)

    # Clip LIDAR scenes to overlay geometry
    lidar_swath_clipped = lidar_swath.rio.clip(clip_geometry)
    label = lidar_swath_clipped.data
    label = label[0, :, :]
    label_min = np.nanmin(label)
    label_max = np.nanmax(label)
    print(label_min)
    print(label_max)
    label_min_max_filename = os.path.join(data_dir, 'label_min_max.csv')
    label_min_max = np.stack([label_min, label_max], axis=0)
    pd.DataFrame(label_min_max).to_csv(label_min_max_filename, header=None, index=None)

    if mask_path and mask_preprocessing:
        mask = rxr.open_rasterio(mask_path).rio.reproject_match(lidar_swath)
        mask = mask.rio.clip(clip_geometry)
        mask = mask.data
        mask = mask[0, :, :]
        print(mask.shape)

    if cloudmask_dir is not None:

        # get filename from cloudmask
        cloudmask_filename = os.path.basename(
            row['scene_id']).replace('sr-02m.tif', 'toa.cloudmask.tif')
        cloudmask_filepath = os.path.join(
            cloudmask_dir,
            cloudmask_filename)
        if not os.path.exists(cloudmask_filepath):
            raise FileNotFoundError(cloudmask_filepath)
        # read cloudmask dataset
        cloud_mask = \
            rxr.open_rasterio(cloudmask_filepath).rio.reproject_match(
                lidar_swath)
        cloud_mask = cloud_mask.rio.clip(clip_geometry)
        cloud_mask = cloud_mask.data
        cloud_mask = cloud_mask[0, :, :]
        print(cloud_mask.shape)
        cloudcover_percentage = \
            np.count_nonzero(cloud_mask == 1)/cloud_mask.size
        cloudcover_percentage = round(cloudcover_percentage, 3)
        print(f'Cloud cover percentage: {cloudcover_percentage}')
        if cloudcover_percentage > cloudcover_threshold:
            msg = f'Cloud-cover {cloudcover_percentage} ' + \
                f'exceeded threshold {cloudcover_threshold}'
            warnings.warn(msg)
            return

    # image = modify_bands(
    #    image, input_bands, output_bands)

    # sub-tile filename starts with VHR basename
    output_filename = os.path.basename(row['scene_id'])

    generated_tiles = 0  # counter for generated tiles
    bad_tiles = 0
    cloud_masked_tiles = 0
    total_tries = 0
    all_same = 0
    while generated_tiles < max_patches:
        total_tries += 1
        # Generate random integers from image
        y = random.randint(0, image.shape[0] - tile_size)
        x = random.randint(0, image.shape[1] - tile_size)

        # Generate img and mask patches
        image_tile = image[y:(y + tile_size), x:(x + tile_size)]
        label_tile = label[y:(y + tile_size), x:(x + tile_size)]
        if mask is not None:
            mask_tile = mask[y:(y + tile_size), x:(x + tile_size)]
        if cloud_mask is not None:
            cloud_mask_tile = cloud_mask[y:(y + tile_size), x:(x + tile_size)]

        # if nodata is present, or if tiles is incorrect, skip
        image_tile_min = image_tile.min()
        if image_tile is None or \
                image_tile_min < 0 or np.isnan(image_tile_min):
            bad_tiles += 1
            continue

        if mask_tile is not None:
            mask_tile_thresholded = np.where(mask_tile >= 0.05, 1, 0)
            label_tile = np.where(mask_tile_thresholded == 0, 0, label_tile)

        label_tile_max = label_tile.max()
        label_tile_min = label_tile.min()
        if label_tile_max == label_tile_min:
            all_same += 1
            continue

        if label_tile_min < 0 or \
                label_tile_max > 200 or np.isnan(label_tile_max):
            bad_tiles += 1
            continue

        if cloud_mask_tile is not None:
            cloud_pixels = (cloud_mask_tile == 1)
            if np.any(cloud_pixels):
                cloud_masked_tiles += 1
                continue

        # Add to the tiles counter
        generated_tiles += 1
        filename = f'{Path(output_filename).stem}_{generated_tiles}.npy'

        if np.min(label_tile) == np.max(label_tile):
            raise RuntimeError('PRE PRE BAD BAD PREPROCESS')

        # Apply some random transformations
        if augment:

            if np.random.random_sample() > 0.5:
                # metadata[filename]['fliplr'] = True
                image_tile = np.fliplr(image_tile)
                label_tile = np.fliplr(label_tile)

            if np.random.random_sample() > 0.5:
                # metadata[filename]['flipud'] = True
                image_tile = np.flipud(image_tile)
                label_tile = np.flipud(label_tile)

            if np.random.random_sample() > 0.5:
                # metadata[filename]['rot90'] = True
                image_tile = np.rot90(image_tile, 1)
                label_tile = np.rot90(label_tile, 1)

            if np.random.random_sample() > 0.5:
                # metadata[filename]['rot180'] = True
                image_tile = np.rot90(image_tile, 2)
                label_tile = np.rot90(label_tile, 2)

            if np.random.random_sample() > 0.5:
                # metadata[filename]['rot270'] = True
                image_tile = np.rot90(image_tile, 3)
                label_tile = np.rot90(label_tile, 3)

        # save tiles to disk
        if np.min(label_tile) == np.max(label_tile):
            raise RuntimeError('BAD BAD PREPROCESS')
        np.save(os.path.join(images_output_dir, filename), image_tile)
        np.save(os.path.join(labels_output_dir, filename), label_tile)
    print(f'Tiles generated: {generated_tiles}/{max_patches}')
    print(f'Total tile attempts: {total_tries}')
    print(f'Bad data tiles: {bad_tiles}/{total_tries}')
    print(f'All same tiles: {all_same}/{total_tries}')
    print(f'Cloud masked tiles: {cloud_masked_tiles}/{total_tries}')
    return


def filter_polygon_in_raster(filename, atl08_gdf, output_dir):
    """
    Multiprocessing wrapper to process a single point/polygon row.
    TODO:
        options to filter by year
        options to filter by month
    """
    # create polygon row
    date_match = re.search(
        r'(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})', filename)

    # adding site name
    if os.path.basename(os.path.dirname(filename)) == 'M1BS':
        study_area = os.path.basename(
            os.path.dirname(os.path.dirname(filename)))
    else:
        study_area = os.path.basename(os.path.dirname(filename))

    # open raster and create gdb object
    raster = rasterio.open(filename)
    d = {
        'study_area': study_area,
        'scene_id': filename,
        'wv_year': int(date_match['year']),
        'wv_month': int(date_match['month']),
        'geometry': box(*raster.bounds)
    }
    polygon_row = gpd.GeoDataFrame(d, crs=raster.crs, index=[1])
    atl08_gdf = atl08_gdf.to_crs(raster.crs)

    # intersect both datasets
    intersection = gpd.overlay(atl08_gdf, polygon_row, how='intersection')

    # filter by year
    # intersection = intersection[intersection['y'] == intersection['wv_year']]

    # set output_dir and create directory
    output_dir = os.path.join(output_dir, study_area)
    os.makedirs(output_dir, exist_ok=True)

    # save geopackage file within the output_dir
    try:
        intersection.to_file(
            os.path.join(
                output_dir, f"{Path(filename).stem}.gpkg"),
            driver='GPKG', layer='intersection'
        )
    except ValueError:
        return
    return


def filter_polygon_in_raster_parallel(
    wv_evhr_gdf,
    atl08_gdf,
    output_dir: str,
    n_processes: int = cpu_count()
) -> None:
    """
    Return overalapping points
    """
    logging.info(f"Iterating over {len(wv_evhr_gdf)} rasters")
    logging.info(f"Iterating over {atl08_gdf.shape[0]} points")

    # multiprocessing pool
    p = Pool(processes=n_processes)
    p.starmap(
        filter_polygon_in_raster,
        zip(
            wv_evhr_gdf,
            repeat(atl08_gdf),
            repeat(output_dir)
        )
    )
    p.close()
    p.join()
    return


def extract_tiles(
    gpd_iter,
    tile_size: int,
    input_bands: list,
    output_bands: list,
    images_output_dir: str,
    labels_output_dir: str,
    label_attribute: str = 'h_can',
    mask_dir: str = None,
    cloudmask_dir: str = None,
    mask_preprocessing: bool = True
):
    """
    Extract and save tile from pandas dataframe metadata.
    """
    try:
        # decompress row iter object into row_id and geopandas row
        row_id, row = gpd_iter

        # open raster via GDAL
        image_dataset = osgeo.gdal.Open(row['scene_id'])

        # get window pixel location from coordinates
        window_pixel_x, window_pixel_y = convert_coords_to_pixel_location(
            [row['geometry'].centroid.x, row['geometry'].centroid.y],
            image_dataset.GetGeoTransform()
        )

        # extract data array from the greater raster
        data_array = extract_centered_window(
            image_dataset, window_pixel_x, window_pixel_y,
            tile_size, tile_size
        )

        # if nodata is present, or if tiles is incorrect, skip
        if data_array is None or data_array.min() < 0:
            return

        data_array = modify_bands(
            data_array, input_bands, output_bands)

        # this if statement takes care of the experiment
        # where labels utilize landcover mask as a primer
        if mask_dir is not None and mask_preprocessing:

            # set mask filename
            mask_filename = os.path.join(
                mask_dir, row['study_area'],
                f"{Path(row['scene_id']).stem}.trees.tif")

            # open raster via GDAL
            mask_dataset = osgeo.gdal.Open(mask_filename)

            # extract data array from the greater raster
            mask_array = extract_centered_window(
                mask_dataset, window_pixel_x, window_pixel_y,
                tile_size, tile_size
            )

            if cloudmask_dir is not None:

                # get filename from cloudmask
                cloudmask_filename = os.path.join(
                    cloudmask_dir, row['study_area'],
                    f"{Path(row['scene_id']).stem}.cloudmask.tif")

                # read cloudmask dataset
                cloudmask_dataset = osgeo.gdal.Open(cloudmask_filename)

                # extract tile from cloudmask
                cloudmask_array = extract_centered_window(
                    cloudmask_dataset, window_pixel_x, window_pixel_y,
                    tile_size, tile_size
                )
                cloudmask_array[cloudmask_array == 1] = 255

                # sum the cloud mask to the array
                mask_array = mask_array + cloudmask_array

            # multiply label to mask, since 1 are trees, these will
            # be the only greater than 0 values
            mask_array = mask_array * round(row[label_attribute], 3)

            if mask_array.max() <= 0 or mask_array.max() > 200:
                return

        # this else statement takes care of the experiment
        # where all pixels are the same h_can value, and
        # landcover is used as a postprocessing step
        else:
            # Future implementation:
            # set entire tile to be h_can value
            # clipped_mask = np.full(
            #    (1, clipped_data.shape[1], clipped_data.shape[2]),
            #    h_can
            # )
            return

        # set output filenames
        output_data_filename = os.path.join(
            images_output_dir,
            f'{Path(row["scene_id"]).stem}_{str(row_id+1)}.npy'
        )
        output_label_filename = os.path.join(
            labels_output_dir,
            f'{Path(row["scene_id"]).stem}_{str(row_id+1)}_' +
            f'{round(row[label_attribute], 3)}.npy'
        )

        data_array = np.moveaxis(data_array, 0, -1)
        mask_array = np.expand_dims(mask_array, -1)

        # output to disk
        np.save(output_data_filename, data_array)
        np.save(output_label_filename, mask_array)

    except (AttributeError, IndexError) as e:
        logging.info(e)
        return
    return


def extract_tiles_parallel(
    intersection_filenames: list,
    tile_size: int,
    input_bands: list,
    output_bands: list,
    images_output_dir: str,
    labels_output_dir: str,
    label_attribute: str = 'h_can',
    mask_dir: str = None,
    cloudmask_dir: str = None,
    mask_preprocessing: bool = False,
    n_processes=cpu_count()
):

    for intersection_filename in intersection_filenames:

        # open the filename
        dataset_gdf = gpd.read_file(intersection_filename)

        p = Pool(processes=n_processes)
        p.starmap(
            extract_tiles,
            zip(
                dataset_gdf.iterrows(),
                repeat(tile_size),
                repeat(input_bands),
                repeat(output_bands),
                repeat(images_output_dir),
                repeat(labels_output_dir),
                repeat(label_attribute),
                repeat(mask_dir),
                repeat(cloudmask_dir),
                repeat(mask_preprocessing)
            )
        )
        p.close()
        p.join()
    return
