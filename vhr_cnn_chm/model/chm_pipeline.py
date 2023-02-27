import os
import sys
import time
import logging
import rasterio
import osgeo.gdal
import numpy as np
import xarray as xr
import rioxarray as rxr
import geopandas as gpd

from tqdm import tqdm
from glob import glob
from pathlib import Path
from itertools import repeat
from multiprocessing import Pool, cpu_count

from vhr_cnn_chm.model.config import CHMConfig as Config
from vhr_cnn_chm.model.atl08 import ATL08
from tensorflow_caney.utils.vector.extract import \
    convert_coords_to_pixel_location, extract_centered_window
from tensorflow_caney.utils.data import modify_bands, \
    get_dataset_filenames, get_mean_std_dataset, get_mean_std_metadata
from tensorflow_caney.utils.system import seed_everything
from tensorflow_caney.model.pipelines.cnn_regression import CNNRegression
from tensorflow_caney.model.dataloaders.regression import RegressionDataLoader
from tensorflow_caney.utils import indices
from tensorflow_caney.utils.model import load_model
from tensorflow_caney.inference import regression_inference
from pygeotools.lib import iolib, warplib

osgeo.gdal.UseExceptions()


class CHMPipeline(CNNRegression, ATL08):

    # -------------------------------------------------------------------------
    # __init__
    # -------------------------------------------------------------------------
    def __init__(self, config_filename, data_csv=None, logger=None):

        # TODO:
        # slurm filename in output dir

        # Set logger
        self.logger = logger if logger is not None else self._set_logger()

        # Configuration file intialization
        self.conf = self._read_config(config_filename, Config)

        # Set Data CSV
        self.data_csv = data_csv

        # output directory to store metadata and artifacts
        self.metadata_dir = os.path.join(self.conf.data_dir, 'metadata')
        self.logger.info(f'Metadata dir: {self.metadata_dir}')

        # output directory to store intersections
        self.intersection_dir = os.path.join(self.metadata_dir, 'intersection')
        self.logger.info(f'Intersection dir: {self.intersection_dir}')

        # Set output directories and locations
        self.images_dir = os.path.join(self.conf.data_dir, 'images')
        self.logger.info(f'Images dir: {self.images_dir}')

        self.labels_dir = os.path.join(self.conf.data_dir, 'labels')
        self.logger.info(f'Labels dir: {self.labels_dir}')

        self.model_dir = self.conf.model_dir
        self.logger.info(f'Model dir: {self.labels_dir}')

        # Create output directories
        for out_dir in [
                self.images_dir, self.labels_dir,
                self.metadata_dir, self.intersection_dir, self.model_dir]:
            os.makedirs(out_dir, exist_ok=True)

        # Seed everything
        seed_everything(self.conf.seed)

    # -------------------------------------------------------------------------
    # preprocess
    # -------------------------------------------------------------------------
    def preprocess(self):
        """
        Perform general preprocessing.
        TODO: FIX TREE MASK TO TAKE ON LANDCOVER INPUT IF MORE THAN TWO CLASSES
        """
        logging.info('Starting preprocessing stage')

        # Generate WorldView vs. ICESAT-2 footprint (geopackages with matches)
        self.gen_intersection_database()

        # Iterate over the previously saved intersection files, generate tiles
        self.gen_dataset_tiles()

        # Calculate mean and std values for training
        data_filenames = get_dataset_filenames(self.images_dir)
        label_filenames = get_dataset_filenames(self.labels_dir)
        logging.info(f'Mean and std values from {len(data_filenames)} files.')

        # Temporarily disable standardization and augmentation
        current_standardization = self.conf.standardization
        self.conf.standardization = None
        metadata_output_filename = os.path.join(
            self.model_dir, f'mean-std-{self.conf.experiment_name}.csv')
        os.makedirs(self.model_dir, exist_ok=True)

        # Set main data loader
        main_data_loader = RegressionDataLoader(
            data_filenames, label_filenames, self.conf, False
        )

        # Get mean and std array
        mean, std = get_mean_std_dataset(
            main_data_loader.train_dataset, metadata_output_filename)
        logging.info(f'Mean: {mean.numpy()}, Std: {std.numpy()}')

        # Re-enable standardization for next pipeline step
        self.conf.standardization = current_standardization

        logging.info('Done with preprocessing stage')

    # -------------------------------------------------------------------------
    # predict
    # -------------------------------------------------------------------------
    def predict(self) -> None:

        logging.info('Starting prediction stage')

        # Load model for inference
        model = load_model(
            model_filename=self.conf.model_filename,
            model_dir=self.model_dir
        )

        # Retrieve mean and std, there should be a more ideal place
        if self.conf.standardization in ["global", "mixed"]:
            mean, std = get_mean_std_metadata(
                os.path.join(
                    self.model_dir,
                    f'mean-std-{self.conf.experiment_name}.csv'
                )
            )
            logging.info(f'Mean: {mean}, Std: {std}')
        else:
            mean = None
            std = None

        # Gather filenames to predict
        if len(self.conf.inference_regex_list) > 0:
            data_filenames = self.get_filenames(self.conf.inference_regex_list)
        else:
            data_filenames = self.get_filenames(self.conf.inference_regex)
        logging.info(f'{len(data_filenames)} files to predict')

        # iterate files, create lock file to avoid predicting the same file
        for filename in sorted(data_filenames):

            # start timer
            start_time = time.time()

            # set output directory
            basename = os.path.basename(os.path.dirname(filename))
            if basename == 'M1BS' or basename == 'P1BS':
                basename = os.path.basename(
                    os.path.dirname(os.path.dirname(filename)))

            output_directory = os.path.join(
                self.conf.inference_save_dir, basename)
            os.makedirs(output_directory, exist_ok=True)

            # TEMPORARY WHILE WE RUN MORE MASKS FOR THE NEW TAPPANS
            tree_mask_regex = glob(
                os.path.join(
                    self.conf.tree_mask_dir, basename,
                    f"{Path(filename).stem}*.tif")
            )
            if len(tree_mask_regex) == 0:
                continue

            # set prediction output filename
            output_filename = os.path.join(
                output_directory,
                f'{Path(filename).stem}.{self.conf.experiment_type}.tif')

            # lock file for multi-node, multi-processing
            lock_filename = f'{output_filename}.lock'

            # predict only if file does not exist and no lock file
            if not os.path.isfile(output_filename) and \
                    not os.path.isfile(lock_filename):

                try:

                    logging.info(f'Starting to predict {filename}')

                    # create lock file
                    open(lock_filename, 'w').close()

                    # open filename
                    image = rxr.open_rasterio(filename)
                    logging.info(f'Prediction shape: {image.shape}')

                except rasterio.errors.RasterioIOError:
                    logging.info(f'Skipped {filename}, probably corrupted.')
                    continue

                # Calculate indices and append to the original raster
                image = indices.add_indices(
                    xraster=image, input_bands=self.conf.input_bands,
                    output_bands=self.conf.output_bands)

                # Modify the bands to match inference details
                image = modify_bands(
                    xraster=image, input_bands=self.conf.input_bands,
                    output_bands=self.conf.output_bands)
                logging.info(f'Prediction shape after modf: {image.shape}')

                # Transpose the image for channel last format
                image = image.transpose("y", "x", "band")

                # Remove no-data values to account for edge effects
                temporary_tif = xr.where(image > -100, image, 600)

                # Sliding window prediction
                prediction = regression_inference.sliding_window_tiler(
                    xraster=temporary_tif,
                    model=model,
                    n_classes=self.conf.n_classes,
                    overlap=self.conf.inference_overlap,
                    batch_size=self.conf.pred_batch_size,
                    standardization=self.conf.standardization,
                    mean=mean,
                    std=std,
                    normalize=self.conf.normalize,
                    window=self.conf.window_algorithm
                )
                prediction[prediction < 0] = 0
                prediction = prediction * 250

                # get and apply water mask
                worldcover_mask_ma = self.get_water_mask(filename)
                prediction[worldcover_mask_ma == 80] = \
                    self.conf.prediction_nodata
                prediction[prediction > 200] = 0

                # get and apply tree mask
                tree_mask_ma = self.get_tree_mask(filename)
                prediction[tree_mask_ma != 1] = 0

                logging.info(
                    f'Prediction min: {prediction.min()}, ' +
                    f'max: {prediction.max()}, mean: {prediction.mean()}')

                # Drop image band to allow for a merge of mask
                image = image.drop(
                    dim="band",
                    labels=image.coords["band"].values[1:],
                )

                # Get metadata to save raster
                prediction = xr.DataArray(
                    np.expand_dims(prediction, axis=-1),
                    name=self.conf.experiment_type,
                    coords=image.coords,
                    dims=image.dims,
                    attrs=image.attrs
                )

                # Add metadata to raster attributes
                prediction.attrs['long_name'] = (self.conf.experiment_type)
                prediction.attrs['model_name'] = (self.conf.model_filename)
                prediction = prediction.transpose("band", "y", "x")

                # Set nodata values on mask
                nodata = prediction.rio.nodata
                prediction = prediction.where(image != nodata)
                prediction.rio.write_nodata(
                    self.conf.prediction_nodata, encoded=True, inplace=True)

                # Save output raster file to disk
                prediction.rio.to_raster(
                    output_filename,
                    BIGTIFF="IF_SAFER",
                    compress=self.conf.prediction_compress,
                    driver=self.conf.prediction_driver,
                    dtype=self.conf.prediction_dtype
                )
                del prediction

                # create symlink to the data
                os.symlink(
                    filename,
                    os.path.join(output_directory, os.path.basename(filename))
                )

                # delete lock file
                try:
                    os.remove(lock_filename)
                except FileNotFoundError:
                    logging.info(f'Lock file not found {lock_filename}')
                    continue

                logging.info(f'Finished processing {output_filename}')
                logging.info(f"{(time.time() - start_time)/60} min")

            # This is the case where the prediction was already saved
            else:
                logging.info(f'{output_filename} already predicted.')

    # -------------------------------------------------------------------------
    # utilities
    # -------------------------------------------------------------------------
    def gen_intersection_database(self):
        """
        Generate intersection filenames
        """
        if self.conf.footprint:

            # get ATL08 dataframe points
            atl08_gdf = self.get_atl08_gdf(
                self.conf.atl08_dir,
                self.conf.atl08_start_year,
                self.conf.atl08_end_year,
            )
            logging.info(f'Load ATL08 GDF files, {atl08_gdf.shape[0]} rows.')

            # some EDA information to catch non-correct labels
            logging.info(
                f'Before postprocessing {self.conf.label_attribute} - ' +
                f'min: {atl08_gdf[self.conf.label_attribute].min()}, ' +
                f'max: {atl08_gdf[self.conf.label_attribute].max()}, ' +
                f'mean: {atl08_gdf[self.conf.label_attribute].mean()}'
            )

            # filter labels for correct heights, no tree is taller than 250m
            atl08_gdf = atl08_gdf.loc[
                atl08_gdf[self.conf.label_attribute] < 250]

            logging.info(
                f'After postprocessing {self.conf.label_attribute} - ' +
                f'min: {atl08_gdf[self.conf.label_attribute].min()}, ' +
                f'max: {atl08_gdf[self.conf.label_attribute].max()}, ' +
                f'mean: {atl08_gdf[self.conf.label_attribute].mean()}'
            )
            logging.info(f'ATL08 without no-data, {atl08_gdf.shape[0]} rows.')

            # Get WorldView filenames
            wv_evhr_filenames = self.get_filenames(self.conf.wv_data_regex)
            logging.info(f'WorldView files, {len(wv_evhr_filenames)} files')

            # Filter list by year range
            wv_evhr_filenames = self.filter_raster_filenames_by_year(
                wv_evhr_filenames,
                self.conf.atl08_start_year,
                self.conf.atl08_end_year
            )
            logging.info(
                f'WorldView after year filter, {len(wv_evhr_filenames)} files')

            # Intersection of the two GDBs, output geopackages files
            self.filter_polygon_in_raster_parallel(
                wv_evhr_filenames, atl08_gdf, self.intersection_dir
            )

            # Gather intersection filenames
            intersection_files = glob(
                os.path.join(self.intersection_dir, '*', '*.gpkg'))
            logging.info(
                f'{len(intersection_files)} WorldView matches saved ' +
                f'in {self.intersection_dir}.'
            )
        else:
            logging.info(
                'Skipping WV-ATL08 intersection since it has been done.')
        return

    def gen_dataset_tiles(self):
        """
        Generate dataset training tiles.
        """
        if self.conf.tiles:

            # list dataset filenames from disk
            intersection_filenames = sorted(glob(
                os.path.join(self.intersection_dir, '*', '*.gpkg')))
            assert len(intersection_filenames) > 0, \
                f"No gpkg files found under {self.intersection_dir}."

            logging.info(
                f'Generating tiles from {len(intersection_filenames)} rasters')

            self.gen_tiles_from_gpkg_parallel(
                intersection_filenames,
                self.conf.tile_size,
                self.conf.input_bands,
                self.conf.output_bands,
                self.images_dir,
                self.labels_dir,
                self.conf.label_attribute,
                self.conf.tree_mask_dir,
                self.conf.cloud_mask_dir,
                self.conf.mask_preprocessing
            )
        return

    def gen_tiles_from_gpkg(
                self,
                gpd_iter,
                tile_size: int,
                input_bands: list,
                output_bands: list,
                images_output_dir: str,
                labels_output_dir: str,
                label_attribute: str = 'h_can',
                tree_mask_dir: str = None,
                cloud_mask_dir: str = None,
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

            # get the number of bands we want to train on
            data_array = modify_bands(
                data_array, input_bands, output_bands)

            # this if statement takes care of the experiment
            # where labels utilize landcover mask as a primer
            if tree_mask_dir is not None and mask_preprocessing:

                # set mask filename
                mask_filename = os.path.join(
                    tree_mask_dir, row['study_area'],
                    f"{Path(row['scene_id']).stem}.trees.tif")

                # open raster via GDAL
                mask_dataset = osgeo.gdal.Open(mask_filename)

                # extract data array from the greater raster
                mask_array = extract_centered_window(
                    mask_dataset, window_pixel_x, window_pixel_y,
                    tile_size, tile_size
                )

                if cloud_mask_dir is not None:

                    # get filename from cloudmask
                    cloud_mask_filename = os.path.join(
                        cloud_mask_dir, row['study_area'],
                        f"{Path(row['scene_id']).stem}.cloudmask.tif")

                    # read cloudmask dataset
                    cloudmask_dataset = osgeo.gdal.Open(cloud_mask_filename)

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

                # if label is negative, or higher than tallest tree
                if mask_array.max() <= 0 or mask_array.max() > 250:
                    return

            else:
                sys.exit(
                    'You need to specify tree_mask_dir and mask_preprocessing')

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

            # Expand tiles
            data_array = np.moveaxis(data_array, 0, -1)
            mask_array = np.expand_dims(mask_array, -1)

            # output to disk
            np.save(output_data_filename, data_array)
            np.save(output_label_filename, mask_array)

        except (AttributeError, IndexError) as e:
            logging.info(e)
        return

    def gen_tiles_from_gpkg_parallel(
                self,
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
        """Extract tiles from gpkg.

        Args:
            intersection_filenames (list): _description_
            tile_size (int): _description_
            input_bands (list): _description_
            output_bands (list): _description_
            images_output_dir (str): _description_
            labels_output_dir (str): _description_
            label_attribute (str, optional): \
                _description_. Defaults to 'h_can'.
            mask_dir (str, optional): _description_. Defaults to None.
            cloudmask_dir (str, optional): \
                _description_. Defaults to None.
            mask_preprocessing (bool, optional): \
                _description_. Defaults to False.
            n_processes (_type_, optional): \
                _description_. Defaults to cpu_count().
        """
        for intersection_filename in tqdm(intersection_filenames):

            # open the filename
            dataset_gdf = gpd.read_file(intersection_filename)

            p = Pool(processes=n_processes)
            p.starmap(
                self.gen_tiles_from_gpkg,
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

    def get_water_mask(self, filename):
        """
        Return WorldCover water mask
        """
        # warp land cover to worldview resolution
        warp_landcover_list = warplib.memwarp_multi_fn(
            [self.conf.landcover_mask_virt],
            res=filename,
            extent=filename,
            t_srs=filename,
            r='mode',
            dst_ndv=self.conf.prediction_nodata
        )
        worldcover_warp_ma = iolib.ds_getma(warp_landcover_list[0])
        return worldcover_warp_ma

    def get_tree_mask(self, filename):
        """
        Return tree mask
        """
        # get basename from dataset
        basename = os.path.basename(os.path.dirname(filename))
        if basename == 'M1BS' or basename == 'P1BS':
            basename = os.path.basename(
                os.path.dirname(os.path.dirname(filename)))

        # regex to find file that matches
        tree_mask_regex = glob(
            os.path.join(
                self.conf.tree_mask_dir, basename,
                f"{Path(filename).stem}*.tif")
        )
        if len(tree_mask_regex) == 0:
            sys.exit(f'Could not find tree mask with {tree_mask_regex}')
        else:
            tree_mask = np.squeeze(
                rxr.open_rasterio(tree_mask_regex[0]).values)
            tree_mask[tree_mask != 1] = 0
        return tree_mask
