import os
import time
import glob
import logging
import fiona
import rasterio
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxr
import geopandas as gpd
import shapely.speedups
from omegaconf.listconfig import ListConfig
from multiprocessing import Pool, cpu_count

# import cuspatial
import tensorflow as tf
from pathlib import Path
from shapely.geometry import box
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error

from .geoscitools.atl08lib import atl08_io
from .cnn_model import get_2d_cnn_tf

from tensorflow_caney.config.cnn_config import Config
from tensorflow_caney.utils.system import seed_everything, set_gpu_strategy
from tensorflow_caney.utils.system import set_mixed_precision, set_xla
from tensorflow_caney.utils.data import get_dataset_filenames
from tensorflow_caney.utils.regression_tools import RegressionDataLoader

from tensorflow_caney.utils.losses import get_loss
from tensorflow_caney.utils.optimizers import get_optimizer
from tensorflow_caney.utils.metrics import get_metrics
from tensorflow_caney.utils.callbacks import get_callbacks
from tensorflow_caney.utils.model import get_model
from tensorflow_caney.inference import regression_inference

from tensorflow_caney.networks.unet_regression import unet_batchnorm_regression

from tensorflow.python.keras import losses
from tensorflow.python.keras import optimizers


shapely.speedups.enable()


# -------------------------------------------------------------------------------
# class CNNPipeline
#
# This class preprocesses, trains and predicts rasters through regression.
# -------------------------------------------------------------------------------

class CNNRegressionPipeline(object):

    # ---------------------------------------------------------------------------
    # TODO:
    # ---------------------------------------------------------------------------
    # - add cuspatial to the processing mix, cuspatial.point_in_polygon()
    # - get container with cuspatial installed, using tensorflow as well
    # - get actual data tiles from WorldView
    # - make get points in polygon run in parallel
    # - filter non WorldView satellites
    # - ideally, output of filter should be a single gpkg with wv filename
    # as column

    # ---------------------------------------------------------------------------
    # __init__
    # ---------------------------------------------------------------------------
    def __init__(self, conf, n_processes=cpu_count()):

        self.working_dir = 'output'
        self.conf = conf
        self.n_processes = n_processes

    # ---------------------------------------------------------------------------
    # methods
    # ---------------------------------------------------------------------------

    # ---------------------------------------------------------------------------
    # input methods
    # ---------------------------------------------------------------------------
    def get_atl08_gdf(
                self, data_dir: str,
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
        atl08_gdf = pd.concat([
            atl08_io(data_dir, str(year), do_pickle=False)
            for year in range(start_year, end_year)
        ])

        if crs is not None:
            atl08_gdf = atl08_gdf.to_crs(epsg=crs.split(':')[-1])
        return atl08_gdf.reset_index(drop=True)

    def get_wv_evhr_gdf(self, data_regex: str, crs: str = None):
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
        scene_ids_list, bounds_ids_list, years_ids_list, \
            study_area_list = [], [], [], []

        if crs is None:
            crs = rasterio.open(filenames[0]).crs

        for f in filenames:
            study_area_list.append(f.split('/')[-3])
            scene_ids_list.append(f)
            years_ids_list.append(int(Path(f).stem[5:9]))
            bounds_ids_list.append(box(*rasterio.open(f).bounds))

        d = {
            'study_area': study_area_list,
            'scene_id': scene_ids_list,
            'acq_year': years_ids_list,
            'geometry': bounds_ids_list
        }
        return gpd.GeoDataFrame(d, crs=crs)

    def get_gdb_gdf(self, gdb_filename: str, gdb_layer: str = None):
        """
        Get GDF from GDB.
        """
        # List layers from DB
        gdb_layers = fiona.listlayers(gdb_filename)

        # Take the layer to read from GDB, if None, take default layer
        if gdb_layer is None:
            gdb_layer = gdb_layers[0]
            logging.warning(f'No gdb_layer given, {gdb_layer} layer loaded.')

        # Read GDB and return as a GDF object
        return gpd.read_file(gdb_filename, layer=gdb_layer)

    def get_point_in_polygon_by_scene(self):
        """
        Return overalapping points
        """
        logging.info(f"Iterating over {self.wv_evhr_gdf.shape[0]} polygons")
        logging.info(f"Iterating over {self.atl08_gdf.shape[0]} points")

        # multiprocessing pool
        p = Pool(processes=self.n_processes)
        p.map(
            self.filter_single_point_in_polygon_by_scene,
            self.wv_evhr_gdf.iterrows()
        )
        p.close()
        p.join()
        return

    def get_wv_tile_data(self, atl08_gpkg_filename):
        """
        Extract data tiles based on ATL08 point footprint.
        """
        # read the data
        atl08_data = gpd.read_file(
            atl08_gpkg_filename, layer='ATL08_WorldView')
        logging.info(f'Processing {atl08_gpkg_filename}')

        # buffer point geometry for tile size
        atl08_data = atl08_data.to_crs(epsg=3395)
        buffer_series = atl08_data.buffer(self.conf.tile_buffer, cap_style=3)
        atl08_data['geometry'] = buffer_series
        atl08_data = atl08_data.to_crs(
            epsg=self.conf.general_crs.split(':')[-1])

        # here we extract the point extent
        for _, poly_row in atl08_data.iterrows():

            # read raster image
            raster_data = rxr.open_rasterio(poly_row['scene_id'])

            # TODO: change last regex for something else
            # not the static trees.tif
            location = poly_row['scene_id'].split('/')[-3]

            if self.conf.landcover_directory is not None:
                mask_filename = os.path.join(
                    self.conf.landcover_directory, location,
                    f"{Path(poly_row['scene_id']).stem}.trees.tif")
                raster_mask = rxr.open_rasterio(mask_filename)

            # clip buffered tile from the raster
            clipped_data = raster_data.rio.clip(
                [poly_row['geometry']], raster_data.rio.crs)
            clipped_mask = raster_mask.rio.clip(
                [poly_row['geometry']], raster_mask.rio.crs)
            clipped_data.attrs['h_can'] = poly_row['h_can']
            clipped_mask.attrs['h_can'] = poly_row['h_can']

            # multiply label to mask, since 1 are trees, these will
            # be the only greater than 0 values
            clipped_mask = clipped_mask * poly_row['h_can']

            # set output_dir and create directory
            output_dir = os.path.join(
                self.conf.tiles_output_dir,
                location
            )
            output_dir_data = os.path.join(output_dir, 'image')
            output_dir_label = os.path.join(output_dir, 'label')

            os.makedirs(output_dir_data, exist_ok=True)
            os.makedirs(output_dir_label, exist_ok=True)

            # output to raster tile
            scene = Path(poly_row["scene_id"]).stem

            # validate size of the tile
            clipped_data = clipped_data.data
            clipped_mask = clipped_mask.data

            if clipped_data.shape[1] < self.conf.tile_size \
                    or clipped_data.shape[2] < self.conf.tile_size:
                continue

            if clipped_data.min() < -100:
                continue

            np.save(
                os.path.join(
                    output_dir_data, f'{scene}_hcan_{poly_row["h_can"]}.npy'
                ), clipped_data)
            np.save(
                os.path.join(
                    output_dir_label, f'{scene}_hcan_{poly_row["h_can"]}.npy'
                ), clipped_mask)

            # clipped_data.rio.to_raster(
            #    os.path.join(
            #        output_dir_data, f'{scene}_hcan_{poly_row["h_can"]}.tif'
            #    ), compress='LZW', dtype="int16"
            # )
            # clipped_mask.rio.to_raster(
            #    os.path.join(
            #        output_dir_label, f'{scene}_hcan_{poly_row["h_can"]}.tif'
            #    ), compress='LZW', dtype="float32"
            # )

        return

    def get_data_from_gpkg(self, atl08_gpkg_regex: str):
        """
        Get data filenames from ATL08 GPKG.
        """
        atl08_intersection_filenames = glob.glob(atl08_gpkg_regex)
        logging.info(f'Processing {len(atl08_intersection_filenames)} files.')

        # multiprocessing pool
        p = Pool(processes=self.n_processes)
        p.map(self.get_wv_tile_data, atl08_intersection_filenames)
        p.close()
        p.join()
        return

    # ---------------------------------------------------------------------------
    # preprocess methods
    # ---------------------------------------------------------------------------
    def filter_gdf_by_list(
                self,
                gdf,
                gdf_key: str = 'acq_year',
                isin_list: list = [],
                reset_index: bool = True
            ):
        """
        Filter GDF by year range.
        """
        return gdf[gdf[gdf_key].isin(isin_list)].reset_index(drop=True)

    def filter_single_point_in_polygon_by_scene(self, polygon_row):
        """
        Multiprocessing wrapper to process a single point/polygon row.
        """
        # get polygon row metadata, get indices of points within intersection
        _, polygon_row = polygon_row
        points_mask = self.atl08_gdf['geometry'].within(
            polygon_row['geometry'])
        points_mask = points_mask.to_frame(name='IN_POLYGON')
        points_mask = points_mask.index[
            points_mask['IN_POLYGON'] == True].tolist()

        if points_mask:

            # localize the list of indices with intersection points
            points_mask = self.atl08_gdf.loc[points_mask]
            points_mask['scene_id'] = polygon_row['scene_id']

            # set output_dir and create directory
            output_dir = os.path.join(
                self.conf.intersection_output_dir, polygon_row['study_area'])
            os.makedirs(output_dir, exist_ok=True)

            # save geopackage file within the output_dir
            points_mask.to_file(
                os.path.join(
                    output_dir, f"{Path(polygon_row['scene_id']).stem}.gpkg"),
                driver='GPKG', layer='ATL08_WorldView'
            )
        return

    # ---------------------------------------------------------------------------
    # train methods
    # ---------------------------------------------------------------------------
    def get_data_labels(self, data_filenames: str):
        """
        Get data and labels arrays.
        """
        data_list = []
        labels_list = []
        for data_filename in data_filenames:

            tile = rxr.open_rasterio(data_filename).values
            h_can = float(data_filename.split('_')[-1][:-4])

            tile = np.moveaxis(tile, 0, -1)
            tile = resize(tile, (128, 128))

            if tile.min() < -100:
                continue

            data_list.append(tile)
            labels_list.append(h_can)

            # couple of augmentation steps
            if np.random.random_sample() > 0.5:
                tile = np.fliplr(tile)
                data_list.append(tile)
                labels_list.append(h_can)
            if np.random.random_sample() > 0.5:
                tile = np.flipud(tile)
                data_list.append(tile)
                labels_list.append(h_can)
            if np.random.random_sample() > 0.5:
                tile = np.rot90(tile, 1)
                data_list.append(tile)
                labels_list.append(h_can)
            if np.random.random_sample() > 0.5:
                tile = np.rot90(tile, 2)
                data_list.append(tile)
                labels_list.append(h_can)
            if np.random.random_sample() > 0.5:
                tile = np.rot90(tile, 3)
                data_list.append(tile)
                labels_list.append(h_can)

        return np.array(data_list), pd.DataFrame(labels_list)

    def sliding_window_inference(self):
        """
        Sliding window inference.
        """

        return

    # ---------------------------------------------------------------------------
    # main methods
    # ---------------------------------------------------------------------------
    def preprocess(self):
        """
        Preprocessing method for the pipeline.
        """
        logging.info('Starting main preprocessing method.')

        # Read ATL08 points
        self.atl08_gdf = self.get_atl08_gdf(
            self.conf.atl08_dir,
            self.conf.atl08_start_year,
            self.conf.atl08_end_year,
            crs=self.conf.general_crs
        )
        logging.info(f'Load ATL08 GDF files, {self.atl08_gdf.shape[0]} rows.')

        # Read WorldView footprints database
        self.wv_evhr_gdf = self.get_wv_evhr_gdf(
            self.conf.wv_data_regex, crs=self.conf.general_crs)
        logging.info(f'Load WorldView GDF, {self.wv_evhr_gdf.shape[0]} rows.')

        # Filter GDF by year range
        self.wv_evhr_gdf = self.filter_gdf_by_list(
            self.wv_evhr_gdf, 'acq_year', list(
                range(self.conf.atl08_start_year, self.conf.atl08_end_year)))
        logging.info(
            f'Filter WorldView GDF by year, {self.wv_evhr_gdf.shape[0]} rows.')

        # Intersection of the two GDBs, output geopackages files
        if len(os.listdir(self.conf.intersection_output_dir)) == 0:
            self.get_point_in_polygon_by_scene()
        else:
            logging.info(
                'Skipping WV-ATL08 intersection since it has been done.')

        # Extract tiles from geopackages that have overlapping points
        self.get_data_from_gpkg(
            atl08_gpkg_regex=os.path.join(
                self.conf.intersection_output_dir, '*/*.gpkg'))
        logging.info('Done with main preprocessing method.')

        return

    def train(self):
        """
        Training method for the pipeline.
        """
        logging.info('Starting training stage')

        # set data variables for directory management
        images_dir = os.path.join(self.conf.data_dir, 'image')
        labels_dir = os.path.join(self.conf.data_dir, 'label')

        # Set and create model directory
        model_dir = os.path.join(self.conf.model_dir, 'model')
        os.makedirs(model_dir, exist_ok=True)

        # Set hardware acceleration options
        gpu_strategy = set_gpu_strategy(self.conf.gpu_devices)
        set_mixed_precision(self.conf.mixed_precision)
        set_xla(self.conf.xla)

        # Get data and label filenames for training
        data_filenames = get_dataset_filenames(images_dir)
        label_filenames = get_dataset_filenames(labels_dir)
        assert len(data_filenames) == len(label_filenames), \
            'Number of data and label filenames do not match'

        logging.info(
            f'Data: {len(data_filenames)}, Label: {len(label_filenames)}')

        # Set main data loader
        main_data_loader = RegressionDataLoader(
            data_filenames, label_filenames, self.conf
        )

        # Set multi-GPU training strategy
        with gpu_strategy.scope():

            # Get and compile the model
            model = get_model(self.conf.model)

            # loss = losses.get('MeanSquaredError')
            # optimizer = optimizers.get('SGD')

            model.compile(
                loss=get_loss(self.conf.loss),
                optimizer=get_optimizer(
                    self.conf.optimizer)(self.conf.learning_rate),
                metrics=get_metrics(self.conf.metrics)
            )
            model.summary()

        # Fit the model and start training
        model.fit(
            main_data_loader.train_dataset,
            validation_data=main_data_loader.val_dataset,
            epochs=self.conf.max_epochs,
            steps_per_epoch=main_data_loader.train_steps,
            validation_steps=main_data_loader.val_steps,
            callbacks=get_callbacks(self.conf.callbacks)
        )

        # Close multiprocessing Pools from the background
        # atexit.register(gpu_strategy._extended._collective_ops._pool.close)

        """
        ypred = model.predict(testImagesX)
        print(model.evaluate(testImagesX, testAttrY))
        print("MSE: %.4f" % mean_squared_error(testAttrY, ypred))
        """
        return

    def predict(self):
        """
        Prediction method for the pipeline.
        """
        logging.info("Full scene prediction, smoothing via tiles")

        # questions, how to smooth between locations for better regression
        # weighted prediction based on landcover input

        logging.info('Starting prediction stage')

        # Set and create model directory
        os.makedirs(self.conf.inference_save_dir, exist_ok=True)

        # Load model
        model = tf.keras.models.load_model(self.conf.model_filename)
        model.summary()

        # Gather filenames to predict
        data_filenames = sorted(glob.glob(self.conf.inference_regex))
        assert len(data_filenames) > 0, \
            f'No files under {self.conf.inference_regex}.'
        logging.info(f'{len(data_filenames)} files to predict')

        # iterate files, create lock file to avoid predicting the same file
        for filename in data_filenames:

            start_time = time.time()

            # output filename to save prediction on
            output_filename = os.path.join(
                self.conf.inference_save_dir,
                f'{Path(filename).stem}.{self.conf.experiment_type}.tif'
            )

            # lock file for multi-node, multi-processing
            lock_filename = f'{output_filename}.lock'

            # predict only if file does not exist and no lock file
            if not os.path.isfile(output_filename) and \
                not os.path.isfile(lock_filename):

                logging.info(f'Starting to predict {filename}')

                # create lock file - remove while testing
                # open(lock_filename, 'w').close()

                # open filename
                image = rxr.open_rasterio(filename)
                logging.info(f'Prediction shape: {image.shape}')

                # Calculate indices and append to the original raster
                # image = indices.add_indices(
                #    xraster=image, input_bands=self.conf.input_bands,
                #    output_bands=self.conf.output_bands)

                # Modify the bands to match inference details
                # image = modify_bands(
                #    xraster=image, input_bands=self.conf.input_bands,
                #    output_bands=self.conf.output_bands)
                # logging.info(f'Prediction shape after modf: {image.shape}')

                # Transpose the image for channel last format
                image = image.transpose("y", "x", "band")

                # Remove no-data values to account for edge effects
                # temporary_tif = image.values
                temporary_tif = xr.where(image > -100, image, 600)
                # temporary_tif = temporary_tif / 10000.0

                prediction = regression_inference.sliding_window_tiler(
                    xraster=temporary_tif,
                    model=model,
                    n_classes=self.conf.n_classes,
                    overlap=0.50,
                    batch_size=self.conf.pred_batch_size,
                    standardization=self.conf.standardization,
                    mean=self.conf.mean,
                    std=self.conf.std,
                    normalize=self.conf.normalize
                )
                print(prediction.min(), prediction.max())

                prediction = prediction * 100

                # landcover = rxr.open_rasterio(
                #    '/adapt/nobackup/projects/ilab/projects/Senegal/3sl/products/land_cover/dev/tcbo.v1/CASTest/Tappan01_WV02_20110430_M1BS_103001000A27E100_data.landcover.tif')
                # landcover = np.squeeze(landcover.values)
                # print("UNIQUE LAND COVER", np.unique(landcover))

                # landcover[landcover > 1] = 0

                # prediction = prediction * landcover

                #    overlap=0.20,
                #    batch_size=conf.pred_batch_size,
                #    standardization=conf.standardization
                # )
                # logging.info(f'Prediction unique values {np.unique(prediction)}')
                # logging.info(f'Done with prediction')

                # Drop image band to allow for a merge of mask
                image = image.drop(
                    dim="band",
                    labels=image.coords["band"].values[1:],
                    drop=True
                )

                # Get metadata to save raster
                prediction = xr.DataArray(
                    np.expand_dims(prediction, axis=-1),
                    name=self.conf.experiment_type,
                    coords=image.coords,
                    dims=image.dims,
                    attrs=image.attrs
                )
                prediction.attrs['long_name'] = (self.conf.experiment_type)
                prediction.attrs['model_name'] = (self.conf.model_filename)
                prediction = prediction.transpose("band", "y", "x")

                # Set nodata values on mask
                nodata = prediction.rio.nodata
                prediction = prediction.where(image != nodata)
                prediction.rio.write_nodata(nodata, encoded=True, inplace=True)

                # Save COG file to disk
                prediction.rio.to_raster(
                    output_filename, BIGTIFF="IF_SAFER", compress='LZW',
                    num_threads='all_cpus'  # , driver='COG'
                )

                del prediction

                # delete lock file
                # os.remove(lock_filename)

                logging.info(f'Finished processing {output_filename}')
                logging.info(f"{(time.time() - start_time)/60} min")
            # This is the case where the prediction was already saved
            else:
                logging.info(f'{output_filename} already predicted.')

        # Close multiprocessing Pools from the background
        # atexit.register(gpu_strategy._extended._collective_ops._pool.close)

        return
