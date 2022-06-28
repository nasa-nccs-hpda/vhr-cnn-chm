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
                    normalize=self.conf.normalize,
                    window='boxcar'
                )
                print(prediction.min(), prediction.max())

                #prediction = prediction * 100

                # apply landcover postprocessing mask to output


                #landcover = rxr.open_rasterio(
                #    '/adapt/nobackup/projects/ilab/projects/Senegal/3sl/products/land_cover/dev/tcbo.v1/CASTest/Tappan01_WV02_20110430_M1BS_103001000A27E100_data.landcover.tif')
                #landcover = np.squeeze(landcover.values)
                #print("UNIQUE LAND COVER", np.unique(landcover))
                #landcover[landcover > 1] = 0
                #prediction = prediction * landcover

                """
                # output filename to save prediction on
                landcover_filename = os.path.join(
                    '/adapt/nobackup/projects/ilab/projects/Senegal/3sl/products/land_cover/dev/trees.v2/Tappan',
                    f'{Path(filename).stem}.trees.tif'
                )
                landcover = rxr.open_rasterio(landcover_filename)
                landcover = np.squeeze(landcover.values)
                landcover[landcover > 1] = 0
                prediction = prediction * landcover
                """

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
                # prediction.rio.write_nodata(nodata, encoded=True, inplace=True)
                prediction.rio.write_nodata(255, encoded=True, inplace=True)

                # Save COG file to disk
                prediction.rio.to_raster(
                    output_filename, BIGTIFF="IF_SAFER", compress='LZW',
                    num_threads='all_cpus', dtype='float32'#, driver='COG'
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