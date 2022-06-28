# --------------------------------------------------------------------------
# Preprocessing and dataset creation from vhr data. This assumes you provide
# a configuration file with required parameters and files.
# --------------------------------------------------------------------------
import os
import sys
import logging
import argparse
import omegaconf
from glob import glob

from vhr_cnn_chm.config import CHMConfig as Config
from vhr_cnn_chm.utils import get_atl08_gdf, get_wv_evhr_gdf, \
    filter_gdf_by_list, filter_polygon_in_raster_parallel, \
    extract_tiles_parallel
from tensorflow_caney.utils.system import seed_everything


CHUNKS = {'band': 'auto', 'x': 'auto', 'y': 'auto'}

__status__ = "Development"


# ---------------------------------------------------------------------------
# script preprocess.py
# ---------------------------------------------------------------------------
def run(args, conf) -> None:

    logging.info('Starting main preprocessing method.')

    # define output directories
    metadata_output_dir = os.path.join(conf.data_dir, 'metadata')
    intersection_output_dir = os.path.join(metadata_output_dir, 'intersection')
    images_output_dir = os.path.join(conf.data_dir, 'images')
    labels_output_dir = os.path.join(conf.data_dir, 'labels')

    # create output directories
    for odir in [
            images_output_dir, labels_output_dir, intersection_output_dir]:
        os.makedirs(odir, exist_ok=True)

    if conf.footprint:

        # get ATL08 dataframe points
        atl08_gdf = get_atl08_gdf(
            conf.atl08_dir,
            conf.atl08_start_year,
            conf.atl08_end_year,
            crs=conf.general_crs
        )
        logging.info(f'Load ATL08 GDF files, {atl08_gdf.shape[0]} rows.')

        # some EDA information
        logging.info(f'min: {atl08_gdf[conf.label_attribute].min()}')
        logging.info(f'max: {atl08_gdf[conf.label_attribute].max()}')
        logging.info(f'mean: {atl08_gdf[conf.label_attribute].mean()}')

        # Read WorldView footprints database
        wv_evhr_gdf = get_wv_evhr_gdf(
            conf.wv_data_regex, crs=conf.general_crs)
        logging.info(f'Load WorldView GDF, {wv_evhr_gdf.shape[0]} rows.')

        # Filter GDF by year range
        wv_evhr_gdf = filter_gdf_by_list(
            wv_evhr_gdf, 'wv_year', list(
                range(conf.atl08_start_year, conf.atl08_end_year))
        )
        logging.info(
            f'Filter WorldView GDF by year, {wv_evhr_gdf.shape[0]} rows.')

        # Intersection of the two GDBs, output geopackages files
        filter_polygon_in_raster_parallel(
            wv_evhr_gdf, atl08_gdf, intersection_output_dir
        )

    else:
        logging.info(
            'Skipping WV-ATL08 intersection since it has been done.')

    # iterate over the previously saved intersection files
    # extract tiles fully in parallel with multi-core parallelization
    if conf.tiles:

        # list dataset filenames from disk
        intersection_filenames = glob(
            os.path.join(intersection_output_dir, '*', '*.gpkg'))
        assert len(intersection_filenames) > 0, \
            f"No gpkg files found under {intersection_output_dir}."

        extract_tiles_parallel(
            intersection_filenames, conf.tile_size,
            images_output_dir, labels_output_dir, conf.label_attribute,
            conf.mask_dir, conf.mask_preprocessing
        )

    logging.info('Done with preprocessing stage')

    return


def main() -> None:

    # Process command-line args.
    desc = 'Use this application to map LCLUC in Senegal using WV data.'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('-c',
                        '--config-file',
                        type=str,
                        required=True,
                        dest='config_file',
                        help='Path to the configuration file')

    args = parser.parse_args()

    # Logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s; %(levelname)s; %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Configuration file intialization
    schema = omegaconf.OmegaConf.structured(Config)
    conf = omegaconf.OmegaConf.load(args.config_file)
    try:
        conf = omegaconf.OmegaConf.merge(schema, conf)
    except BaseException as err:
        sys.exit(f"ERROR: {err}")

    # Seed everything
    seed_everything(conf.seed)

    # Call run for preprocessing steps
    run(args, conf)

    return


# -------------------------------------------------------------------------------
# main
# -------------------------------------------------------------------------------

if __name__ == "__main__":

    main()
