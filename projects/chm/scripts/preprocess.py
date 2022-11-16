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
from vhr_cnn_chm.utils import get_atl08_gdf, \
    filter_polygon_in_raster_parallel, \
    extract_tiles_parallel, get_filenames, filter_evhr_year_range
from tensorflow_caney.utils.system import seed_everything


CHUNKS = {'band': 'auto', 'x': 'auto', 'y': 'auto'}

__status__ = "development"


# ---------------------------------------------------------------------------
# script preprocess.py
# ---------------------------------------------------------------------------
def run(args, conf) -> None:

    logging.info('Starting main preprocessing method.')

    # Define output directories
    metadata_output_dir = os.path.join(conf.data_dir, 'metadata')
    intersection_output_dir = os.path.join(metadata_output_dir, 'intersection')
    images_output_dir = os.path.join(conf.data_dir, 'images')
    labels_output_dir = os.path.join(conf.data_dir, 'labels')

    # Create output directories
    for out_dir in [
            images_output_dir, labels_output_dir, intersection_output_dir]:
        os.makedirs(out_dir, exist_ok=True)

    # Generate WorldView vs. ICESAT-2 footprint (geopackages with matches)
    if conf.footprint:

        # get ATL08 dataframe points
        atl08_gdf = get_atl08_gdf(
            conf.atl08_dir,
            conf.atl08_start_year,
            conf.atl08_end_year,
        )
        logging.info(f'Load ATL08 GDF files, {atl08_gdf.shape[0]} rows.')

        # some EDA information to catch non-correct labels
        logging.info(
            f'{conf.label_attribute} - ' +
            f'min: {atl08_gdf[conf.label_attribute].min()}, ' +
            f'max: {atl08_gdf[conf.label_attribute].max()}, ' +
            f'mean: {atl08_gdf[conf.label_attribute].mean()}'
        )

        # filter labels to catch correct heights, not tree is higher than 250m
        atl08_gdf = atl08_gdf.loc[atl08_gdf[conf.label_attribute] < 250]

        logging.info(
            f'{conf.label_attribute} - ' +
            f'min: {atl08_gdf[conf.label_attribute].min()}, ' +
            f'max: {atl08_gdf[conf.label_attribute].max()}, ' +
            f'mean: {atl08_gdf[conf.label_attribute].mean()}'
        )
        logging.info(f'ATL08 without no-data, {atl08_gdf.shape[0]} rows.')

        # Get WorldView filenames
        wv_evhr_filenames = get_filenames(conf.wv_data_regex)
        logging.info(f'WorldView, {len(wv_evhr_filenames)} files.')

        # Filter list by year range
        wv_evhr_filenames = filter_evhr_year_range(
            wv_evhr_filenames, conf.atl08_start_year, conf.atl08_end_year)
        logging.info(f'WorldView, {len(wv_evhr_filenames)} files.')

        # Intersection of the two GDBs, output geopackages files
        filter_polygon_in_raster_parallel(
            wv_evhr_filenames, atl08_gdf, intersection_output_dir
        )

        # Gather intersection filenames
        intersection_files = glob(
            os.path.join(intersection_output_dir, '*', '*.gpkg'))
        logging.info(
            f'{len(intersection_files)} WorldView matches saved ' +
            f'in {intersection_output_dir}.'
        )
    else:
        logging.info(
            'Skipping WV-ATL08 intersection since it has been done.')

    # iterate over the previously saved intersection files
    # extract tiles fully in parallel with multi-core parallelization
    if conf.tiles:

        # list dataset filenames from disk
        intersection_filenames = sorted(glob(
            os.path.join(intersection_output_dir, '*', '*.gpkg')))
        assert len(intersection_filenames) > 0, \
            f"No gpkg files found under {intersection_output_dir}."

        extract_tiles_parallel(
            intersection_filenames[:40], conf.tile_size,
            conf.input_bands, conf.output_bands,
            images_output_dir, labels_output_dir, conf.label_attribute,
            conf.mask_dir, conf.cloudmask_dir, conf.mask_preprocessing
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
