# -*- coding: utf-8 -*-

import os
import sys
import argparse
import logging

sys.path.append('/adapt/nobackup/people/jacaraba/development/vhr-cnn-chm')

from vhr_cnn_chm.model.cnn_regression_pipeline import CNNRegressionPipeline

__status__ = "development"

# -----------------------------------------------------------------------------
# main
#
# python cnn_pipeline.py -c config.yaml -s train -d config.csv
# -----------------------------------------------------------------------------
def main():

    # Process command-line args.
    #desc = 'Use this application to map LCLUC in Senegal using WV data.'
    #parser = argparse.ArgumentParser(description=desc)

    #parser.add_argument('-c',
    #                    '--config-file',
    #                    type=str,
    #                    required=True,
    #                    dest='config_file',
    #                    help='Path to the configuration file')

    #parser.add_argument('-d',
    #                    '--data-csv',
    #                    type=str,
    #                    required=True,
    #                    dest='data_csv',
    #                    help='Path to the data CSV configuration file')

    #parser.add_argument(
    #                    '-s',
    #                    '--step',
    #                    type=str,
    #                    nargs='*',
    #                    required=True,
    #                    dest='pipeline_step',
    #                    help='Pipeline step to perform',
    #                    default=['preprocess', 'train', 'predict'],
    #                    choices=['preprocess', 'train', 'predict'])

    #args = parser.parse_args()

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

    # Regression CHM pipeline
    cnn_pipeline = CNNRegressionPipeline()

    # Read ATL08 points
    atl08_gdf = cnn_pipeline.get_atl08_gdf(
        '/adapt/nobackup/people/pmontesa/userfs02/data/icesat2/atl08.005/senegal',
        2018, 2022
    )

    # Read WorldView footprints database
    nga_senegal_gdf = cnn_pipeline.get_gdb_gdf(
        '/adapt/nobackup/people/zwwillia/Senegal_Cloud/NGA_Senegal')
    
    # Filter GDF by year range
    nga_senegal_gdf = cnn_pipeline.filter_gdf_by_list(
        nga_senegal_gdf, 'acq_year', list(range(2018, 2022)))
    
    # Filter GDF by prod code (e.g. M1BS)
    nga_senegal_gdf = cnn_pipeline.filter_gdf_by_list(
        nga_senegal_gdf, 'prod_code', ['M1BS'])

    # Intersection of the two GDBs
    cnn_pipeline.filter_point_in_polygon_by_scene(
        atl08_gdf, nga_senegal_gdf,
        '/adapt/nobackup/projects/ilab/projects/Senegal/CNN_CHM/metadata')

# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())

