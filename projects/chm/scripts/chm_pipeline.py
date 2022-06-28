# -*- coding: utf-8 -*-

import sys
import time
import omegaconf
import argparse
import logging
from multiprocessing import cpu_count
# from vhr_cnn_chm.model.cnn_regression_pipeline import CNNRegressionPipeline
# from vhr_cnn_chm.model.cnn_config import Config

__status__ = "development"


# -----------------------------------------------------------------------------
# main
#
# python cnn_regression_pipeline.py -c config.yaml -s train -d config.csv
# -----------------------------------------------------------------------------
def main():

    # Process command-line args.
    desc = 'Use this application to map LCLUC in Senegal using WV data.'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('-c',
                        '--config-file',
                        type=str,
                        required=True,
                        dest='config_file',
                        help='Path to the configuration file')

    parser.add_argument(
                        '-s',
                        '--step',
                        type=str,
                        nargs='*',
                        required=True,
                        dest='pipeline_step',
                        help='Pipeline step to perform',
                        default=['preprocess', 'train', 'predict'],
                        choices=['preprocess', 'train', 'predict'])

    parser.add_argument(
                        '-p',
                        '--num-processes',
                        type=int,
                        required=False,
                        dest='n_processes',
                        help='Number of processes',
                        default=cpu_count())

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

    """
    # Configuration file intialization
    schema = omegaconf.OmegaConf.structured(Config)
    conf = omegaconf.OmegaConf.load(args.config_file)
    try:
        conf = omegaconf.OmegaConf.merge(schema, conf)
    except BaseException as err:
        sys.exit(f"ERROR: {err}")

    # Regression CHM pipeline
    cnn_pipeline = CNNRegressionPipeline(conf, args.n_processes)
    timer = time.time()

    # Regression CHM pipeline steps
    if "preprocess" in args.pipeline_step:
        cnn_pipeline.preprocess()
    if "train" in args.pipeline_step:
        cnn_pipeline.train()
    if "predict" in args.pipeline_step:
        cnn_pipeline.predict()

    logging.info(f'Took {(time.time()-timer)/60.0:.2f} min.')
    """

# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
