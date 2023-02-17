import sys
import time
import omegaconf
import argparse
import logging

from vhr_cnn_chm.config import CHMConfig as Config
from .preprocess import run as run_preprocess
from .train import run as run_train
from .predict import run as run_predict
# from vhr_cnn_chm.model.cnn_regression_pipeline \
#   import CNNRegressionPipeline

__status__ = "development"


# -----------------------------------------------------------------------------
# main
#
# python chm_pipeline.py -c config.yaml -s preprocess train
# -----------------------------------------------------------------------------
def main():

    # Process command-line args.
    parser = argparse.ArgumentParser(description=desc)
	desc = 'Use this application to map LCLUC in Senegal using WV data.'

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

    timer = time.time()

    # Regression CHM pipeline steps
    if "preprocess" in args.pipeline_step:
        run_preprocess(args, conf)
    if "train" in args.pipeline_step:
        run_train(args, conf)
    if "predict" in args.pipeline_step:
        run_predict(args, conf)

    logging.info(f'Took {(time.time()-timer)/60.0:.2f} min.')


# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
