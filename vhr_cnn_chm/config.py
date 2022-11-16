from typing import List
from dataclasses import dataclass, field
from tensorflow_caney.config.cnn_config import Config


@dataclass
class CHMConfig(Config):

    # Directory where atl08 training data is stored
    atl08_dir: str = 'input'

    # Start year for atl08 training data
    atl08_start_year: int = 2018

    # End year for atl08 training data
    atl08_end_year: int = 2022

    # Label attribute to choose from atl08 dataset
    label_attribute: str = 'h_can_20m'

    # General CRS for the study area, variable between UTM zones
    general_crs: str = 'EPSG:32628'

    # Regex to find WorldView data
    wv_data_regex: List[str] = field(default_factory=lambda: [])

    # Directory where intermediary mask lives in
    mask_dir: str = 'trees'

    # Directory where cloud mask lives in
    cloudmask_dir: str = 'clouds'

    # Directory where landcover mask lives in
    landcover_dir: str = 'landcover'

    # Perform preprocessing with mask
    mask_preprocessing: bool = True

    # Perform postprocessing with mask
    mask_postprocessing: bool = False

    # Define if we need to create footprint of WV and ICESAT-2 matches
    footprint: bool = True

    # Define if we need to generate tiles during preprocessing
    tiles: bool = True
