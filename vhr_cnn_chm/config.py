from typing import List, Optional
from dataclasses import dataclass, field
from tensorflow_caney.config.cnn_config import Config


@dataclass
class CHMConfig(Config):

    # Directory where atl08 training data is stored
    atl08_dir: Optional[str] = 'input'

    # Start year for atl08 training data
    atl08_start_year: Optional[int] = 2018

    # End year for atl08 training data
    atl08_end_year: Optional[int] = 2022

    # Label attribute to choose from atl08 dataset
    label_attribute: Optional[str] = 'h_can_20m'

    # General CRS for the study area, variable between UTM zones
    general_crs: Optional[str] = 'EPSG:32628'

    # Regex to find WorldView data
    wv_data_regex: Optional[List[str]] = field(default_factory=lambda: [])

    # Directory where intermediary mask lives in
    mask_dir: Optional[str] = 'trees'

    # Directory where cloud mask lives in
    cloudmask_dir: Optional[str] = 'clouds'

    # Directory where landcover mask lives in
    landcover_dir: Optional[str] = 'landcover'

    # Perform preprocessing with mask
    mask_preprocessing: Optional[bool] = True

    # Perform postprocessing with mask
    mask_postprocessing: Optional[bool] = False

    # Define if we need to create footprint of WV and ICESAT-2 matches
    footprint: Optional[bool] = True

    # Define if we need to generate tiles during preprocessing
    tiles: Optional[bool] = True
