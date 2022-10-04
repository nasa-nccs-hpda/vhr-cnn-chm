from typing import List
from dataclasses import dataclass, field
from tensorflow_caney.config.cnn_config import Config


@dataclass
class CHMConfig(Config):

    atl08_dir: str = 'input'
    atl08_start_year: int = 2018
    atl08_end_year: int = 2022
    general_crs: str = 'EPSG:32628'
    wv_data_regex: List[str] = field(default_factory=lambda: [])
    mask_dir: str = 'trees'
    cloudmask_dir: str = 'clouds'
    mask_preprocessing: bool = True
    mask_postprocessing: bool = False
    crop: bool = True
    footprint: bool = True
    tiles: bool = True
    label_attribute: str = 'h_can_20m'
