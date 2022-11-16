import sys
import pytest
from omegaconf import OmegaConf
from vhr_cnn_chm.config import CHMConfig as Config


@pytest.mark.parametrize(
    "filename, expected_experiment_name, expected_experiment_type",
    [(
        'tests/test_data/chm_test.yaml',
        'cnn-chm',
        '2m_chm'
    )]
)
def test_config_yaml(
            filename: str,
            expected_experiment_name: str,
            expected_experiment_type: str
        ):
    schema = OmegaConf.structured(Config)
    conf = OmegaConf.load(filename)

    try:
        conf = OmegaConf.merge(schema, conf)
    except BaseException as err:
        sys.exit(f"ERROR: {err}")

    assert expected_experiment_name == conf.experiment_name
    assert expected_experiment_type == conf.experiment_type
