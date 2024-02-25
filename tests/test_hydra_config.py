from hydra import compose, initialize
from omegaconf import OmegaConf


def test_hydra_config():
    with initialize(version_base=None, config_path="../berrrt/conf"):
        cfg = compose(config_name="config")
        print()
        print(OmegaConf.to_yaml(cfg))

        assert cfg.utils.random_seed == 43
