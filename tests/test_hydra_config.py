from hydra import compose, initialize
from omegaconf import OmegaConf as om


def test_hydra_config():
    with initialize(
        version_base=None,
        config_path="../berrrt/conf",
    ):
        cfg = compose(
            config_name="config",
            overrides=["modules=bert", "modules_name=bert", "dataset=emotion"],
        )
        print()
        print(om.to_yaml(cfg))

        print(cfg.modules.additional_prefix)
        print(cfg.logging.tags)
        print(cfg.logging.tags[0])
        print(list(cfg.logging.tags))
        print(cfg.dataset.sample_size)
        print(cfg.modules.num_classes)

        assert cfg.utils.random_seed == 43
