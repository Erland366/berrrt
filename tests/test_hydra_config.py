from hydra import compose, initialize
from omegaconf import OmegaConf as om


def test_hydra_config():
    with initialize(
        version_base=None,
        config_path="../berrrt/conf",
    ):
        cfg = compose(
            config_name="config",
            overrides=["modules=berrrt"],
        )
        print()
        print(om.to_yaml(cfg))

        dict_result = om.to_container(cfg, resolve=True)

        print(f"{dict_result['modules'] = }")

        print(f"{cfg.run_name.model_type =}")

        assert cfg.utils.random_seed == 43
