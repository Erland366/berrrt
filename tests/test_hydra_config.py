from hydra import compose, initialize


def test_hydra_config():
    with initialize(
        version_base=None,
        config_path="../berrrt/conf",
    ):
        cfg = compose(
            config_name="config",
            overrides=["modules=bert", "modules_name=bert", "dataset=emotion"],
        )
    assert cfg.utils.random_seed == 43
