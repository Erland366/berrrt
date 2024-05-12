import os

import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

import wandb
from berrrt.dataset import BERRRTDataset
from berrrt.modules.base import ModulesFactory
from berrrt.torch_utils import get_default_device, set_seed
from berrrt.utils import (
    compute_metrics,
    create_run_name,
    print_headers,
    setup_wandb,
    compute_metrics_multi,
)


@hydra.main(version_base=None, config_path="berrrt/conf", config_name="config")
def run(cfg: DictConfig):
    run_name = create_run_name(
        cfg.run_name.model_type,
        cfg.modules.additional_prefix,
        cfg.dataset.name,
        cfg.train.num_train_epochs,
        cfg.train.learning_rate,
        cfg.train.optim,
        cfg.mode.lower() == "sanity_check",
    )
    del cfg.modules.additional_prefix

    if cfg.mode == "debug":
        print_headers(cfg)
        print("Debug mode, here's your configuration")
        print()
        print(om.to_yaml(cfg))
        print(f"{run_name = }")
        return
    elif cfg.mode == "sanity_check":
        cfg.train.num_train_epochs = 1
        cfg.train.report_to = None
    set_seed(cfg.utils.random_seed)

    if cfg.logging.name is None:
        cfg.logging.name = run_name

    if cfg.train.report_to == "wandb":
        setup_wandb(cfg)
    run_name = os.path.join(cfg.model_output_path, run_name)
    cfg.run_name.run_name = run_name

    if not os.path.exists(run_name):
        os.makedirs(run_name)
    print_headers(cfg)

    main(cfg)


def main(cfg: DictConfig):
    model = ModulesFactory(cfg.modules_name).create_model(**cfg.modules)

    device = get_default_device()
    model = model.to(device)

    training_args = TrainingArguments(
        output_dir=f"./{cfg.run_name.run_name}",
        run_name=cfg.run_name.run_name,
        **cfg.train,
    )

    dataset = BERRRTDataset(cfg)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset.train_encoded,
        eval_dataset=dataset.eval_encoded,
        compute_metrics=compute_metrics
        if cfg.dataset.num_classes == 2
        else compute_metrics_multi,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()

    wandb.config.update({"hydra": om.to_container(cfg, resolve=True)})

    if not cfg.mode == "full":
        trainer.evaluate(eval_dataset=dataset.eval_encoded)
        test_results = trainer.predict(test_dataset=dataset.test_encoded)
        print(test_results.metrics)

    wandb.finish()


if __name__ == "__main__":
    run()
