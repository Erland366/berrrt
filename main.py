import os

import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from transformers import (
    Trainer,
    TrainingArguments,
)

from berrrt.dataset import BERRRTDataset
from berrrt.torch_utils import get_default_device, set_seed
from berrrt.utils import compute_metrics, create_run_name


@hydra.main(version_base=None, config_path="berrrt/conf", config_name="config")
def run(cfg: DictConfig):
    if cfg.debug:
        print("Debug mode, here's your configuration")
        print()
        print(om.to_yaml(cfg))
        return
    set_seed(cfg.utils.random_seed)

    os.environ["WANDB_PROJECT"] = cfg.project_name
    run_name = create_run_name(
        cfg.run_name.model_type,
        cfg.dataset.name,
        cfg.train.num_train_epochs,
        cfg.train.learning_rate,
        cfg.train.optim,
    )

    if cfg.dataset.sample:
        run_name += "_sample"
    cfg.run_name.run_name = run_name
    main(cfg)


def main(cfg: DictConfig):
    if cfg.run_name.model_type == "berrrt":
        from berrrt.modules.berrrt import BERRRTModel

        model = BERRRTModel(**cfg.modules)
    elif cfg.run_name.model_type == "bert":
        from berrrt.modules.bert import BERTModel

        model = BERTModel(cfg.modules.pretrained_model_name_or_path)
    elif cfg.run_name.model_type == "berrrt_gate":
        from berrrt.modules.berrrt_gate import BERRRTGateModel

        model = BERRRTGateModel(**cfg.modules)
    else:
        raise ValueError(f"Unknown model type: {cfg.run_name.model_type}")
    device = get_default_device()
    model.to(device)

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
        compute_metrics=compute_metrics,
    )

    trainer.train()

    if not cfg.dataset.sample:
        trainer.evaluate(eval_dataset=dataset.eval_encoded)
        test_results = trainer.predict(test_dataset=dataset.test_encoded)
        print(test_results.metrics)


if __name__ == "__main__":
    run()
