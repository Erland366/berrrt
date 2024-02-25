import hydra
import torch
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from transformers import (
    BertTokenizerFast,
    Trainer,
    TrainingArguments,
)

from berrrt.modules import BERRRTModel
from berrrt.utils import compute_metrics, create_run_name, set_seed


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    if cfg.debug:
        print("Debug mode, here's your configuration")
        print(OmegaConf.to_yaml(cfg))
        return
    set_seed(cfg.utils.random_seed)
    print("Hello world")


def main_yay():
    set_seed(42)

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    dataset = load_dataset("glue", "mrpc")
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]
    test_dataset = dataset["test"]

    def tokenize_function(examples):
        return tokenizer(
            examples["sentence1"],
            examples["sentence2"],
            truncation=True,
            padding="max_length",
            max_length=128,
        )

    train_encoded = train_dataset.map(tokenize_function, batched=True)
    eval_encoded = eval_dataset.map(tokenize_function, batched=True)
    test_encoded = test_dataset.map(tokenize_function, batched=True)

    train_encoded.set_format(
        type="torch", columns=["input_ids", "attention_mask", "label"]
    )
    eval_encoded.set_format(
        type="torch", columns=["input_ids", "attention_mask", "label"]
    )
    test_encoded.set_format(
        type="torch", columns=["input_ids", "attention_mask", "label"]
    )
    model = BERRRTModel("bert-base-uncased", 0, 11)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model_name = "BERRRT"
    dataset_name = "MRPC"
    epochs = 5
    learning_rate = 5e-5
    optimizer_name = "adamw_torch_fused"

    run_name = create_run_name(
        model_name, dataset_name, epochs, learning_rate, optimizer_name
    )

    training_args = TrainingArguments(
        output_dir=f"./{run_name}",
        run_name=run_name,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        optim=optimizer_name,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="wandb",
        project="berrrt",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_encoded,
        eval_dataset=eval_encoded,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.evaluate(eval_dataset=eval_encoded)
    test_results = trainer.predict(test_dataset=test_encoded)
    print(test_results.metrics)


if __name__ == "__main__":
    main()
