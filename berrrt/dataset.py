from datasets import load_dataset
from omegaconf import DictConfig
from transformers import BertTokenizerFast


class BERRRTDataset(object):
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.tokenizer = BertTokenizerFast.from_pretrained(
            cfg.dataset.pretrained_model_name_or_path
        )
        self.dataset = load_dataset(cfg.dataset.path, cfg.dataset.name)
        self.columns = cfg.dataset.get("columns", ["text"])
        self.train_encoded = None
        self.eval_encoded = None
        self.test_encoded = None

        if cfg.mode == "sanity_check":
            from collections import Counter

            import numpy as np

            self.train_dataset = (
                self.dataset["train"]
                .shuffle(seed=cfg.utils.random_seed)
                .select(range(cfg.dataset.sample_size))
            )
            self.eval_dataset = self.train_dataset
            # Get the label values as a NumPy array
            label_values = np.array(self.train_dataset["label"])

            # Count the occurrences of each label value
            label_counts = Counter(label_values)

            # Print the counts for each class
            for label, count in label_counts.items():
                print(f"Class {label}: {count} samples")
        else:
            self.train_dataset = self.dataset["train"]
            self.eval_dataset = self.dataset.get("validation", None)
        self.test_dataset = self.dataset.get("test", None)
        self.prepare_datasets()

    def tokenize_function(self, examples):

        return self.tokenizer(
            *[examples[column] for column in self.columns],
            truncation=self.cfg.dataset.tokenizer.truncation,
            padding=self.cfg.dataset.tokenizer.padding,
            max_length=self.cfg.dataset.tokenizer.max_length,
        )

    def prepare_datasets(self):
        self.train_encoded = self.train_dataset.map(
            self.tokenize_function, batched=self.cfg.dataset.batched
        )
        self.train_encoded = self.train_encoded.remove_columns(
            self.columns
        )

        if self.eval_dataset is not None:
            self.eval_encoded = self.eval_dataset.map(
                self.tokenize_function, batched=self.cfg.dataset.batched
            )
            self.eval_encoded = self.eval_encoded.remove_columns(
                self.columns
            )
            self.set_format(self.eval_encoded)

        if self.test_dataset is not None:
            self.test_encoded = self.test_dataset.map(
                self.tokenize_function, batched=self.cfg.dataset.batched
            )
            self.test_encoded = self.test_encoded.remove_columns(
                self.columns
            )
            self.set_format(self.test_encoded)

        self.set_format(self.train_encoded)

    def set_format(self, dataset):
        dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "label"]
        )
