import logging
import os
from enum import Enum

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import wandb

log = logging.getLogger(__name__)


class StrEnum(str, Enum):
    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"'{str(self)}'"


def setup_wandb():
    run = wandb.init()
    return run


def compute_metrics(eval_pred):
    import numpy as np

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f_score, _ = precision_recall_fscore_support(labels, predictions)

    return {
        "accuracy": accuracy,
        "f1": f_score.tolist(),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
    }


def create_run_name(model_type, task, num_epochs, learning_rate, optimizer_name):
    lr_str = f"{learning_rate:.0e}".replace("-", "m")
    return f"{model_type}-{task}-{num_epochs}epochs-LR{lr_str}-{optimizer_name}"
