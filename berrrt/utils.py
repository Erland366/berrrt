import logging
import platform
from pprint import pprint

import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers.trainer_utils import EvalPrediction

import wandb

log = logging.getLogger(__name__)


def setup_wandb(cfg):
    run = wandb.init(
        project=cfg.logging.project, name=cfg.logging.name, entity=cfg.logging.entity
    )
    return run


# Copied from https://github.com/unslothai/unsloth/blob/b4fe3cd67d171d899e3a90b0d9157940b6aaba3c/unsloth/models/llama.py#L917
def print_headers(cfg):
    HAS_FLASH_ATTENTION = False
    try:
        from flash_attn.flash_attn_interface import flash_attn_cuda  # noqa

        HAS_FLASH_ATTENTION = True
    except Exception as _:
        pass
    SUPPORTS_BFLOAT16 = torch.cuda.is_bf16_supported()
    gpu_stats = torch.cuda.get_device_properties(0)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print("=========================================")
    print(
        f"  ____   _____  ____   ____   ____  _____ \tSTART TRAINING! Experiment Type: {cfg.modules_name}"
    )
    print(
        f" | __ ) | ____||  _ \ |  _ \ |  _ \|_   _|\tGPU: {gpu_stats.name}. Max memory: {max_memory} GB. Platform: {platform.system()}"
    )
    print(
        f" |  _ \ |  _|  | |_) || |_) || |_) | | |\tPytorch: {torch.__version__}, CUDA: {gpu_stats.major}.{gpu_stats.minor}. CUDA Toolkit: {torch.version.cuda}"
    )
    print(
        f" | |_) || |___ |  _ < |  _ < |  _ <  | |\tBFLOAT16: {str(SUPPORTS_BFLOAT16).upper()}. FA: {str(HAS_FLASH_ATTENTION).upper()}"
    )
    print(" |____/ |_____||_| \_\|_| \_\|_| \_\ |_|")
    print("=========================================")
    print("Experiment Config:", end=" ")
    pprint(cfg.modules)


def compute_metrics(pred: EvalPrediction) -> dict:
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def create_run_name(model_type, task, num_epochs, learning_rate, optimizer_name) -> str:
    lr_str = f"{learning_rate:.0e}".replace("-", "m")
    return f"{model_type}-{task}-{num_epochs}epochs-LR{lr_str}-{optimizer_name}"
