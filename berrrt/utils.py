import logging
import platform
from pprint import pprint

import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers.trainer_utils import EvalPrediction
from transformers import TrainerCallback, TrainerState
from transformers.integrations import WandbCallback
import numpy as np


import wandb

log = logging.getLogger(__name__)


class AccuracyLogitsCallback(TrainerCallback):
    def on_train_end(self, args, state, control, kwargs):
        pass


def setup_wandb(cfg):
    run = wandb.init(
        project=cfg.logging.project,
        name=cfg.logging.name,
        entity=cfg.logging.entity,
        tags=list(cfg.logging.tags),
    )
    return run


def list_to_indexed_dict(lst, prefix):
    """
    Convert a list to a dictionary with keys as a combination of a prefix and the index.

    Parameters:
    lst (list): The list of values to be converted into a dictionary.
    prefix (str): The prefix to be used for the keys.

    Returns:
    dict: A dictionary with keys formed by the prefix and index, and values from the list.
    """
    return {f"{prefix}-{i}": value for i, value in enumerate(lst)}


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
    if cfg.mode == "debug":
        print(
            f"  ____   _____  ____   ____   ____  _____ \tDEBUG MODE, NOT GOING TO TRAIN! Experiment Type: {cfg.modules_name}"
        )
    elif cfg.mode == "sanity_check":
        print(
            f"  ____   _____  ____   ____   ____  _____ \tSANITY CHECK MODE, MAKE SURE RESULT IS OVERFIT! Experiment Type: {cfg.modules_name}"
        )
    elif cfg.mode == "full":
        print(
            f"  ____   _____  ____   ____   ____  _____ \tSTART TRAINING! Experiment Type: {cfg.modules_name}"
        )
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")
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
    if isinstance(pred.predictions, tuple):
        logits = pred.predictions[0]
    else:
        logits = pred.predictions

    # Convert logits and labels if they are tensors
    if isinstance(logits, torch.Tensor):
        logits = logits.cpu().numpy()  # Convert to NumPy array if it's a Tensor
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    final_logits = logits
    preds = logits.argmax(-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)

    # Convert metrics to lists if they are numpy arrays or tensors
    f1 = f1.tolist() if isinstance(f1, np.ndarray) else f1
    precision = precision.tolist() if isinstance(precision, np.ndarray) else precision
    recall = recall.tolist() if isinstance(recall, np.ndarray) else recall

    results = {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }

    if isinstance(pred.predictions, tuple):
        # Log the last sample from final_logits
        last_final_logits = final_logits[-1]
        last_final_logits = last_final_logits.tolist() if hasattr(last_final_logits, 'tolist') else last_final_logits
        results["final_logits_last_sample"] = last_final_logits

        all_logits = pred.predictions[1]  # List of [batch_size, num_classes]
        all_accs = [accuracy_score(labels, p.argmax(-1)) for p in all_logits]

        # Determine number of classes from the first element of all_logits if available
        num_classes = all_logits[0].shape[1] if all_logits else 0
        
        # Prepare to log all_logits using wandb.Table
        logits_table = wandb.Table(columns=[f"Class {j+1}" for j in range(num_classes)])

        for layer_index, layer_logits in enumerate(all_logits):
            # Get the last sample for this layer
            last_sample = layer_logits[-1]
            last_sample = last_sample.tolist() if hasattr(last_sample, 'tolist') else last_sample
            logits_table.add_data(*last_sample)  # Add this as a new row in the table

        results.update({
            "all_accs": all_accs,
            "all_logits_table": logits_table
        })

    return results

def compute_metrics_multi(pred: EvalPrediction) -> dict:
    labels = pred.label_ids
    if isinstance(pred.predictions, tuple):
        logits = pred.predictions[0]
    else:
        logits = pred.predictions

    # Convert logits and labels if they are tensors
    if isinstance(logits, torch.Tensor):
        logits = logits.cpu().numpy()  # Convert to NumPy array if it's a Tensor
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    final_logits = logits
    preds = logits.argmax(-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average=None
    )
    acc = accuracy_score(labels, preds)

    # Convert metrics to lists if they are numpy arrays or tensors
    f1 = f1.tolist() if isinstance(f1, np.ndarray) else f1
    precision = precision.tolist() if isinstance(precision, np.ndarray) else precision
    recall = recall.tolist() if isinstance(recall, np.ndarray) else recall

    results = {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }

    if isinstance(pred.predictions, tuple):
        # Log the last sample from final_logits
        last_final_logits = final_logits[-1]
        last_final_logits = last_final_logits.tolist() if hasattr(last_final_logits, 'tolist') else last_final_logits
        results["final_logits_last_sample"] = last_final_logits

        all_logits = pred.predictions[1]  # List of [batch_size, num_classes]
        all_accs = [accuracy_score(labels, p.argmax(-1)) for p in all_logits]

        # Determine number of classes from the first element of all_logits if available
        num_classes = all_logits[0].shape[1] if all_logits else 0
        
        # Prepare to log all_logits using wandb.Table
        logits_table = wandb.Table(columns=[f"Class {j+1}" for j in range(num_classes)])

        for layer_index, layer_logits in enumerate(all_logits):
            # Get the last sample for this layer
            last_sample = layer_logits[-1]
            last_sample = last_sample.tolist() if hasattr(last_sample, 'tolist') else last_sample
            logits_table.add_data(*last_sample)  # Add this as a new row in the table

        results.update({
            "all_accs": all_accs,
            "all_logits_table": logits_table
        })

    return results


def create_run_name(
    model_type,
    additional_prefix: str,
    task,
    num_epochs,
    learning_rate,
    optimizer_name,
    sanity_check: bool = False,
    add_id: bool = True,
) -> str:
    lr_str = f"{learning_rate:.0e}".replace("-", "m")
    run_name = f"{model_type}-{additional_prefix}-{task}-{num_epochs}epochs-LR{lr_str}-{optimizer_name}"
    if sanity_check:
        run_name += "-sanity"
    if add_id:
        run_name += f"-{wandb.util.generate_id()}"
    return run_name
