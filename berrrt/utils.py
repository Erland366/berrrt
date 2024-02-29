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


# Taken from https://github.com/unslothai/unsloth/blob/main/unsloth/save.py#L854
def upload_to_huggingface(
    model: str,
    save_directory: str | os.PathLike[str],
    token: str,
    method: str,
    extra: str = "",
    file_location: str | os.PathLike[str] | None = None,
    old_username: str | None = None,
    private: bool | None = None,
):
    username = ""
    save_directory = save_directory.lstrip("./")
    if "/" not in save_directory:
        from huggingface_hub import whoami

        try:
            username = whoami(token=token)["name"]
            if isinstance(old_username, str) and username != old_username:
                username = old_username
            save_directory = f"{username}/{save_directory}"
        except Exception as _:
            raise RuntimeError(f"{save_directory} is not a valid path on Hugging Face.")
    else:
        username = save_directory.split("/")[0]

    from huggingface_hub import create_repo

    try:
        create_repo(
            repo_id=save_directory,
            repo_type="model",
            token=token,
            private=private,
            exist_ok=False,
        )

        from huggingface_hub import ModelCard

        content = ModelCard.format(
            username=username,
            base_model=model.config._name_or_path,
            model_type=model.config.model_type,
            method="",
            extra=extra,
        )
        card = ModelCard(content)
        card.push_to_hub(save_directory, token=token)
    except Exception as _:
        pass

    if file_location is not None:
        from huggingface_hub import HfApi

        hf_api = HfApi(token=token)

        if "/" in file_location:
            uploaded_location = file_location[file_location.rfind("/") + 1 :]
        else:
            uploaded_location = file_location

        hf_api.upload_file(
            path_or_fileobj=file_location,
            path_in_repo=uploaded_location,
            repo_id=save_directory,
            repo_type="model",
            commit_message="(Trained with Unsloth)",
        )

        # We also upload a config.json file
        import json

        with open("_temporary_config.json", "w") as file:
            json.dump({"model_type": model.config.model_type}, file, indent=4)
        pass
        hf_api.upload_file(
            path_or_fileobj="_temporary_config.json",
            path_in_repo="config.json",
            repo_id=save_directory,
            repo_type="model",
            commit_message="",
        )
        os.remove("_temporary_config.json")


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
