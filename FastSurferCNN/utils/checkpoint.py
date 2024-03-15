# Copyright 2022 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# IMPORTS
import os
from pathlib import Path
from typing import MutableSequence, Optional, Union, Literal

import requests
import torch
import yacs.config
import yaml

from FastSurferCNN.utils import logging


Scheduler = "torch.optim.lr_scheduler"
LOGGER = logging.getLogger(__name__)

# Defaults
FASTSURFER_ROOT = Path(__file__).parents[2]
YAML_DEFAULT = FASTSURFER_ROOT / "FastSurferCNN/config/checkpoint_paths.yaml"


def get_plane_dict(filename : Path = YAML_DEFAULT) -> dict[str, Union[str, list[str]]]:
    """
    Load the plane dictionary from the yaml file.

    Parameters
    ----------
    filename : Path
        The path to the yaml file. Either absolute or relative to the FastSurfer root directory.

    Returns
    -------
    dict[str, Union[str, list[str]]]
        Dictionary of the yaml file.

    """
    if not filename.absolute():
        filename = FASTSURFER_ROOT / filename

    with (open(filename, "r")) as file:
        dictionary = yaml.load(file, Loader=yaml.FullLoader)
    return dictionary

def get_plane_default(type : Literal["URL", "CKPT", "CFG"], plane : Optional[str] = None, filename : str | Path = YAML_DEFAULT) -> str | list[str]:
    """
    Get the default value for a specific plane.

    Parameters
    ----------
    type : "URL", "CKPT", "CFG"
        Type of value.
    plane : str
        Plane to get the default value for. Can be "axial", "coronal" or "sagittal".
    filename : str | Path
        The path to the yaml file. Either absolute or relative to the FastSurfer root directory.

    Returns
    -------
    str, list[str]
        Default value for the plane.

    """
    if not isinstance(filename, Path):
        filename = Path(filename)

    type = type.upper()
    if type not in ["URL", "CKPT", "CFG"]:
        raise ValueError("Type must be URL, CKPT or CFG")
    
    if type == "URL":
        return get_plane_dict(filename)[type.upper()]
    
    if plane is None:
        raise ValueError("Plane must be specified for type CKPT or CFG")
    return get_plane_dict(filename)[type.upper()][plane.upper()]


def create_checkpoint_dir(expr_dir: Union[os.PathLike], expr_num: int):
    """
    Create the checkpoint dir if not exists.

    Parameters
    ----------
    expr_dir : Union[os.PathLike]
        Directory to create.
    expr_num : int
        Number of expr [MISSING].

    Returns
    -------
    checkpoint_dir
        Directory of the checkpoint.
    """
    checkpoint_dir = os.path.join(expr_dir, "checkpoints", str(expr_num))
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir


def get_checkpoint(ckpt_dir: str, epoch: int) -> str:
    """
    Find the standardizes checkpoint name for the checkpoint in the directory ckpt_dir for the given epoch.

    Parameters
    ----------
    ckpt_dir : str
        Checkpoint directory.
    epoch : int
        Number of the epoch.

    Returns
    -------
    checkpoint_dir
        Standardizes checkpoint name.
    """
    checkpoint_dir = os.path.join(
        ckpt_dir, "Epoch_{:05d}_training_state.pkl".format(epoch)
    )
    return checkpoint_dir


def get_checkpoint_path(
    log_dir: Path | str, resume_experiment: Union[str, int, None] = None
) -> MutableSequence[Path]:
    """
    Find the paths to checkpoints from the experiment directory.

    Parameters
    ----------
    log_dir : Path, str
        Experiment directory.
    resume_experiment : Union[str, int, None]
        Sub-experiment to search in for a model (Default value = None).

    Returns
    -------
    prior_model_paths : MutableSequence[Path]
        A list of filenames for checkpoints.
    """
    if resume_experiment == "Default" or resume_experiment is None:
        return []
    if not isinstance(log_dir, Path):
        log_dir = Path(log_dir)
    checkpoint_path = log_dir / "checkpoints" / str(resume_experiment)
    prior_model_paths = sorted(
        checkpoint_path.glob("Epoch_*"), key=lambda p: p.stat().st_mtime
    )
    return list(prior_model_paths)


def load_from_checkpoint(
    checkpoint_path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Scheduler] = None,
    fine_tune: bool = False,
    drop_classifier: bool = False,
):
    """
    Load the model from the given experiment number.

    Parameters
    ----------
    checkpoint_path : str, Path
        Path to the checkpoint.
    model : torch.nn.Module
        Network model.
    optimizer : Optional[torch.optim.Optimizer]
        Network optimizer (Default value = None).
    scheduler : Optional[Scheduler]
        Network scheduler (Default value = None).
    fine_tune : bool
        Whether to fine tune or not (Default value = False).
    drop_classifier : bool
        Whether to drop the classifier or not (Default value = False).

    Returns
    -------
    loaded_epoch : int
        Epoch number.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if drop_classifier:
        classifier_conv = ["classifier.conv.weight", "classifier.conv.bias"]
        for key in classifier_conv:
            if key in checkpoint["model_state"]:
                del checkpoint["model_state"][key]

    # if this is a multi-gpu model, get the underlying model
    mod = model.module if hasattr(model, "module") else model
    mod.load_state_dict(checkpoint["model_state"], strict=not drop_classifier)

    if not fine_tune:
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if scheduler is not None and "scheduler_state" in checkpoint.keys():
            scheduler.load_state_dict(checkpoint["scheduler_state"])

    return checkpoint["epoch"] + 1, checkpoint.get("best_metric", None)


def save_checkpoint(
    checkpoint_dir: str | Path,
    epoch: int,
    best_metric,
    num_gpus: int,
    cfg: yacs.config.CfgNode,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Scheduler] = None,
    best: bool = False,
) -> None:
    """
    Save the state of training for resume or fine-tune.

    Parameters
    ----------
    checkpoint_dir : str, Path
        Path to the checkpoint directory.
    epoch : int
        Current epoch.
    best_metric : best_metric
        Best calculated metric.
    num_gpus : int
        Number of used gpus.
    cfg : yacs.config.CfgNode
        Configuration node.
    model : torch.nn.Module
        Used network model.
    optimizer : torch.optim.Optimizer
        Used network optimizer.
    scheduler : Optional[Scheduler]
        Used network scheduler. Optional (Default value = None).
    best : bool
        Whether this was the best checkpoint so far [MISSING] (Default value = False).
    """
    save_name = f"Epoch_{epoch:05d}_training_state.pkl"
    saving_model = model.module if num_gpus > 1 else model
    checkpoint = {
        "model_state": saving_model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
        "config": cfg.dump(),
    }

    if scheduler is not None:
        checkpoint["scheduler_state"] = scheduler.state_dict()
    if not isinstance(checkpoint_dir, Path):
        checkpoint_dir = Path(checkpoint_dir)

    torch.save(checkpoint, checkpoint_dir / save_name)

    if best:
        remove_ckpt(checkpoint_dir / "Best_training_state.pkl")
        torch.save(checkpoint, checkpoint_dir / "Best_training_state.pkl")


def remove_ckpt(ckpt: str | Path):
    """
    Remove the checkpoint.

    Parameters
    ----------
    ckpt : str, Path
        Path and filename to the checkpoint.
    """
    try:
        Path(ckpt).unlink()
    except FileNotFoundError:
        pass


def download_checkpoint(
        checkpoint_name: str,
        checkpoint_path: str | Path,
        urls : list[str],
) -> None:
    """
    Download a checkpoint file.

    Raises an HTTPError if the file is not found or the server is not reachable.

    Parameters
    ----------
    checkpoint_name : str
        Name of checkpoint.
    checkpoint_path : Path, str
        Path of the file in which the checkpoint will be saved.
    urls : list[str]
        List of URLs of checkpoint hosting sites.
    """
    response = None
    for url in urls:
        try:
            LOGGER.info(f"Downloading checkpoint from {url}")
            response = requests.get(url + "/" + checkpoint_name, verify=True)
            # Raise error if file does not exist:
            response.raise_for_status()
            break
            
        except requests.exceptions.HTTPError as e:
            LOGGER.info(f"Server {url} not reachable.")
            LOGGER.warn(f"Response code: {e.response.status_code}")
        except requests.exceptions.RequestException as e:
            LOGGER.warn(f"Server {url} not reachable.")
    
    if response is None:
        raise requests.exceptions.RequestException("No server reachable.")
    else:
        response.raise_for_status() # Raise error if no server is reachable

    with open(checkpoint_path, "wb") as f:
        f.write(response.content)


def check_and_download_ckpts(checkpoint_path: Path | str, urls: list[str]) -> None:
    """
    Check and download a checkpoint file, if it does not exist.

    Parameters
    ----------
    checkpoint_path : Path, str
        Path of the file in which the checkpoint will be saved.
    url : str
        URL of checkpoint hosting site.
    """
    if not isinstance(checkpoint_path, Path):
        checkpoint_path = Path(checkpoint_path)
    # Download checkpoint file from url if it does not exist
    if not checkpoint_path.exists():
        # create dir if it does not exist
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        download_checkpoint(checkpoint_path.name, checkpoint_path, urls)

def get_checkpoints(*checkpoints: Path | str, urls : list[str]) -> None:
    """
    Check and download checkpoint files if not exist.

    Parameters
    ----------
    *checkpoints : Path, str
        Paths of the files in which the checkpoint will be saved.
    urls : Path, str
        URLs of checkpoint hosting sites.
    """
    try:
        for file in checkpoints:
            check_and_download_ckpts(file, urls)
    except requests.exceptions.HTTPError:
        LOGGER.error(f"Could not find nor download checkpoints from {urls}")
        raise

