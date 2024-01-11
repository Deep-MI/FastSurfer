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

import glob

# IMPORTS
import os
from typing import Collection, Iterable, MutableSequence, Optional, Union

import requests
import torch
import yacs.config

from FastSurferCNN.utils import logging

Scheduler = "torch.optim.lr_scheduler"
LOGGER = logging.getLogger(__name__)

# Defaults
URL = "https://b2share.fz-juelich.de/api/files/a423a576-220d-47b0-9e0c-b5b32d45fc59"
FASTSURFER_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
VINN_AXI = os.path.join(FASTSURFER_ROOT, "checkpoints/aparc_vinn_axial_v2.0.0.pkl")
VINN_COR = os.path.join(FASTSURFER_ROOT, "checkpoints/aparc_vinn_coronal_v2.0.0.pkl")
VINN_SAG = os.path.join(FASTSURFER_ROOT, "checkpoints/aparc_vinn_sagittal_v2.0.0.pkl")


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
    log_dir: str, resume_experiment: Union[str, int, None] = None
) -> Optional[MutableSequence[str]]:
    """
    Find the paths to checkpoints from the experiment directory.

    Parameters
    ----------
    log_dir : str
        Experiment directory.
    resume_experiment : Union[str, int, None]
        Sub-experiment to search in for a model (Default value = None).

    Returns
    -------
    prior_model_paths : Optional[MutableSequence[str]]
        None, if no models are found, or a list of filenames for checkpoints.
    """
    if resume_experiment == "Default" or resume_experiment is None:
        return None
    checkpoint_path = os.path.join(log_dir, "checkpoints", str(resume_experiment))
    prior_model_paths = sorted(
        glob.glob(os.path.join(checkpoint_path, "Epoch_*")), key=os.path.getmtime
    )
    if len(prior_model_paths) == 0:
        return None
    return prior_model_paths


def load_from_checkpoint(
    checkpoint_path: str,
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
    checkpoint_path : str
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
    checkpoint_dir: str,
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
    checkpoint_dir : str
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

    torch.save(checkpoint, checkpoint_dir + "/" + save_name)

    if best:
        remove_ckpt(checkpoint_dir + "/Best_training_state.pkl")
        torch.save(checkpoint, checkpoint_dir + "/Best_training_state.pkl")


def remove_ckpt(ckpt: str):
    """
    Remove the checkpoint.

    Parameters
    ----------
    ckpt : str
        Path and filename to the checkpoint.
    """
    try:
        os.remove(ckpt)
    except FileNotFoundError:
        pass


def download_checkpoint(
    download_url: str, checkpoint_name: str, checkpoint_path: str
) -> None:
    """
    Download a checkpoint file.

    Raises an HTTPError if the file is not found or the server is not reachable.

    Parameters
    ----------
    download_url : str
        URL of checkpoint hosting site.
    checkpoint_name : str
        Name of checkpoint.
    checkpoint_path : str
        Path of the file in which the checkpoint will be saved.
    """
    try:
        response = requests.get(download_url + "/" + checkpoint_name, verify=True)
        # Raise error if file does not exist:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        LOGGER.info("Response code: {}".format(e.response.status_code))
        response = requests.get(download_url + "/" + checkpoint_name, verify=False)
        response.raise_for_status()

    with open(checkpoint_path, "wb") as f:
        f.write(response.content)


def check_and_download_ckpts(checkpoint_path: str, url: str) -> None:
    """
    Check and download a checkpoint file, if it does not exist.

    Parameters
    ----------
    checkpoint_path : str
        Path of the file in which the checkpoint will be saved.
    url : str
        URL of checkpoint hosting site.
    """
    # Download checkpoint file from url if it does not exist
    if not os.path.exists(checkpoint_path):
        ckptdir, ckptname = os.path.split(checkpoint_path)
        if not os.path.exists(ckptdir) and ckptdir:
            os.makedirs(ckptdir)
        download_checkpoint(url, ckptname, checkpoint_path)


def get_checkpoints(axi: str, cor: str, sag: str, url: str = URL) -> None:
    """
    Check and download checkpoint files if not exist.

    Parameters
    ----------
    axi : str
        Axial path of the file in which the checkpoint will be saved.
    cor : str
        Coronal path of the file in which the checkpoint will be saved.
    sag : str
        Sagittal path of the file in which the checkpoint will be saved.
    url : str
        URL of checkpoint hosting site (Default value = URL).
    """
    check_and_download_ckpts(axi, url)
    check_and_download_ckpts(cor, url)
    check_and_download_ckpts(sag, url)
