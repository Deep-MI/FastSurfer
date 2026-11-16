# Copyright 2022 DeepMI Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
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
import hashlib
import os
import warnings
from collections.abc import Iterable, MutableSequence, Sequence
from functools import lru_cache
from pathlib import Path
from time import sleep
from typing import TYPE_CHECKING, Literal, NamedTuple, TypedDict, cast, overload
from uuid import uuid4

import requests
import torch
import yacs.config
import yaml

from FastSurferCNN.utils import Plane, logging
from FastSurferCNN.utils.parser_defaults import FASTSURFER_ROOT

if TYPE_CHECKING:
    from torch.optim import lr_scheduler as Scheduler
else:
    class Scheduler:
        ...

LOGGER = logging.getLogger(__name__)

class CheckpointFile(NamedTuple):
    """A checkpoint file and the md5 of the published file, None where there is none to check against."""

    path: Path
    md5: str | None = None


class CheckpointConfigDict(TypedDict, total=False):
    url: list[str]
    checkpoint: dict[Plane, CheckpointFile]
    config: dict[Plane, Path]


CheckpointConfigFields = Literal["checkpoint", "config", "url"]


@lru_cache
def load_checkpoint_config(filename: Path | str) -> CheckpointConfigDict:
    """
    Load the plane dictionary from the yaml file.

    Parameters
    ----------
    filename : Path, str
        Path to the yaml file. Either absolute or relative to the FastSurfer root
        directory.

    Returns
    -------
    CheckpointConfigDict
        A dictionary representing the contents of the yaml file, with the entries of checkpoint
        (``{"path": ..., "md5": ...}``) as CheckpointFile.
    """
    filename = Path(filename)
    if not filename.is_absolute():
        filename = FASTSURFER_ROOT / filename

    with open(filename) as file:
        data = yaml.load(file, Loader=yaml.FullLoader)

    required_fields = ("url", "checkpoint")
    checks = [k not in data for k in required_fields]
    if any(checks):
        missing = tuple(k for k, c in zip(required_fields, checks, strict=False) if c)
        message = f"The file {filename} is not valid, missing key(s): {missing}"
        raise OSError(message)
    if isinstance(data["url"], str):
        data["url"] = [data["url"]]
    else:
        data["url"] = list(data["url"])
    if "config" in data:
        data["config"] = {k: Path(v) for k, v in data["config"].items()}
    checkpoints = {}
    for key, entry in data["checkpoint"].items():
        if not isinstance(entry, dict) or not {"path", "md5"} <= entry.keys():
            raise OSError(f"The file {filename} is not valid, checkpoint {key} needs a 'path' and an 'md5'.")
        checkpoints[key] = CheckpointFile(Path(entry["path"]), str(entry["md5"]).lower())
    data["checkpoint"] = checkpoints
    return data


@overload
def load_checkpoint_config_defaults(
        configtype: Literal["config"],
        filename: str | Path,
) -> dict[Plane, Path]: ...


@overload
def load_checkpoint_config_defaults(
        configtype: Literal["url"],
        filename: str | Path,
) -> list[str]: ...


@overload
def load_checkpoint_config_defaults(
        configtype: Literal["checkpoint"],
        filename: str | Path,
) -> dict[str, CheckpointFile]: ...


@lru_cache
def load_checkpoint_config_defaults(
        configtype: CheckpointConfigFields,
        filename: str | Path,
) -> dict[Plane, Path] | dict[str, CheckpointFile] | list[str]:
    """
    Get the default value for a specific plane or the url.

    Parameters
    ----------
    configtype : "checkpoint", "config", "url"
        Type of value.
    filename : str, Path
        The path to the yaml file. Either absolute or relative to the FastSurfer root
        directory.

    Returns
    -------
    dict[str, CheckpointFile], dict[Plane, Path], list[str]
        The checkpoint files with their md5, the config files, or the urls.
    """
    if not isinstance(filename, Path):
        filename = Path(filename)

    configtype = cast(CheckpointConfigFields, configtype.lower())
    if configtype not in ("url", "checkpoint", "config"):
        raise ValueError("Type must be 'url', 'checkpoint' or 'config'")

    return load_checkpoint_config(filename)[configtype]


def create_checkpoint_dir(expr_dir: os.PathLike, expr_num: int):
    """
    Create the checkpoint dir if not exists.

    Parameters
    ----------
    expr_dir : Union[os.PathLike]
        Directory to create.
    expr_num : int
        Experiment number.

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
    Find the standardizes checkpoint name for the checkpoint in the directory
    ckpt_dir for the given epoch.

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
        ckpt_dir, f"Epoch_{epoch:05d}_training_state.pkl"
    )
    return checkpoint_dir


def get_config_file(module: str) -> Path:
    """
    Returns the path to the checkpoint_paths.yaml file of `module`.

    Parameters
    ==========
    module : str
        The FastSurfer module name.

    Returns
    =======
    Path
        The path to the checkpoint_paths.yaml file of `module`.
    """
    return FASTSURFER_ROOT / module / "config/checkpoint_paths.yaml"


def get_checkpoint_path(
        log_dir: Path | str, resume_experiment: str | int | None = None
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
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Scheduler | None = None,
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
    # WARNING: weights_only=False can cause unsafe code execution, but here the
    # checkpoint can be considered to be from a safe source
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

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
        scheduler: Scheduler | None = None,
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
    best : bool, default=False
        Whether this was the best checkpoint so far (Default value = False).
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


# A read timeout, unlike a connect timeout, bounds the gap between received chunks. Without one a
# server that accepts the connection and then stops sending leaves this hanging with nothing to
# time it out, which in a docker build means hanging until the job's own limit.
DOWNLOAD_TIMEOUT = (5, 60)  # (connect, read) in seconds
# Attempts per url before moving to the next one. The urls are alternative hosts, so falling through
# already covers one being down; this covers the transfer itself breaking, which is what the hosts
# actually do, and it is the only thing protecting the last url in the list.
DOWNLOAD_ATTEMPTS = 3
DOWNLOAD_BACKOFF = 5.0  # seconds before the second attempt, doubled for each one after
# Statuses where the host is up but is refusing for now, which is what an overloaded host returns
# and is worth another attempt. Every other status, 404 and 403 in particular, is the same answer
# every time, so the next url is the faster move.
DOWNLOAD_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})


def download_checkpoint(
        checkpoint_name: str,
        checkpoint_path: str | Path,
        urls: Sequence[str],
        md5: str | None = None,
) -> None:
    """
    Download a checkpoint file.

    Each url is tried up to DOWNLOAD_ATTEMPTS times, backing off between attempts, for a broken
    transfer, a status in DOWNLOAD_RETRY_STATUS or a file that does not match md5. Any other status
    moves straight to the next url. Raises an ExceptionGroup, or a RuntimeError whenever no host
    answered with an error status, once every url has been exhausted. The file is only written to
    checkpoint_path once it matches md5.

    Parameters
    ----------
    checkpoint_name : str
        Name of checkpoint.
    checkpoint_path : Path, str
        Path of the file in which the checkpoint will be saved.
    urls : Sequence[str]
        URLs of checkpoint hosting sites.
    md5 : str, optional
        The md5 of the published file. None accepts any file, with a warning.
    """
    if md5 is None:
        LOGGER.warning(f"No md5 known for checkpoint {checkpoint_name}, so its download cannot be verified.")
    else:
        md5 = md5.lower()
    responses = []
    # the last outcome of each url that delivered files, but only with the wrong md5
    mismatches = []
    content = None
    for url in urls:
        # this url's own reply, so that exhausting the attempts here does not read the reply of
        # the url before it
        reply = None
        for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
            try:
                LOGGER.info(f"Downloading checkpoint {checkpoint_name} from {url}")
                reply = requests.get(
                    url + "/" + checkpoint_name,
                    verify=True,
                    timeout=DOWNLOAD_TIMEOUT,
                )
                if reply.ok:
                    actual_md5 = hashlib.md5(reply.content).hexdigest()
                    if md5 is None or actual_md5 == md5:
                        content = reply.content
                        break
                    reason = f"Checkpoint {checkpoint_name} from {url} has md5 {actual_md5}, expected {md5}"
                elif reply.status_code not in DOWNLOAD_RETRY_STATUS:
                    break  # a settled answer, and it decides whether to try the next url
                else:
                    reason = f"Server {url} answered {reply.status_code}"

            except requests.exceptions.RequestException as e:
                reason = f"Server {url} not reachable ({type(e).__name__}): {e}"
                if isinstance(e.response, requests.Response):
                    reply = e.response

            # the transport broke, the file arrived corrupted or the host asked for later, and another
            # attempt fixes any of these
            LOGGER.warning(reason)
            if attempt < DOWNLOAD_ATTEMPTS:
                delay = DOWNLOAD_BACKOFF * 2 ** (attempt - 1)
                LOGGER.info(
                    f"Retrying {url} in {delay:.0f}s ({attempt} of {DOWNLOAD_ATTEMPTS} used)"
                )
                sleep(delay)
        if content is not None:
            break
        # one entry per url rather than per attempt, so the error below reads as a list of hosts
        if reply is not None:
            if reply.ok:
                # the reply is the checkpoint itself, so its text has nothing to report
                mismatches.append(reason)
            else:
                responses.append(reply)

    # if no request was successful, raise an error with all responses
    if content is None:
        import textwrap
        # the urls, because a transport failure leaves no response to report below
        message = (f"Could not download checkpoint {checkpoint_name} from any of "
                   f"{', '.join(urls)}.")
        exceptions = []
        for _response in responses:
            message += f"\n\nResponse code from {_response.url}: {_response.status_code}"
            message += f"\nResponse text:\n{textwrap.indent(_response.text, '    ')}"
            try:
                _ = _response.raise_for_status()
            except Exception as e:
                exceptions.append(e)
        for mismatch in mismatches:
            message += f"\n\n{mismatch}"
        # exceptions is empty when every url failed in transport or delivered the wrong file, which
        # leaves no response to raise_for_status, and an ExceptionGroup must hold at least one
        # exception
        if exceptions:
            raise ExceptionGroup(message, exceptions)
        else:
            raise RuntimeError(message, responses)
    else:
        checkpoint_path = Path(checkpoint_path)
        temporary_path = checkpoint_path.with_name(
            f".{checkpoint_path.name}.{uuid4().hex}.tmp"
        )
        try:
            # Opening a unique file with ``xb`` preserves the permissions dictated by
            # the process umask and prevents concurrent downloads from sharing a
            # temporary file. The same-directory replace atomically publishes the
            # checkpoint only after it has been written and closed completely.
            with open(temporary_path, "xb") as f:
                f.write(content)
            os.replace(temporary_path, checkpoint_path)
        finally:
            temporary_path.unlink(missing_ok=True)


def _checkpoint_file(checkpoint: CheckpointFile | Path | str, function: str) -> CheckpointFile:
    """Return checkpoint as CheckpointFile, warning that a Path or str is deprecated."""
    if isinstance(checkpoint, CheckpointFile):
        return checkpoint
    warnings.warn(
        f"Passing a {type(checkpoint).__name__} to {function} is deprecated, pass a CheckpointFile, so the "
        f"download can be checked against its md5.",
        DeprecationWarning,
        stacklevel=3,
    )
    return CheckpointFile(Path(checkpoint))


def to_checkpoint_files(*paths: Path | str, published: Iterable[CheckpointFile]) -> list[CheckpointFile]:
    """
    Pair each path with the md5 of the published checkpoint of the same file name.

    The file name is what the download requests from the hosts, so a checkpoint path from the command
    line that points to another folder still gets the md5 of the published file.

    Parameters
    ----------
    *paths : Path, str
        Paths of the checkpoint files.
    published : Iterable[CheckpointFile]
        The published checkpoints, see load_checkpoint_config_defaults.

    Returns
    -------
    list[CheckpointFile]
        One CheckpointFile per path, the md5 is None, if no published checkpoint has its file name.
    """
    md5 = {file.path.name: file.md5 for file in published}
    return [CheckpointFile(path, md5.get(path.name)) for path in map(Path, paths)]


def check_and_download_ckpts(
        checkpoint: CheckpointFile | Path | str | None = None,
        urls: Sequence[str] = (),
        verify_existing: bool = True,
        *,
        checkpoint_path: Path | str | None = None,
) -> None:
    """
    Check and download a checkpoint file, if it does not exist.

    A checkpoint that already exists is not downloaded again. With verify_existing, it is checked against
    the md5, and a mismatch logs a warning, the file is used anyway.

    Parameters
    ----------
    checkpoint : CheckpointFile
        The file in which the checkpoint will be saved, and the md5 the download is checked against.
        Passing a Path or str is deprecated, their downloads are not checked.
    urls : Sequence of str
        URLs of checkpoint hosting sites, at least one.
    verify_existing : bool, default=True
        Whether to check the checkpoint against the md5, if it already exists.
    checkpoint_path : Path, str, optional
        Deprecated, use checkpoint. The former name of checkpoint, raises a DeprecationWarning.
    """
    if checkpoint_path is not None:
        if checkpoint is not None:
            raise TypeError("check_and_download_ckpts() got checkpoint and checkpoint_path, pass only checkpoint.")
        warnings.warn(
            "The checkpoint_path argument of check_and_download_ckpts is deprecated, use checkpoint.",
            DeprecationWarning,
            stacklevel=2,
        )
        checkpoint = checkpoint_path
    if checkpoint is None:
        raise TypeError("check_and_download_ckpts() missing required argument checkpoint.")
    if len(urls) == 0:
        raise ValueError("check_and_download_ckpts() requires at least one url in urls.")
    checkpoint_path, md5 = _checkpoint_file(checkpoint, "check_and_download_ckpts")
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.exists():
        if verify_existing and md5 is not None:
            with open(checkpoint_path, "rb") as file:
                actual = hashlib.file_digest(file, "md5").hexdigest()
            if actual != md5.lower():
                LOGGER.warning(
                    f"Checkpoint {checkpoint_path} has md5 {actual}, but the published file has {md5}. It is "
                    f"used anyway, delete it, so the published file is downloaded on the next run."
                )
    else:
        # create dir if it does not exist
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        download_checkpoint(checkpoint_path.name, checkpoint_path, urls, md5)


def get_checkpoints(
        *checkpoints: CheckpointFile | Path | str,
        urls: list[str],
        verify_existing: bool = True,
) -> list[Path]:
    """
    Check and download checkpoint files if not exist.

    Downloads are always checked against the md5 of their CheckpointFile, see download_checkpoint. With
    verify_existing, checkpoints that already exist are checked as well, and a mismatch logs a warning, see
    check_and_download_ckpts.

    Parameters
    ----------
    *checkpoints : CheckpointFile
        The files in which the checkpoints will be saved, and the md5 they are checked against, see
        to_checkpoint_files. Passing a Path or str is deprecated, their downloads are not checked.
    urls : list of str
        URLs of checkpoint hosting sites.
    verify_existing : bool, default=True
        Whether to also check checkpoints that already exist against their md5.

    Returns
    -------
    list[Path]
        The path of each checkpoint, relative paths resolved against the FastSurfer root directory.
    """
    files = []
    try:
        for checkpoint in checkpoints:
            file, md5 = _checkpoint_file(checkpoint, "get_checkpoints")
            file = Path(file)
            if not file.is_absolute() and file.parts[0] != ".":
                file = FASTSURFER_ROOT / file
            files.append(file)
            check_and_download_ckpts(CheckpointFile(file, md5), urls, verify_existing)
    except requests.exceptions.HTTPError:
        LOGGER.error(f"Could not find nor download checkpoints from {urls}")
        raise
    return files


def get_module_checkpoints(module: str, *paths: Path | str, verify_existing: bool = True) -> list[Path]:
    """
    Check and download checkpoint files of a FastSurfer module if not exist.

    The urls and the md5 of the published checkpoints are those in the checkpoint_paths.yaml of module,
    see get_config_file. Each path gets the md5 of the published checkpoint of the same file name, see
    to_checkpoint_files.

    Parameters
    ----------
    module : str
        The FastSurfer module the checkpoints belong to, e.g. "FastSurferCNN".
    *paths : Path, str
        Paths of the checkpoint files, relative paths are relative to the FastSurfer root directory.
    verify_existing : bool, default=True
        Whether to also check checkpoints that already exist against their md5, see get_checkpoints.

    Returns
    -------
    list[Path]
        The path of each checkpoint, relative paths resolved against the FastSurfer root directory.
    """
    config_file = get_config_file(module)
    urls = load_checkpoint_config_defaults("url", config_file)
    published = load_checkpoint_config_defaults("checkpoint", config_file).values()
    checkpoints = to_checkpoint_files(*paths, published=published)
    return get_checkpoints(*checkpoints, urls=urls, verify_existing=verify_existing)
