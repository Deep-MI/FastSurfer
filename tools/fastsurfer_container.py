#!/usr/bin/env python3
# Copyright 2026 DeepMI Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
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

"""
Run FastSurfer in its container image, without writing the container command.

Takes the flags of run_fastsurfer.sh (or, with --brun, of brun_fastsurfer.sh) and runs them in the
official image with Apptainer, Singularity or Docker, whichever is installed:

    python3 fastsurfer_container.py --t1 data/t1.nii.gz --sd results --sid subject1

It runs the FastSurfer release image installed on the machine (the newest release if there is none,
or the image --image names), picks the image type that fits the NVIDIA GPU and driver (or the CPU
image), downloads it once,
shares every file and directory the flags name with the container at the same path, passes the
FreeSurfer license from $FS_LICENSE if --fs_license is not given, and runs the container as the
calling user.

Only the Python standard library is used, so this runs with the system Python of a cluster,
without FastSurfer installed.
"""

from __future__ import annotations

import argparse
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

DOCKER_REPO = "deepmi/fastsurfer"
DOC_URL = "https://deep-mi.org/FastSurfer/stable/overview/CONTAINERS.html"
LAUNCHER_URL = "https://raw.githubusercontent.com/Deep-MI/FastSurfer/stable/tools/fastsurfer_container.py"
# seconds to wait for Docker Hub and GitHub, so a machine without internet is not held up
HUB_TIMEOUT = 5
MACOS_PACKAGE_URL = "https://github.com/Deep-MI/FastSurfer/releases/latest/download/FastSurfer-macos-darwin_arm64.pkg"
MACOS_DOC_URL = "https://deep-mi.org/FastSurfer/stable/overview/install/MACOS.html"
# the 15 GB the macOS page asks Docker to have, less what the kernel of Docker's VM keeps for itself
MIN_DOCKER_MEMORY_GIB = 14

# The CUDA images, as in FastSurferCNN/gpu_support.py: (CUDA version, tag prefix), the oldest GPU
# each supports and the newest the legacy one does. The test in test/lint keeps them in sync.
NEWEST_BUILD = ((13, 2), "cu132")
LEGACY_BUILD = ((12, 6), "cu126")
# CUDA 11 for older releases, whose images can still be installed
MIN_DRIVER = {11: 450, 12: 525, 13: 580}
MIN_CAPABILITY = {"cu132": (7, 5), "cu126": (5, 0)}
LEGACY_MAX_CAPABILITY = (9, 0)
# the graphics memory FastSurfer needs at 0.8 mm, see the system requirements in the README
MIN_GPU_MEMORY_GIB = 8

# flags of run_fastsurfer.sh and brun_fastsurfer.sh whose value is a file that has to exist
INPUT_FLAGS = {"--t1", "--t2", "--lesion_mask", "--fs_license", "--subjects_list", "--subject_list"}
# flags whose value is a file or directory FastSurfer writes, so its directory is created if missing
OUTPUT_FLAGS = {
    "--sd", "--seg_log", "--conformed_name", "--norm_name", "--norm_name_t2", "--seg",
    "--asegdkt_segfile", "--aparc_aseg_segfile", "--asegdkt_statsfile", "--aseg_statsfile",
    "--aseg_segfile", "--mask_name", "--cereb_segfile", "--cereb_statsfile", "--hypo_segfile",
    "--hypo_statsfile", "--statusfile",
}


class LauncherError(Exception):
    """A problem with the request, reported as one message without a traceback."""


@dataclass
class Gpu:
    """An NVIDIA GPU as nvidia-smi reports it; capability and memory are None if it cannot."""

    index: int
    capability: tuple[int, int] | None
    driver: int
    memory_gib: float | None = None


@dataclass
class Build:
    """The image build to run, whether the container gets GPUs, and which (None for all)."""

    prefix: str
    gpu: bool
    notes: list[str] = field(default_factory=list)
    devices: list[int] | None = None


def parse_version(version: str) -> tuple[int, ...] | None:
    """A release version such as 2.6.0 as numbers, None for anything else (e.g. a development build)."""
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", version)
    return tuple(int(v) for v in match.groups()) if match else None


def release_versions(tags: list[str], prefix: str) -> list[tuple[int, ...]]:
    """The releases among image tags such as cu132-v2.6.0, oldest first, ignoring development builds."""
    versions = set()
    for tag in tags:
        if tag.startswith(f"{prefix}-v"):
            version = parse_version(tag[len(prefix) + 2:])
            if version is not None:
                versions.add(version)
    return sorted(versions)


def newest_version(tags: list[str], prefix: str) -> str | None:
    """The newest release among image tags such as cu132-v2.6.0, ignoring development builds."""
    versions = release_versions(tags, prefix)
    return ".".join(str(v) for v in versions[-1]) if versions else None


def age_note(version: str, releases: list[tuple[int, ...]], gpu_unsupported: bool) -> tuple[str, str] | None:
    """
    A note for a run of an older release, as (level, message), None for the newest or an unknown version.

    Parameters
    ----------
    version : str
        The version --image requested.
    releases : list[tuple[int, ...]]
        All releases, oldest first.
    gpu_unsupported : bool
        Whether the newest images do not support this GPU, so an old image is the only way to use it.
    """
    requested = parse_version(version)
    if requested is None or not releases or requested >= releases[-1]:
        return None
    newest = ".".join(str(v) for v in releases[-1])
    behind = sum(release > requested for release in releases)
    message = f"FastSurfer {version} is {behind} release{'s' if behind > 1 else ''} older than the newest, {newest}."
    if gpu_unsupported:
        return "WARNING", (
            f"{message} The images of newer versions do not support this GPU, so they run on the CPU here. "
            f"Running the newest release on the CPU (--update without --image), or on a machine with a newer "
            f"GPU, is recommended over this old version, which lacks the improvements and fixes since."
        )
    return "INFO", f"{message} --update without --image switches to the newest release."


def query_nvidia_gpus() -> list[Gpu]:
    """
    The NVIDIA GPUs, from nvidia-smi, in its (PCI bus) order.

    Empty without an NVIDIA GPU. A capability or memory the driver cannot report, e.g. as N/A, is None.
    """
    if shutil.which("nvidia-smi") is None:
        return []
    for fields in ("index,compute_cap,driver_version,memory.total", "index,driver_version,memory.total"):
        try:
            out = subprocess.run(
                ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True, timeout=30,
            ).stdout.strip().splitlines()
        except (OSError, subprocess.SubprocessError):
            continue
        gpus = []
        try:
            for line in out:
                values = [v.strip() for v in line.split(",")]
                capability = None
                if len(values) == 4 and re.fullmatch(r"\d+\.\d+", values[1]):
                    major, minor = values[1].split(".")
                    capability = (int(major), int(minor))
                memory = float(values[-1]) / 1024 if re.fullmatch(r"\d+(\.\d+)?", values[-1]) else None
                gpus.append(Gpu(int(values[0]), capability, int(values[-2].split(".")[0]), memory))
        except (ValueError, IndexError):
            continue  # an unreadable index or driver version: try the next query
        return gpus
    return []


def _runs(build: tuple[tuple[int, int], str], gpus: list[Gpu]) -> bool:
    """Whether the image of this build runs on all of these GPUs with their driver."""
    (cuda_major, _), prefix = build
    for gpu in gpus:
        if gpu.driver < MIN_DRIVER[cuda_major]:
            return False
        if gpu.capability is not None:
            if prefix == LEGACY_BUILD[1] and gpu.capability > LEGACY_MAX_CAPABILITY:
                return False
            if gpu.capability < MIN_CAPABILITY[prefix]:
                return False
    return True


def _best_build(gpus: list[Gpu]) -> tuple[tuple[int, int], str] | None:
    """The newest CUDA build whose image runs on all of these GPUs, None if there is none."""
    return next((build for build in (NEWEST_BUILD, LEGACY_BUILD) if _runs(build, gpus)), None)


def image_type_runs(image_type: str, gpus: list[Gpu]) -> bool:
    """
    Whether images of a type (cu132, cu118, cpu, ...) run on these GPUs, or, without GPUs, on the CPU.

    The GPUs of the current CUDA types are known; for older ones, only the driver can be checked.
    """
    if not gpus:
        return image_type == "cpu"
    match = re.fullmatch(r"cu(\d+)(\d)", image_type)
    if match is None:
        return False
    build = next((b for b in (NEWEST_BUILD, LEGACY_BUILD) if b[1] == image_type), None)
    if build is not None:
        return _runs(build, gpus)
    major = int(match.group(1))
    return major in MIN_DRIVER and min(gpu.driver for gpu in gpus) >= MIN_DRIVER[major]


def describe(gpu: Gpu) -> str:
    """E.g. "GPU 1 (compute capability 8.6, 24 GB)"."""
    details = []
    if gpu.capability is not None:
        details.append(f"compute capability {gpu.capability[0]}.{gpu.capability[1]}")
    if gpu.memory_gib is not None:
        details.append(f"{gpu.memory_gib:.0f} GB")
    return f"GPU {gpu.index}" + (f" ({', '.join(details)})" if details else "")


def choose_build(gpus: list[Gpu], use: str) -> Build:
    """
    The image that runs on all of these GPUs and their driver, preferring the default CUDA version.

    Parameters
    ----------
    gpus : list[Gpu]
        The GPUs FastSurfer will use.
    use : str
        "cuda" if a GPU was requested explicitly, "auto" if FastSurfer would use one if it can, and
        "cpu" if it runs on the CPU only.

    Returns
    -------
    Build
        The tag prefix, whether to give the container the GPU, and notes for the user.
    """
    if use == "cpu":
        return Build("cpu", False)
    if not gpus:
        if use == "cuda":
            raise LauncherError("A cuda device was requested, but nvidia-smi found no NVIDIA GPU.")
        return Build("cpu", False, ["No NVIDIA GPU found, so FastSurfer runs on the CPU."])
    driver = min(gpu.driver for gpu in gpus)
    capabilities = [gpu.capability for gpu in gpus if gpu.capability is not None]
    newest_min_driver = MIN_DRIVER[NEWEST_BUILD[0][0]]
    build = _best_build(gpus)
    if build == NEWEST_BUILD:
        return Build(NEWEST_BUILD[1], True)
    if build == LEGACY_BUILD:
        notes = []
        if driver < newest_min_driver and all(c >= MIN_CAPABILITY[NEWEST_BUILD[1]] for c in capabilities):
            notes.append(
                f"The NVIDIA driver {driver} is older than {newest_min_driver}, so this uses the image for "
                f"CUDA {LEGACY_BUILD[0][0]}.{LEGACY_BUILD[0][1]}. Update the driver to use the default image."
            )
        return Build(LEGACY_BUILD[1], True, notes)
    driver_reason = (f"The NVIDIA driver {driver} is too old for FastSurfer's images, update it to "
                     f"{newest_min_driver} or newer")
    if not capabilities:
        # with unknown capabilities, only the driver can rule the images out
        reason = driver_reason
    elif min(capabilities) < MIN_CAPABILITY[LEGACY_BUILD[1]]:
        oldest = min(capabilities)
        reason = f"The GPU (compute capability {oldest[0]}.{oldest[1]}) is too old for FastSurfer's images"
    elif (driver >= newest_min_driver and min(capabilities) < MIN_CAPABILITY[NEWEST_BUILD[1]]
          and max(capabilities) > LEGACY_MAX_CAPABILITY):
        oldest, newest = min(capabilities), max(capabilities)
        reason = (f"No FastSurfer image supports both GPUs of compute capability {oldest[0]}.{oldest[1]} "
                  f"and {newest[0]}.{newest[1]}; select GPUs of one kind with --device cuda:<index>")
    else:
        reason = driver_reason
    if use == "cuda":
        raise LauncherError(f"{reason}.")
    return Build("cpu", False, [f"{reason}, so FastSurfer runs on the CPU."])


def select_gpus(gpus: list[Gpu], brun: bool) -> Build | None:
    """
    The GPU to run on when none was named: one an image supports, with enough memory, the newest.

    The container then sees only that GPU, so FastSurfer uses it as cuda:0. With brun_fastsurfer.sh,
    it sees every GPU the chosen image supports. None if no image supports any of the GPUs.
    """
    supported = [gpu for gpu in gpus if _best_build([gpu]) is not None]
    if not supported:
        return None

    def rank(gpu: Gpu) -> tuple:
        memory = gpu.memory_gib or 0.0
        if gpu.memory_gib is None or memory >= MIN_GPU_MEMORY_GIB:
            return True, gpu.capability or (0, 0), memory
        return False, (memory,), gpu.capability or (0, 0)

    best = max(supported, key=rank)
    build_tuple = _best_build([best])
    chosen = [gpu for gpu in supported if _runs(build_tuple, [gpu])] if brun else [best]
    build = choose_build([best], "auto")
    build.devices = [gpu.index for gpu in chosen]
    build.notes.insert(0, "Using " + ", ".join(describe(gpu) for gpu in chosen) + ".")
    return build


def cuda_indices(value: str) -> list[int] | None:
    """
    The GPU indices a device value names, e.g. [0, 1, 2, 5] for cuda:0-2,5, and None if it names none.

    Accepts the forms run_fastsurfer.sh and brun_fastsurfer.sh accept: cuda:<i>, lists and ranges.
    """
    if not value.startswith("cuda:"):
        return None
    indices = []
    for part in value[len("cuda:"):].split(","):
        match = re.fullmatch(r"(\d+)(?:-(\d+))?", part)
        if match is None:
            raise LauncherError(f"Cannot read the device {value}; use e.g. cuda:0, cuda:0,1 or cuda:0-3.")
        first, last = int(match.group(1)), int(match.group(2) or match.group(1))
        if last < first:
            # an empty range would read as no GPU named, and the launcher would choose one itself
            raise LauncherError(f"Cannot read the device {value}; a range has to go upward, e.g. cuda:0-3.")
        indices.extend(range(first, last + 1))
    return indices


def gpu_request(args: list[str], brun: bool = False) -> tuple[str, list[int] | None]:
    """
    Whether FastSurfer would use a GPU with these arguments, and the indices of the GPUs it names.

    --device decides the segmentation and --viewagg_device the view aggregation, which runs on the CPU
    with --device cpu unless it names cuda itself. Lists and ranges of GPUs (cuda:0,1, cuda:0-3) are
    for brun_fastsurfer.sh only; run_fastsurfer.sh takes one GPU.

    Returns
    -------
    str
        "cuda" if a GPU is requested, "auto" if FastSurfer would use one if it can, "cpu" otherwise.
    list[int], None
        The GPUs named with cuda:<index>, None if none is named.
    """
    values = [_last_value(args, "--device") or "auto", _last_value(args, "--viewagg_device") or "auto"]
    cuda = [v for v in values if v == "cuda" or v.startswith("cuda:")]
    several = [v for v in cuda if v.startswith("cuda:") and not re.fullmatch(r"cuda:\d+", v)]
    if several and not brun:
        raise LauncherError(f"{several[0]}: run_fastsurfer.sh uses one GPU, e.g. cuda:1. Lists and ranges of "
                            f"GPUs are for processing several subjects with --brun.")
    named = [i for v in cuda for i in (cuda_indices(v) or [])]
    if named and values[0] in ("auto", "cuda"):
        # only the view aggregation names a GPU; the segmentation then runs on the first one
        named.append(0)
    use = "cuda" if cuda else ("auto" if values[0] == "auto" else "cpu")
    return use, sorted(set(named)) if named else None


def plan_build(args: list[str], gpus: list[Gpu], brun: bool) -> Build:
    """
    The image and the GPUs for these arguments.

    Named GPUs are used as they are, and explained in detail if no image supports them. Otherwise the
    best GPU an image supports is selected, and the CPU only used if there is none.
    """
    use, named = gpu_request(args, brun)
    if use == "cpu":
        return Build("cpu", False)
    if named is not None:
        by_index = {gpu.index: gpu for gpu in gpus}
        missing = [i for i in named if i not in by_index]
        if missing:
            raise LauncherError(f"There is no GPU {missing[0]}, nvidia-smi found {len(gpus)}.")
        # the container sees all GPUs, so the named indices keep their meaning
        return choose_build([by_index[i] for i in named], "cuda")
    return select_gpus(gpus, brun) or choose_build(gpus, use)


def image_reference(prefix: str, version: str) -> str:
    """The Docker image to run, e.g. deepmi/fastsurfer:cu132-v2.6.0."""
    return f"{DOCKER_REPO}:{prefix}-v{version}"


RELEASE_TAG = re.compile(r"([a-z0-9.]+)-v(\d+\.\d+\.\d+)")
# official Apptainer images are named fastsurfer-<tag>.sif, as in the documentation
SIF_PREFIX = DOCKER_REPO.rsplit("/", 1)[-1] + "-"


def hub_tags() -> dict[str, set[str]] | None:
    """
    The release tags on Docker Hub and the digests each is known by, None if Docker Hub cannot be reached.

    One request per run: every decision about versions comes from this list. A tag is known by the
    digest of its index and those of its platform images, so a local copy matches whichever one Docker
    recorded.
    """
    import json
    import urllib.request

    url = f"https://hub.docker.com/v2/repositories/{DOCKER_REPO}/tags?page_size=100&name=-v"
    tags: dict[str, set[str]] = {}
    try:
        while url and len(tags) < 2000:
            with urllib.request.urlopen(url, timeout=HUB_TIMEOUT) as response:
                page = json.load(response)
            for result in page.get("results", []):
                digests = {result.get("digest")} | {image.get("digest") for image in result.get("images", [])}
                tags[result["name"]] = {d for d in digests if d}
            url = page.get("next")
    except (OSError, ValueError, KeyError):
        return None
    return tags


def local_docker_images() -> dict[str, str] | None:
    """
    The tags of the official images in the local Docker with the digest each was pulled with.

    The digest is "" for an image not pulled from Docker Hub, e.g. a local build tagged like a release.
    None if Docker is not installed or not running.
    """
    if shutil.which("docker") is None:
        return None
    try:
        out = subprocess.run(["docker", "image", "ls", DOCKER_REPO, "--format", "{{.Tag}}\t{{.Digest}}"],
                             capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    images = {}
    for line in out.stdout.splitlines():
        tag, _, digest = line.partition("\t")
        if tag and tag != "<none>":
            images[tag] = "" if digest in ("", "<none>") else digest
    return images


def local_docker_id(reference: str) -> str | None:
    """The ID of an image in the local Docker, None if it is not there."""
    if shutil.which("docker") is None:
        return None
    try:
        out = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", reference],
                             capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def sif_path(image_dir: Path, tag: str) -> Path:
    """Where the Apptainer image of a release tag is kept, e.g. fastsurfer-cu132-v2.6.0.sif."""
    return image_dir / f"{SIF_PREFIX}{tag}.sif"


def _recorded(sif: Path) -> str:
    """What the launcher wrote next to an image it built: the digest or image ID of its source, or ""."""
    record = sif.with_name(sif.name + ".digest")
    return record.read_text().strip() if record.is_file() else ""


def local_sif_images(image_dir: Path) -> dict[str, str]:
    """The tags of the Apptainer images in image_dir, with the digest recorded when they were built ("")."""
    if not image_dir.is_dir():
        return {}
    return {p.name[len(SIF_PREFIX):-len(".sif")]: _recorded(p) for p in image_dir.glob(f"{SIF_PREFIX}*.sif")}


def parse_image_spec(spec: str | None) -> tuple[str, str]:
    """
    What --image names.

    Returns
    -------
    tuple[str, str]
        ("auto", "") without --image, ("version", "2.6.0"), ("tag", "cu132-v2.6.0") for a release tag,
        ("release", "cu132-v2.6.0") for the full reference deepmi/fastsurfer:cu132-v2.6.0, ("sif", path)
        for an Apptainer file and ("docker", reference) for any other Docker image.
    """
    if not spec:
        return "auto", ""
    if spec.endswith(".sif"):
        return "sif", spec
    if parse_version(spec):
        return "version", spec
    if RELEASE_TAG.fullmatch(spec):
        return "tag", spec
    official = re.fullmatch(rf"(?:docker\.io/)?{re.escape(DOCKER_REPO)}:(.+)", spec)
    if official and RELEASE_TAG.fullmatch(official.group(1)):
        return "release", official.group(1)
    return "docker", spec


@dataclass
class ImagePlan:
    """The image to run and how to get it."""

    runtime: str
    # the Docker reference, or the path of the .sif file for Apptainer
    image: str
    # the source to build the .sif file from, docker://... or docker-daemon://..., if it has to be built
    build_from: str | None = None
    # what to write next to a built .sif file, to recognise its source later
    record: str | None = None
    pull: bool = False
    tag: str | None = None
    notes: list[tuple[str, str]] = field(default_factory=list)


def _apptainer_runtime(runtime: str, installed: set[str]) -> str | None:
    """The Apptainer or Singularity to use, None if none is requested or installed."""
    if runtime in ("apptainer", "singularity"):
        return runtime
    if runtime == "auto":
        return next((name for name in ("apptainer", "singularity") if name in installed), None)
    return None


def decide_image(kind: str, value: str, prefix: str, runtime: str, installed: set[str],
                 docker_images: dict[str, str] | None, sif_images: dict[str, str],
                 hub: dict[str, set[str]] | None, image_dir: Path, update: bool,
                 local_id: str | None = None, gpus: list[Gpu] | None = None) -> ImagePlan:
    """
    Which image to run, with which runtime, and whether it has to be downloaded or built first.

    Without --image, an official release image already installed that runs on this machine is used,
    wherever it is, also one of an older type than the newest images; several stop the launcher, none
    downloads the newest release. With --update, the newest release (or the one named) is downloaded
    again. A release image whose content differs from Docker Hub stops the launcher, unless it was
    named in full.

    Parameters
    ----------
    kind, value : str
        What --image names, see parse_image_spec.
    prefix : str
        The image type of the newest images for this machine, e.g. cu132.
    runtime : str
        --runtime: auto, apptainer, singularity or docker.
    installed : set[str]
        The runtimes installed.
    docker_images : dict[str, str], None
        Tags and digests of the official images in the local Docker, None without Docker.
    sif_images : dict[str, str]
        Tags and recorded digests of the official images in `image_dir`.
    hub : dict[str, set[str]], None
        Release tags and digests on Docker Hub, None if it cannot be reached.
    image_dir : Path
        Where Apptainer images are kept.
    update : bool
        --update: download or build the image again.
    local_id : str, None
        For a "docker" image, its ID in the local Docker, None if it is not there.
    gpus : list[Gpu], None
        The GPUs the run will use, to tell which installed images run here; empty for a run on the CPU.
        None counts only images of type `prefix`.
    """
    apptainer = _apptainer_runtime(runtime, installed)
    docker_images = docker_images or {}

    def require(name: str) -> str:
        if name not in installed:
            raise LauncherError(f"{name} is not installed.")
        return name

    def place_release(tag: str, checked: bool, forced: bool = update) -> ImagePlan:
        in_docker, in_sif = tag in docker_images, tag in sif_images
        if runtime == "docker" or (runtime == "auto" and not (in_sif and apptainer)
                                   and ((in_docker and "docker" in installed) or apptainer is None)):
            chosen = require("docker")
        else:
            chosen = require(apptainer)
        local = docker_images.get(tag) if chosen == "docker" else sif_images.get(tag)
        known = hub.get(tag) if hub is not None else None
        # a recorded digest of "" on a .sif file means it was not built by the launcher: nothing to compare
        comparable = local is not None and (chosen == "docker" or local)
        if checked and not forced and known and comparable and local not in known:
            where = (f"Docker image {DOCKER_REPO}:{tag}" if chosen == "docker"
                     else f"image {sif_path(image_dir, tag)}")
            explicit = f"{DOCKER_REPO}:{tag}" if chosen == "docker" else str(sif_path(image_dir, tag))
            raise LauncherError(f"The local {where} differs from the release on Docker Hub, e.g. a local build "
                                f"or an image replaced since. --update replaces it with the release; "
                                f"--image {explicit} runs the local one anyway.")
        if chosen == "docker":
            return ImagePlan("docker", f"{DOCKER_REPO}:{tag}", pull=forced, tag=tag)
        sif = sif_path(image_dir, tag)
        build = forced or not in_sif
        # the release pulled into the local Docker: build from there, without a second download (and
        # without Docker Hub credentials, which Apptainer may send even for this public image)
        pulled = docker_images.get(tag, "")
        from_docker = "docker" in installed and pulled and (known is None or pulled in known)
        source = f"docker-daemon://{DOCKER_REPO}:{tag}" if from_docker else f"docker://{DOCKER_REPO}:{tag}"
        record = (sorted(known)[0] if known else pulled or None) if build else None
        return ImagePlan(chosen, str(sif), build_from=source if build else None, record=record, tag=tag)

    def release_exists(tag: str) -> None:
        if hub is not None and tag not in hub and tag not in docker_images and tag not in sif_images:
            version = tag.rsplit("-v", 1)[-1]
            images = sorted(t for t in hub if t.endswith(f"-v{version}"))
            raise LauncherError(f"There is no image {DOCKER_REPO}:{tag}. " + (
                f"The images of FastSurfer {version} are {', '.join(images)}." if images
                else f"FastSurfer {version} is not released."))

    if kind == "sif":
        if runtime == "docker":
            raise LauncherError("A .sif image needs Apptainer or Singularity, not Docker.")
        if apptainer is None:
            raise LauncherError("A .sif image needs Apptainer or Singularity, and neither is installed.")
        return ImagePlan(require(apptainer), value)
    if kind == "docker":
        if runtime == "docker" or (runtime == "auto" and (local_id is not None or apptainer is None)):
            return ImagePlan(require("docker"), value, pull=update)
        # Apptainer runs a Docker image as a .sif file, converted from the local Docker if it is there
        sif = image_dir / (value.rsplit("/", 1)[-1].replace(":", "-") + ".sif")
        source = f"docker-daemon://{value}" if local_id else f"docker://{value}"
        stale = sif.is_file() and local_id is not None and _recorded(sif) != local_id
        build = update or not sif.is_file() or stale
        return ImagePlan(require(apptainer), str(sif), build_from=source if build else None,
                         record=local_id if build else None)
    if kind in ("tag", "release"):
        release_exists(value)
        return place_release(value, checked=kind == "tag")
    if kind == "version":
        tag = f"{prefix}-v{value}"
        release_exists(tag)
        return place_release(tag, checked=True)

    # no --image: the installed release image that runs here, or the newest release
    def runs_here(image_type: str) -> bool:
        return image_type == prefix if gpus is None else image_type_runs(image_type, gpus)

    candidates = sorted({t for t in [*docker_images, *sif_images]
                         if (m := RELEASE_TAG.fullmatch(t)) and runs_here(m.group(1))},
                        key=lambda t: (parse_version(t.rsplit("-v", 1)[-1]), t))
    newest = newest_version(list(hub), prefix) if hub is not None else None
    if update or not candidates:
        if newest is None:
            raise LauncherError("Docker Hub cannot be reached, so the newest FastSurfer release is not known "
                                + ("and --update cannot download it." if update else
                                   "and no FastSurfer image is on this machine yet. Pass the image with --image, "
                                   "e.g. --image 2.6.0."))
        tag = f"{prefix}-v{newest}"
        plan = place_release(tag, checked=True, forced=update)
        old = [t for t in candidates if t != tag]
        if old:
            removal = [f"docker image rm {DOCKER_REPO}:{t}" for t in old if t in docker_images]
            removal += [f"rm {sif_path(image_dir, t)}" for t in old if t in sif_images]
            plan.notes.append(("INFO", f"Older images are still installed ({', '.join(old)}), so the next run "
                                       f"would ask which one to use. Remove them with: {'; '.join(removal)}"))
    elif len(candidates) == 1:
        tag = candidates[0]
        plan = place_release(tag, checked=True)
        version = tag.rsplit("-v", 1)[-1]
        if newest is not None and parse_version(newest) > parse_version(version):
            plan.notes.append(("INFO", f"FastSurfer {newest} is available ({prefix}-v{newest}); --update "
                                       f"switches to it."))
    else:
        def stores(t: str) -> str:
            return " and ".join(s for s, has in (("Docker", t in docker_images), ("Apptainer", t in sif_images)) if has)

        where = [f"{t} ({stores(t)})" for t in candidates]
        update_hint = (f", or download the newest release ({prefix}-v{newest}) with --update" if newest is not None
                       else "")
        raise LauncherError(f"Several FastSurfer images that run here are installed: {'; '.join(where)}. Choose "
                            f"one with --image <tag>, e.g. --image {candidates[-1]}{update_hint}.")
    plan.notes.insert(0, ("INFO", f"Running FastSurfer {plan.tag.rsplit('-v', 1)[-1]} ({plan.tag}). For a study, "
                                  f"keep this image with --image {plan.tag}."))
    return plan


def _absolute(value: str, cwd: Path) -> Path:
    return Path(os.path.normpath(cwd / os.path.expanduser(value)))


def _parent_of_file(path: Path) -> Path:
    return path if path.is_dir() else path.parent


def _link_directories(path: Path) -> list[Path]:
    """
    The directories to share so that `path` can be read in the container: its own, and for a symbolic link
    that of every link on the way to the file it finally names (e.g. a DataLad or git-annex dataset).
    """
    directories, seen = [], set()
    while path.is_symlink() and path not in seen:
        seen.add(path)
        directories.append(path.parent)
        # a relative target is relative to the real directory of the link
        path = Path(os.path.normpath(Path(os.path.realpath(path.parent)) / os.readlink(path)))
    directories.append(_parent_of_file(path))
    return directories


def _mount_for(flag: str, path: Path, create: bool) -> list[Path]:
    """The directories to share for a path flag; an output directory is created, as a mount needs it."""
    if flag not in OUTPUT_FLAGS:
        return _link_directories(path)
    directory = path if flag == "--sd" else path.parent
    if create:
        directory.mkdir(parents=True, exist_ok=True)
    return [directory]


def plan_paths(args: list[str], cwd: Path, create: bool = True) -> tuple[list[str], list[Path]]:
    """
    Make the paths in FastSurfer's arguments absolute and find the directories to share.

    Parameters
    ----------
    args : list[str]
        The arguments for run_fastsurfer.sh or brun_fastsurfer.sh.
    cwd : Path
        The directory relative paths are relative to.
    create : bool, default=True
        Whether to create missing output directories (they have to exist to be shared).

    Returns
    -------
    list[str]
        The arguments, with every path absolute.
    list[Path]
        The directories to share with the container, at the same path.
    """
    out: list[str] = []
    mounts: list[Path] = []
    i = 0
    while i < len(args):
        flag = args[i]
        out.append(flag)
        i += 1
        if flag in INPUT_FLAGS | OUTPUT_FLAGS and i < len(args):
            path = _absolute(args[i], cwd)
            if flag in INPUT_FLAGS and not path.exists():
                raise LauncherError(f"{flag} {args[i]}: no such file.")
            mounts.extend(_mount_for(flag, path, create))
            if flag in ("--subjects_list", "--subject_list"):
                mounts.extend(_subjects_list_mounts(path, create))
            out.append(str(path))
            i += 1
        elif flag == "--subjects":
            # brun_fastsurfer.sh: <subject_id>=<t1_path> ... until the next flag
            while i < len(args) and not args[i].startswith("--"):
                subject, sep, t1 = args[i].partition("=")
                if not sep:
                    raise LauncherError(f"--subjects {args[i]}: expected <subject_id>=<t1_path>.")
                path = _absolute(t1, cwd)
                if not path.exists():
                    raise LauncherError(f"--subjects {args[i]}: no such file {t1}.")
                mounts.extend(_link_directories(path))
                out.append(f"{subject}={path}")
                i += 1
    return out, mounts


def _subjects_list_mounts(subjects_list: Path, create: bool = True) -> list[Path]:
    """
    The directories to share for a brun_fastsurfer.sh subjects list, whose paths must be absolute.

    Covers the T1 of each subject and the paths in its subject-specific options, inputs and outputs
    (e.g. its own --sd), and creates output directories like plan_paths.
    """
    mounts = []
    for number, line in enumerate(subjects_list.read_text().splitlines(), start=1):
        tokens = shlex.split(line, comments=True)
        if not tokens:
            continue
        _, _, t1 = tokens[0].partition("=")
        paths = [("--t1", t1)] + [(tok, tokens[k + 1]) for k, tok in enumerate(tokens[1:-1], start=1)
                                  if tok in INPUT_FLAGS | OUTPUT_FLAGS]
        for flag, value in paths:
            # brun_fastsurfer.sh expands nothing in a subjects list, so ~ is not the home directory there
            if not os.path.isabs(value):
                raise LauncherError(
                    f"{subjects_list}, line {number}: {value} is not an absolute path. Use absolute paths "
                    f"in a subjects list, without ~: the container does not start in the current directory, "
                    f"and nothing in the list is expanded."
                )
            if flag not in OUTPUT_FLAGS and not Path(value).exists():
                raise LauncherError(f"{subjects_list}, line {number}: no such file {value}.")
            mounts.extend(_mount_for(flag, Path(value), create))
    return mounts


def minimal_mounts(paths: list[Path]) -> list[Path]:
    """The given directories without duplicates and without those inside another one."""
    kept: list[Path] = []
    for path in sorted(set(paths), key=lambda p: len(p.parts)):
        if not any(path == k or k in path.parents for k in kept):
            kept.append(path)
    return sorted(kept)


def runtime_too_old(version_text: str) -> str | None:
    """
    Why this Apptainer/Singularity is too old for the launcher's command, None if it is not.

    The command needs --no-mount and --env, which Singularity has from 3.7; every Apptainer has them.

    Parameters
    ----------
    version_text : str
        The output of `<runtime> --version`, e.g. "singularity-ce version 3.11.4".
    """
    match = re.search(r"(apptainer|singularity)\S*\s+version\s+(\d+)\.(\d+)", version_text)
    if match is None or match.group(1) == "apptainer":
        return None
    if (int(match.group(2)), int(match.group(3))) >= (3, 7):
        return None
    return (f"Singularity {match.group(2)}.{match.group(3)} is too old for this launcher, which needs "
            f"Singularity 3.7 or newer, or Apptainer. Ask for a newer version on this machine, or write the "
            f"container command yourself, see {DOC_URL}.")


# CUDA numbers GPUs fastest first, nvidia-smi and the launcher by PCI bus; this makes them agree
CUDA_ORDER = "CUDA_DEVICE_ORDER=PCI_BUS_ID"


def apptainer_command(runtime: str, sif: Path, gpu: bool, mounts: list[Path], script: str,
                      args: list[str], extra: list[str], devices: list[int] | None = None) -> list[str]:
    """The Apptainer/Singularity command, showing only `devices` if given; the home directory is not mounted."""
    cmd = [runtime, "exec"]
    if gpu:
        cmd += ["--nv", "--env", CUDA_ORDER]
        if devices is not None:
            cmd += ["--env", f"CUDA_VISIBLE_DEVICES={','.join(map(str, devices))}"]
    cmd += ["--no-mount", "home,cwd", "-e"]
    for mount in mounts:
        cmd += ["-B", str(mount)]
    return cmd + extra + [str(sif), f"/fastsurfer/{script}"] + args


def docker_command(image: str, gpu: bool, mounts: list[Path], script: str, args: list[str],
                   extra: list[str], user: str, emulate: bool = False, devices: list[int] | None = None,
                   ) -> list[str]:
    """
    The Docker command, running as the calling user and showing only `devices` if given.

    `emulate` runs the Intel image on Apple silicon.
    """
    # --init: the image's entrypoint is a shell, which as the first process ignores Ctrl-C and the
    # signal a cancelled cluster job sends; tini passes them to all of FastSurfer's processes instead
    cmd = ["docker", "run", "--rm", "--init", "--env", "TINI_KILL_PROCESS_GROUP=1", "--user", user]
    if gpu:
        # the quotes are part of the value: without them, Docker splits a list of devices at the comma
        selection = "all" if devices is None else f'"device={",".join(map(str, devices))}"'
        cmd += ["--gpus", selection, "--env", CUDA_ORDER]
    if emulate:
        cmd += ["--platform", "linux/amd64"]
    for mount in mounts:
        cmd += ["-v", f"{mount}:{mount}"]
    cmd += extra
    if script != "run_fastsurfer.sh":
        # the image's entrypoint runs run_fastsurfer.sh; keep its environment setup for the other scripts
        return cmd + ["--entrypoint", "/fastsurfer/tools/Docker/entrypoint.sh", image, f"/fastsurfer/{script}"] + args
    return cmd + [image] + args


def is_apple_silicon() -> bool:
    """Whether this is a Mac with Apple silicon, also when Python itself runs under Rosetta."""
    if platform.system() != "Darwin":
        return False
    if platform.machine() == "arm64":
        return True
    try:
        out = subprocess.run(["sysctl", "-n", "hw.optional.arm64"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return False
    return out.stdout.strip() == "1"


def docker_memory_gib() -> float:
    """The memory Docker's containers can use; on macOS and Windows, that of Docker's VM."""
    try:
        out = subprocess.run(["docker", "info", "--format", "{{.MemTotal}}"],
                             capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as error:
        raise LauncherError(f"Could not ask Docker for its settings: {error}") from error
    if out.returncode != 0:
        reason = (out.stderr.strip().splitlines() or ["docker info failed"])[-1]
        raise LauncherError(f"Docker is installed, but not running: {reason}")
    return int(out.stdout.strip()) / 2**30


def _last_value(args: list[str], flag: str) -> str | None:
    """The value of the last `flag` in the arguments, as the scripts apply repeated flags in order."""
    for i in range(len(args) - 2, -1, -1):
        if args[i] == flag:
            return args[i + 1]
    return None


def _license_args(args: list[str], cwd: Path) -> list[str]:
    """--fs_license from $FS_LICENSE or $FREESURFER_HOME, if the arguments have none."""
    if "--fs_license" in args:
        return []
    candidates = [os.environ.get("FS_LICENSE", "")]
    if os.environ.get("FREESURFER_HOME"):
        candidates += [os.path.join(os.environ["FREESURFER_HOME"], name) for name in ("license.txt", ".license")]
    for candidate in candidates:
        if candidate and _absolute(candidate, cwd).is_file():
            return ["--fs_license", candidate]
    return []


def build_image(command: list[str]) -> bool:
    """
    Run `<runtime> build <sif> <source>` into a temporary file, and move it in place when it is complete.

    Parallel runs, e.g. the jobs of a SLURM array, and an interrupted build then never leave a partial
    image at the path the next run uses. Of two parallel builds, the last one to finish replaces the other.
    """
    sif = Path(command[2])
    sif.parent.mkdir(parents=True, exist_ok=True)
    partial = sif.with_name(f".{sif.stem}.{os.getpid()}.partial.sif")
    try:
        if subprocess.run(command[:2] + [str(partial)] + command[3:]).returncode != 0:
            return False
        os.replace(partial, sif)
        return True
    finally:
        if partial.exists():
            partial.unlink()


def stable_launcher() -> bytes | None:
    """The launcher on the stable branch, which knows the images of the newest release; None if unreachable."""
    import urllib.request

    try:
        with urllib.request.urlopen(LAUNCHER_URL, timeout=HUB_TIMEOUT) as response:
            return response.read()
    except OSError:
        return None


def update_launcher() -> int:
    """--update_launcher: replace this file with the launcher of the stable branch."""
    me = Path(__file__).resolve()
    if any((parent / ".git").exists() for parent in me.parents):
        print(f"ERROR: {me} is part of a git checkout, update it with git pull instead.", file=sys.stderr)
        return 1
    new = stable_launcher()
    if new is None:
        print(f"ERROR: Could not download {LAUNCHER_URL}.", file=sys.stderr)
        return 1
    try:
        compile(new, me.name, "exec")
    except SyntaxError:
        print(f"ERROR: {LAUNCHER_URL} is not a valid launcher, {me} is unchanged.", file=sys.stderr)
        return 1
    if new == me.read_bytes():
        print(f"INFO: {me} is up to date.", file=sys.stderr)
        return 0
    partial = me.with_name(f".{me.name}.{os.getpid()}.new")
    partial.write_bytes(new)
    os.chmod(partial, me.stat().st_mode)
    os.replace(partial, me)
    print(f"INFO: Updated {me} from {LAUNCHER_URL}.", file=sys.stderr)
    return 0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fastsurfer_container.py",
        description=__doc__.split("\n\n")[0].strip(),
        epilog=(
            "All other arguments are passed to run_fastsurfer.sh (or brun_fastsurfer.sh), see "
            "https://deep-mi.org/FastSurfer/stable/scripts/RUN_FASTSURFER.html. Relative paths are "
            "relative to the current directory."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--image",
                        help="the image to run: a release tag such as cu132-v2.6.0 (to keep a study on one "
                             "image), a version such as 2.6.0 (the image type is chosen for this machine), a "
                             "Docker image (e.g. my_fastsurfer:cu132) or an Apptainer .sif file. Default: the "
                             "release image installed on this machine, or the newest release if there is none")
    parser.add_argument("--update", action="store_true",
                        help="download the newest release (or the one --image names) again, even if an image "
                             "is installed, and run it")
    parser.add_argument("--update_launcher", action="store_true",
                        help="replace this launcher with the current one and stop")
    parser.add_argument("--runtime", choices=["auto", "apptainer", "singularity", "docker"], default="auto",
                        help="the container runtime (default: auto, the one that has the image, else Apptainer, "
                             "Singularity, then Docker)")
    parser.add_argument("--image_dir", type=Path,
                        default=Path(os.environ.get("FASTSURFER_IMAGE_DIR", "~/my_singularity_images")).expanduser(),
                        help="where Apptainer/Singularity images are kept (default: $FASTSURFER_IMAGE_DIR or "
                             "~/my_singularity_images)")
    parser.add_argument("--brun", action="store_true", help="run brun_fastsurfer.sh to process several subjects")
    parser.add_argument("--runtime_args", default="",
                        help="additional arguments for the container runtime, e.g. --runtime_args '--group-add 1001'")
    parser.add_argument("--print_only", action="store_true", help="print the commands instead of running them")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = make_parser()
    options, fastsurfer_args = parser.parse_known_args(argv)
    if options.update_launcher:
        return update_launcher()
    if not fastsurfer_args:
        parser.print_help()
        return 1

    def info(message: str, level: str = "INFO") -> None:
        print(f"{level}: {message}", file=sys.stderr)

    try:
        if os.name != "posix":
            raise LauncherError("Run this in WSL on Windows, where Docker's paths are Linux paths.")
        # the images are built for Intel/AMD processors, which a Mac with Apple silicon only emulates
        apple_silicon = is_apple_silicon()
        package_hint = (f"install the FastSurfer package for macOS instead, {MACOS_PACKAGE_URL} (see {MACOS_DOC_URL}). "
                        f"It runs natively and uses the Mac's GPU: in the console it opens, call run_fastsurfer.sh "
                        f"with the same flags.")
        if apple_silicon and options.runtime != "docker":
            raise LauncherError(f"On a Mac with Apple silicon, {package_hint} To run the Intel image in Docker "
                                f"under emulation anyway, which is much slower, pass --runtime docker.")
        if apple_silicon:
            info(f"Docker on a Mac with Apple silicon emulates the Intel image, which is much slower than the "
                 f"macOS package and cannot use the GPU. Better {package_hint}", level="WARNING")
        installed = {name for name in ("apptainer", "singularity", "docker") if shutil.which(name)}
        if not installed:
            raise LauncherError(f"Neither Apptainer, Singularity nor Docker is installed, see {DOC_URL}.")
        if options.runtime != "auto" and options.runtime not in installed:
            raise LauncherError(f"{options.runtime} is not installed.")
        if options.brun and not {"--subjects", "--subjects_list", "--subject_list"} & set(fastsurfer_args):
            # brun_fastsurfer.sh would read the subjects from stdin, which the container does not get
            raise LauncherError("--brun needs the subjects in --subjects_list <file> or --subjects <id>=<t1> ...")

        cwd = Path.cwd()
        fastsurfer_args = fastsurfer_args + _license_args(fastsurfer_args, cwd)
        args, mounts = plan_paths(fastsurfer_args, cwd, create=not options.print_only)
        mounts = minimal_mounts(mounts)
        if Path("/") in mounts:
            raise LauncherError("The paths would share the whole file system with the container, keep the "
                                "data in a directory below /.")
        if Path.home() in mounts:
            info("This shares your home directory with the container. Settings in it may affect "
                 "FastSurfer, so keep the data in a subdirectory.")
        if "--fs_license" not in args and "--seg_only" not in args:
            info("No FreeSurfer license given (--fs_license or $FS_LICENSE); the surfaces need one.")

        # the GPUs and the image type
        kind, value = parse_image_spec(options.image)
        gpus = query_nvidia_gpus()
        # whether the newest images support none of the GPUs, so they run on the CPU here
        gpu_unsupported = bool(gpus) and select_gpus(gpus, False) is None
        wants_gpu = gpu_request(args, options.brun)[0] != "cpu" and bool(gpus)
        if kind in ("auto", "version"):
            build = plan_build(args, gpus, options.brun)
        elif kind in ("tag", "release"):
            prefix = RELEASE_TAG.fullmatch(value).group(1)
            build = Build(prefix, prefix != "cpu" and wants_gpu)
        else:
            build = Build("custom", wants_gpu)

        # the image: one request to Docker Hub decides about versions
        hub = None if kind in ("sif", "docker") else hub_tags()
        if hub is not None and kind in ("auto", "version") and build.prefix != "cpu":
            newest_release = newest_version(list(hub), "cpu")
            if newest_release and f"{build.prefix}-v{newest_release}" not in hub:
                raise LauncherError(f"FastSurfer {newest_release} has no {build.prefix} image, which means this "
                                    f"launcher does not know the images of the newest release. Update it with "
                                    f"--update_launcher, or download it again from {LAUNCHER_URL}.")
        local_id = local_docker_id(value) if kind == "docker" else None
        # the GPUs this run uses, to tell which installed images run here; none for a run on the CPU
        used = [g for g in gpus if build.devices is None or g.index in build.devices] if build.gpu else []
        plan = decide_image(kind, value, build.prefix, options.runtime, installed,
                            local_docker_images() if "docker" in installed else None,
                            local_sif_images(options.image_dir), hub, options.image_dir, options.update, local_id,
                            used)
        runtime = plan.runtime
        if hub is not None:
            stable = stable_launcher()
            if stable is not None and stable != Path(__file__).resolve().read_bytes():
                info("A newer launcher is available, update it with --update_launcher.")

        if runtime != "docker":
            try:
                version_text = subprocess.run([runtime, "--version"], capture_output=True, text=True,
                                              timeout=30).stdout
            except (OSError, subprocess.SubprocessError):
                version_text = ""
            problem = runtime_too_old(version_text)
            if problem is not None:
                raise LauncherError(problem)

        requested = value if kind == "version" else (plan.tag if kind in ("tag", "release") else None)
        if requested is not None and hub is not None:
            version = requested.rsplit("-v", 1)[-1]
            note = age_note(version, release_versions(list(hub), "cpu"), gpu_unsupported and build.gpu)
            if note is not None:
                info(note[1], level=note[0])
        for level, note in plan.notes:
            info(note, level=level)
        for note in build.notes:
            info(note)

        script = "brun_fastsurfer.sh" if options.brun else "run_fastsurfer.sh"
        extra = shlex.split(options.runtime_args)
        commands = []
        if runtime == "docker":
            user = f"{os.getuid()}:{os.getgid()}"
            # official images are built for Intel/AMD only; an own image may be built for this Mac
            emulate = apple_silicon and (kind != "docker" or value.startswith(DOCKER_REPO))
            if plan.pull:
                commands.append(["docker", "pull"] + (["--platform", "linux/amd64"] if emulate else []) + [plan.image])
            commands.append(docker_command(plan.image, build.gpu, mounts, script, args, extra, user, emulate,
                                           build.devices))
            if not options.print_only:
                memory = docker_memory_gib()
                if memory < MIN_DOCKER_MEMORY_GIB:
                    where = ("in Docker's settings (Docker Desktop: Settings > Resources; Colima: colima start "
                             "--memory 16)") if platform.system() == "Darwin" else "for Docker"
                    info(f"Docker can use only {memory:.1f} GB of memory, and FastSurfer may fail with less than "
                         f"15 GB. Increase the memory {where}.", level="WARNING")
        else:
            if kind == "sif" and not Path(plan.image).is_file():
                raise LauncherError(f"--image {value}: no such file.")
            if plan.build_from is not None:
                commands.append([runtime, "build", plan.image, plan.build_from])
            commands.append(apptainer_command(runtime, Path(plan.image), build.gpu, mounts, script, args, extra,
                                              build.devices))
    except LauncherError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    if options.print_only:
        for command in commands:
            print(shlex.join(command))
        return 0
    for command in commands[:-1]:
        if command[1] == "build":
            info(f"Building the image once: {shlex.join(command)}")
            if not build_image(command):
                print("ERROR: Building the image failed. If it reports 'unauthorized' for this public image, "
                      "Apptainer sent stored Docker Hub credentials that are wrong or expired: check "
                      "~/.docker/config.json, `apptainer remote list` and $APPTAINER_DOCKER_USERNAME.",
                      file=sys.stderr)
                return 1
            record = Path(command[2]).with_name(Path(command[2]).name + ".digest")
            if plan.record:
                record.write_text(plan.record + "\n")
            elif record.exists():
                record.unlink()
        elif subprocess.run(command).returncode != 0:
            print(f"ERROR: {shlex.join(command)} failed.", file=sys.stderr)
            return 1
    info(f"Running: {shlex.join(commands[-1])}")
    sys.stdout.flush()
    sys.stderr.flush()
    os.execvp(commands[-1][0], commands[-1])
    return 0  # not reached


if __name__ == "__main__":
    raise SystemExit(main())
