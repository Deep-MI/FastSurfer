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
Explain why a GPU cannot be used, and what to use instead.

PyTorch either reports cuda as unavailable or fails at the first kernel with "no kernel image
is available for execution on the device". Neither tells the user that a different FastSurfer
image or PyTorch build would run on their GPU. This names the cause: the GPU architecture is
not compiled into this PyTorch build, the driver is too old for its CUDA version, the build has
no CUDA at all, or the container was started without access to the GPU.

Run as a script, so run_fastsurfer.sh can check the device once per run:

    python3 FastSurferCNN/gpu_support.py --device auto

Exit codes: 0 when the device can be used, 3 when "auto" found a GPU it cannot use and falls
back to the cpu, 4 when "auto" falls back to the cpu for a reason that is only worth a note, 5
when a requested cuda device cannot be used. None of them is 1 or 2, which a crash or an
argument error returns.

Imports torch only inside the functions that need it, so the rest can be tested without it.
"""

import argparse
import os
import re
import warnings
from pathlib import Path
from typing import Literal

__all__ = ["cuda_problem", "supports_capability"]

# The CUDA builds of FastSurfer, as (CUDA version, image tag prefix). Newer CUDA versions drop
# old GPU architectures and older ones lack the newest, so these are the two to point users to.
LEGACY_BUILD = ((12, 6), "cu126")
NEWEST_BUILD = ((13, 2), "cu132")

# the oldest driver each CUDA major version runs on, per NVIDIA's minor version compatibility
MIN_DRIVER = {11: 450, 12: 525, 13: 580}

NVIDIA_VERSION_FILE = Path("/proc/driver/nvidia/version")

Severity = Literal["warning", "note"]


def supports_capability(capability: tuple[int, int], arch_list: list[str]) -> bool:
    """
    Whether a PyTorch build compiled for these architectures runs on a GPU of this capability.

    A cubin (`sm_XY`) runs on the same major architecture at an equal or higher minor version, so
    sm_120 also covers sm_121. PTX (`compute_XY`) is compiled by the driver for any GPU at or above
    it. Architecture specific variants such as `sm_90a` run on exactly that architecture.

    Parameters
    ----------
    capability : tuple[int, int]
        The compute capability of the GPU, as returned by `torch.cuda.get_device_capability`.
    arch_list : list[str]
        The architectures of the build, as returned by `torch.cuda.get_arch_list`.

    Returns
    -------
    bool
        True if any entry of `arch_list` runs on the GPU.
    """
    for arch in arch_list:
        match = re.fullmatch(r"(sm|compute)_(\d+)(\d)([a-z]?)", arch)
        if match is None:
            continue
        kind, variant = match.group(1), match.group(4)
        major, minor = int(match.group(2)), int(match.group(3))
        if variant:
            if (major, minor) == capability:
                return True
        elif kind == "sm":
            if major == capability[0] and minor <= capability[1]:
                return True
        elif (major, minor) <= capability:
            return True
    return False


def _oldest_arch(arch_list: list[str]) -> tuple[int, int] | None:
    found = [re.fullmatch(r"(?:sm|compute)_(\d+)(\d)[a-z]?", arch) for arch in arch_list]
    versions = [(int(m.group(1)), int(m.group(2))) for m in found if m is not None]
    return min(versions) if versions else None


def in_container() -> bool:
    """Whether this runs in a Docker, Podman, Apptainer or Singularity container."""
    return (
        Path("/.dockerenv").exists()
        or Path("/run/.containerenv").exists()
        or "APPTAINER_CONTAINER" in os.environ
        or "SINGULARITY_CONTAINER" in os.environ
    )


def nvidia_gpu_present() -> bool:
    """Whether an NVIDIA driver is visible, whether or not PyTorch can use it."""
    # the device node and driver version on linux, the driver library WSL maps in from windows
    paths = (Path("/dev/nvidiactl"), NVIDIA_VERSION_FILE, Path("/usr/lib/wsl/lib/libcuda.so"))
    return any(path.exists() for path in paths)


def driver_major() -> int | None:
    """The major version of the NVIDIA kernel driver, or None if it cannot be read (e.g. WSL)."""
    try:
        text = NVIDIA_VERSION_FILE.read_text()
    except OSError:
        return None
    match = re.search(r"\s(\d{3,})\.\d+(?:\.\d+)?\s", text)
    return int(match.group(1)) if match else None


def _fastsurfer_version() -> str:
    """The version of this FastSurfer as in the image tags, e.g. v2.6.0 (dev builds keep their suffix)."""
    try:
        from FastSurferCNN.version import read_and_close_version

        version = read_and_close_version()
    except (ImportError, OSError):
        return "v<VERSION>"
    return "v<VERSION>" if version == "unspecified" else f"v{version}"


def _use_build(build: tuple[tuple[int, int], str], lead: str = "Use") -> str:
    (major, minor), tag = build
    if in_container():
        return f"{lead} the FastSurfer image for CUDA {major}.{minor}, deepmi/fastsurfer:{tag}-{_fastsurfer_version()}."
    return f"{lead} PyTorch built for CUDA {major}.{minor}, from https://download.pytorch.org/whl/{tag}."


def _arch_problem(name: str, capability: tuple[int, int], arch_list: list[str], build: tuple[int, int]) -> list[str]:
    cuda = f"{build[0]}.{build[1]}"
    lines = [
        f"The GPU {name} has compute capability {capability[0]}.{capability[1]}, which this PyTorch "
        f"build for CUDA {cuda} does not support (compiled for {', '.join(arch_list)}).",
    ]
    oldest = _oldest_arch(arch_list)
    if oldest is not None and capability < oldest:
        if build > LEGACY_BUILD[0]:
            lines.append(_use_build(LEGACY_BUILD))
        else:
            lines.append("No FastSurfer build supports this GPU any more.")
    elif build < NEWEST_BUILD[0]:
        lines.append(_use_build(NEWEST_BUILD))
    else:
        lines.append("No FastSurfer build supports this GPU yet, check for a newer FastSurfer release.")
    return lines


def cuda_problem(device_index: int | None = None) -> tuple[Severity, list[str]] | None:
    """
    Find out why cuda cannot be used on this machine, if it cannot.

    Parameters
    ----------
    device_index : int, optional
        The cuda device to check, the current device by default.

    Returns
    -------
    tuple[Severity, list[str]], None
        None if cuda works, or if there is no NVIDIA GPU to use in the first place. Otherwise
        the severity and the explanation, one sentence per line: "warning" when a GPU is present
        but unusable, "note" when a container may just have been started without the GPU.
    """
    import torch

    if torch.version.hip is not None:
        return None  # ROCm answers through the same api, but none of the checks below apply
    if torch.version.cuda is None:
        if not nvidia_gpu_present():
            return None
        lines = ["This PyTorch build has no CUDA support, but an NVIDIA GPU is present."]
        if in_container():
            lines.append("Use a FastSurfer GPU image, deepmi/fastsurfer:latest.")
        else:
            lines.append(_use_build(NEWEST_BUILD, lead="Install"))
        return "warning", lines

    build = tuple(int(v) for v in torch.version.cuda.split(".")[:2])
    # the driver check only warns, and only on the first call in a process
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        available = torch.cuda.is_available()

    if available:
        count = torch.cuda.device_count()
        index = torch.cuda.current_device() if device_index is None else device_index
        if index >= count:
            return "warning", [f"There is no cuda device {index}, found {count} GPU(s)."]
        capability = torch.cuda.get_device_capability(index)
        arch_list = torch.cuda.get_arch_list()
        if supports_capability(capability, arch_list):
            return None
        return "warning", _arch_problem(torch.cuda.get_device_name(index), capability, arch_list, build)

    driver, needed = driver_major(), MIN_DRIVER.get(build[0])
    if driver is not None and needed is not None and driver < needed:
        lines = [
            f"The NVIDIA driver {driver} is too old for this PyTorch build for CUDA {build[0]}.{build[1]}, "
            f"which needs driver {needed} or newer.",
            f"Update the NVIDIA driver to {needed} or newer.",
        ]
        legacy_driver = MIN_DRIVER[LEGACY_BUILD[0][0]]
        if build > LEGACY_BUILD[0] and driver >= legacy_driver:
            lines.append(_use_build(LEGACY_BUILD, lead="Or use"))
        return "warning", lines
    if nvidia_gpu_present():
        reason = next((str(w.message).splitlines()[0] for w in caught if "CUDA" in str(w.message)), None)
        return "warning", [
            "An NVIDIA GPU is present, but PyTorch cannot use it" + (f": {reason}" if reason else "."),
            "Check that the NVIDIA driver works, e.g. with nvidia-smi.",
        ]
    if in_container():
        return "note", [
            "No GPU is visible in this container. If the host has an NVIDIA GPU, start the container "
            "with --gpus all (Docker) or --nv (Apptainer/Singularity).",
        ]
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check that the device FastSurfer is asked to run on can be used, and explain why not.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="the device to check, as passed to --device: auto, cuda or cuda:<n>; others are not checked",
    )
    parser.add_argument(
        "--flag_name",
        default="device",
        help="the flag the device was passed with, for the messages",
    )
    args = parser.parse_args()
    device = args.device
    if device != "auto" and not device.startswith("cuda"):
        return 0
    index = int(device.split(":", 1)[1]) if device.startswith("cuda:") else None
    problem = cuda_problem(index)
    if problem is None:
        return 0
    severity, lines = problem
    if device != "auto":
        print(f"ERROR: {lines[0]}")
        for line in lines[1:] + [f"Or run on the cpu with --{args.flag_name} cpu."]:
            print(f"  {line}")
        return 5
    if severity == "note":
        print(f"INFO: {lines[0]}")
        for line in lines[1:]:
            print(f"  {line}")
        return 4
    rule = "=" * 80
    print(rule)
    print(f"WARNING: {lines[0]}")
    for line in lines[1:]:
        print(f"  {line}")
    print("  FastSurfer continues on the cpu, which takes much longer than on a supported GPU.")
    print(rule)
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
