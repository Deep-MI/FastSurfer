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

It runs the newest FastSurfer release unless --image_version names another, picks the image that
fits the NVIDIA GPU and driver (or the CPU image), downloads it once,
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
MACOS_PACKAGE_URL = "https://github.com/Deep-MI/FastSurfer/releases/latest/download/FastSurfer-macos-darwin_arm64.pkg"
MACOS_DOC_URL = "https://deep-mi.org/FastSurfer/stable/overview/install/MACOS.html"
# the 15 GB the macOS page asks Docker to have, less what the kernel of Docker's VM keeps for itself
MIN_DOCKER_MEMORY_GIB = 14

# The CUDA images, as in FastSurferCNN/gpu_support.py: (CUDA version, tag prefix), the oldest GPU
# each supports and the newest the legacy one does. The test in test/lint keeps them in sync.
NEWEST_BUILD = ((13, 2), "cu132")
LEGACY_BUILD = ((12, 6), "cu126")
MIN_DRIVER = {12: 525, 13: 580}
MIN_CAPABILITY = {"cu132": (7, 5), "cu126": (5, 0)}
LEGACY_MAX_CAPABILITY = (9, 0)

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
class Build:
    """The image build to run, and whether the container gets the GPU."""

    prefix: str
    gpu: bool
    notes: list[str] = field(default_factory=list)


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


def published_tags(name: str) -> list[str] | None:
    """The tags of the image on Docker Hub that contain `name`; None if Docker Hub cannot be reached."""
    import json
    import urllib.parse
    import urllib.request

    url = f"https://hub.docker.com/v2/repositories/{DOCKER_REPO}/tags?page_size=100&name={urllib.parse.quote(name)}"
    tags: list[str] = []
    try:
        while url and len(tags) < 1000:
            with urllib.request.urlopen(url, timeout=20) as response:
                page = json.load(response)
            tags += [result["name"] for result in page.get("results", [])]
            url = page.get("next")
    except (OSError, ValueError, KeyError):
        return None
    return tags


def official_image_version(image: str) -> str | None:
    """The FastSurfer version of an official image, e.g. 2.2.0 for deepmi/fastsurfer:cu118-v2.2.0, else None."""
    name = Path(image).name if image.endswith(".sif") else image
    pattern = (rf"{re.escape(DOCKER_REPO.rsplit('/', 1)[-1])}-[a-z0-9.]+-v(.+)\.sif" if image.endswith(".sif")
               else rf"(?:docker\.io/)?{re.escape(DOCKER_REPO)}:[a-z0-9.]+-v(.+)")
    match = re.fullmatch(pattern, name)
    return match.group(1) if match and parse_version(match.group(1)) else None


def age_note(version: str, releases: list[tuple[int, ...]], gpu_unsupported: bool,
             flag: str = "--image_version") -> tuple[str, str] | None:
    """
    A note for a run of an older release, as (level, message), None for the newest or an unknown version.

    Parameters
    ----------
    version : str
        The requested version.
    releases : list[tuple[int, ...]]
        All releases, oldest first.
    gpu_unsupported : bool
        Whether the newest images do not support this GPU, so an old image is the only way to use it.
    flag : str, default="--image_version"
        The flag that requested the version.
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
            f"Running the newest release on the CPU (leave out --image and --image_version), or on a machine "
            f"with a newer GPU, is recommended over this old version, which lacks the improvements and fixes "
            f"since."
        )
    return "INFO", f"{message} Leave out {flag} to run the newest release."


def local_tags(runtime: str, image_dir: Path) -> list[str]:
    """The tags of the FastSurfer images already on this machine, for runs without internet."""
    if runtime != "docker":
        prefix = DOCKER_REPO.rsplit("/", 1)[-1] + "-"
        return [p.name[len(prefix):-len(".sif")] for p in image_dir.glob(f"{prefix}*.sif")]
    try:
        out = subprocess.run(["docker", "image", "ls", DOCKER_REPO, "--format", "{{.Tag}}"],
                             capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return []
    return out.stdout.split() if out.returncode == 0 else []


def query_nvidia_gpu(index: int = 0) -> tuple[tuple[int, int] | None, int] | None:
    """
    The compute capability and the driver version of an NVIDIA GPU, from nvidia-smi.

    Returns None without an NVIDIA GPU. The capability is None if this driver cannot report it.
    """
    if shutil.which("nvidia-smi") is None:
        return None
    for fields in ("compute_cap,driver_version", "driver_version"):
        try:
            out = subprocess.run(
                ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader"],
                capture_output=True, text=True, check=True, timeout=30,
            ).stdout.strip().splitlines()
        except (OSError, subprocess.SubprocessError):
            continue
        if index >= len(out):
            return None
        values = [v.strip() for v in out[index].split(",")]
        driver = int(values[-1].split(".")[0])
        if len(values) == 1:
            return None, driver
        major, minor = values[0].split(".")
        return (int(major), int(minor)), driver
    return None


def choose_build(gpu: tuple[tuple[int, int] | None, int] | None, device: str) -> Build:
    """
    The image that runs on this GPU and driver, preferring the default CUDA version.

    Parameters
    ----------
    gpu : tuple, None
        The compute capability (None if unknown) and the driver major version, or None without a GPU.
    device : str
        The value of FastSurfer's --device, "auto" if it was not given.

    Returns
    -------
    Build
        The tag prefix, whether to give the container the GPU, and notes for the user.
    """
    if device == "cpu":
        return Build("cpu", False)
    if gpu is None:
        if device.startswith("cuda"):
            raise LauncherError(f"--device {device} needs an NVIDIA GPU, but nvidia-smi found none.")
        return Build("cpu", False, ["No NVIDIA GPU found, so FastSurfer runs on the CPU."])
    capability, driver = gpu

    def runs(build: tuple[tuple[int, int], str]) -> bool:
        (cuda_major, _), prefix = build
        if driver < MIN_DRIVER[cuda_major]:
            return False
        if capability is None:
            return True
        if prefix == LEGACY_BUILD[1] and capability > LEGACY_MAX_CAPABILITY:
            return False
        return capability >= MIN_CAPABILITY[prefix]

    if runs(NEWEST_BUILD):
        return Build(NEWEST_BUILD[1], True)
    newest_min_driver = MIN_DRIVER[NEWEST_BUILD[0][0]]
    if runs(LEGACY_BUILD):
        notes = []
        if driver < newest_min_driver and (capability is None or capability >= MIN_CAPABILITY[NEWEST_BUILD[1]]):
            notes.append(
                f"The NVIDIA driver {driver} is older than {newest_min_driver}, so this uses the image for "
                f"CUDA {LEGACY_BUILD[0][0]}.{LEGACY_BUILD[0][1]}. Update the driver to use the default image."
            )
        return Build(LEGACY_BUILD[1], True, notes)
    if capability is not None and capability < MIN_CAPABILITY[LEGACY_BUILD[1]]:
        reason = f"The GPU (compute capability {capability[0]}.{capability[1]}) is too old for FastSurfer's images"
    else:
        reason = (f"The NVIDIA driver {driver} is too old for FastSurfer's images, update it to "
                  f"{newest_min_driver} or newer")
    if device.startswith("cuda"):
        raise LauncherError(f"{reason}.")
    return Build("cpu", False, [f"{reason}, so FastSurfer runs on the CPU."])


def image_reference(prefix: str, version: str) -> str:
    """The Docker image to run, e.g. deepmi/fastsurfer:cu132-v2.6.0."""
    return f"{DOCKER_REPO}:{prefix}-v{version}"


def resolve_version(prefix: str, runtime: str, image_dir: Path) -> tuple[str, str]:
    """
    The newest released FastSurfer version for this image, and where that was found.

    Asks Docker Hub, and without internet looks at the images already on this machine.
    """
    version = newest_version(published_tags(f"{prefix}-v") or [], prefix)
    if version is not None:
        return version, "the newest release"
    version = newest_version(local_tags(runtime, image_dir), prefix)
    if version is not None:
        return version, "the newest image on this machine, Docker Hub could not be reached"
    raise LauncherError(
        "Could not find out the newest FastSurfer release, because Docker Hub cannot be reached and no "
        "FastSurfer image is on this machine yet. Pass the version with --image_version, e.g. 2.6.0."
    )


def _absolute(value: str, cwd: Path) -> Path:
    return Path(os.path.normpath(cwd / os.path.expanduser(value)))


def _parent_of_file(path: Path) -> Path:
    return path if path.is_dir() else path.parent


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
            if flag in INPUT_FLAGS:
                if not path.exists():
                    raise LauncherError(f"{flag} {args[i]}: no such file.")
                mounts.append(_parent_of_file(path))
                if flag in ("--subjects_list", "--subject_list"):
                    mounts.extend(_subjects_list_mounts(path))
            else:
                directory = path if flag == "--sd" else path.parent
                if create:
                    directory.mkdir(parents=True, exist_ok=True)
                mounts.append(directory)
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
                mounts.append(path.parent)
                out.append(f"{subject}={path}")
                i += 1
    return out, mounts


def _subjects_list_mounts(subjects_list: Path) -> list[Path]:
    """The directories of the files a brun_fastsurfer.sh subjects list names, which must be absolute."""
    mounts = []
    for number, line in enumerate(subjects_list.read_text().splitlines(), start=1):
        tokens = shlex.split(line, comments=True)
        if not tokens:
            continue
        _, _, t1 = tokens[0].partition("=")
        paths = [t1] + [tokens[k + 1] for k, tok in enumerate(tokens[1:-1], start=1) if tok in INPUT_FLAGS]
        for value in paths:
            if not os.path.isabs(os.path.expanduser(value)):
                raise LauncherError(
                    f"{subjects_list}, line {number}: {value} is a relative path. Use absolute paths in a "
                    f"subjects list, the container does not start in the current directory."
                )
            mounts.append(_parent_of_file(Path(os.path.expanduser(value))))
    return mounts


def minimal_mounts(paths: list[Path]) -> list[Path]:
    """The given directories without duplicates and without those inside another one."""
    kept: list[Path] = []
    for path in sorted(set(paths), key=lambda p: len(p.parts)):
        if not any(path == k or k in path.parents for k in kept):
            kept.append(path)
    return sorted(kept)


def find_runtime(requested: str) -> str:
    """The container runtime to use: Apptainer first, then Singularity, then Docker."""
    names = ["apptainer", "singularity", "docker"] if requested == "auto" else [requested]
    for name in names:
        if shutil.which(name):
            return name
    if requested == "auto":
        raise LauncherError(f"Neither Apptainer, Singularity nor Docker is installed, see {DOC_URL}.")
    raise LauncherError(f"{requested} is not installed.")


def apptainer_command(runtime: str, sif: Path, gpu: bool, mounts: list[Path], script: str,
                      args: list[str], extra: list[str]) -> list[str]:
    """The Apptainer/Singularity command; the home directory is not mounted, see the docs."""
    cmd = [runtime, "exec"] + (["--nv"] if gpu else []) + ["--no-mount", "home,cwd", "-e"]
    for mount in mounts:
        cmd += ["-B", str(mount)]
    return cmd + extra + [str(sif), f"/fastsurfer/{script}"] + args


def docker_command(image: str, gpu: bool, mounts: list[Path], script: str, args: list[str],
                   extra: list[str], user: str, emulate: bool = False) -> list[str]:
    """The Docker command, running as the calling user; `emulate` runs the Intel image on Apple silicon."""
    cmd = ["docker", "run", "--rm", "--user", user] + (["--gpus", "all"] if gpu else [])
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


def _fastsurfer_device(args: list[str]) -> str:
    for i in range(len(args) - 1):
        if args[i] == "--device":
            return args[i + 1]
    return "auto"


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
    parser.add_argument("--runtime", choices=["auto", "apptainer", "singularity", "docker"], default="auto",
                        help="the container runtime (default: auto, the first installed of these)")
    parser.add_argument("--image_version",
                        help="the FastSurfer version to run, e.g. 2.6.0 (default: the newest release)")
    parser.add_argument("--image",
                        help="run this image instead: a Docker image (e.g. my_fastsurfer:cu132) or an Apptainer "
                             ".sif file")
    parser.add_argument("--image_dir", type=Path,
                        default=Path(os.environ.get("FASTSURFER_IMAGE_DIR", "~/my_singularity_images")).expanduser(),
                        help="where Apptainer/Singularity images are stored (default: $FASTSURFER_IMAGE_DIR or "
                             "~/my_singularity_images)")
    parser.add_argument("--brun", action="store_true", help="run brun_fastsurfer.sh to process several subjects")
    parser.add_argument("--runtime_args", default="",
                        help="additional arguments for the container runtime, e.g. --runtime_args '--group-add 1001'")
    parser.add_argument("--print_only", action="store_true", help="print the commands instead of running them")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = make_parser()
    options, fastsurfer_args = parser.parse_known_args(argv)
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
        if apple_silicon and options.runtime == "auto":
            raise LauncherError(f"On a Mac with Apple silicon, {package_hint} To run the Intel image in Docker "
                                f"under emulation anyway, which is much slower, pass --runtime docker.")
        if apple_silicon:
            info(f"Docker on a Mac with Apple silicon emulates the Intel image, which is much slower than the "
                 f"macOS package and cannot use the GPU. Better {package_hint}", level="WARNING")
        cwd = Path.cwd()
        runtime = find_runtime(options.runtime)
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

        device = _fastsurfer_device(args)
        index = int(device.split(":", 1)[1]) if device.startswith("cuda:") else 0
        gpu = query_nvidia_gpu(index) if device != "cpu" else None
        # whether the newest images can use this GPU; a GPU they do not support runs them on the CPU
        gpu_unsupported = gpu is not None and not choose_build(gpu, "auto").gpu
        if options.image:
            # an own or an older official image: give it the GPU unless the run is on the CPU
            build = Build("custom", gpu is not None)
            image = options.image
            requested, flag = official_image_version(image), "--image"
        else:
            build = choose_build(gpu, device)
            requested, flag = options.image_version, "--image_version"
            version = requested
            if version is None:
                version, source = resolve_version(build.prefix, runtime, options.image_dir)
                info(f"Running FastSurfer {version}, {source}. Pass --image_version {version} to keep "
                     f"using this version when newer ones are released.")
            image = image_reference(build.prefix, version)
            if requested is not None:
                # an older version may have no image for this GPU, e.g. none for a newer CUDA version
                available = published_tags(f"-v{requested}")
                tag = image.rsplit(":", 1)[-1]
                if available is not None and tag not in available:
                    images = sorted(t for t in available if t.endswith(f"-v{requested}"))
                    raise LauncherError(
                        f"There is no image {image}. "
                        + (f"The images of FastSurfer {requested} are {', '.join(images)}; pass one with "
                           f"--image {DOCKER_REPO}:<tag>." if images else f"FastSurfer {requested} is not released.")
                    )
        if requested is not None:
            note = age_note(requested, release_versions(published_tags("cpu-v") or [], "cpu"),
                            gpu_unsupported and build.gpu, flag)
            if note is not None:
                info(note[1], level=note[0])
        for note in build.notes:
            info(note)

        script = "brun_fastsurfer.sh" if options.brun else "run_fastsurfer.sh"
        extra = shlex.split(options.runtime_args)
        commands = []
        if runtime == "docker":
            user = f"{os.getuid()}:{os.getgid()}"
            # official images are built for Intel/AMD only; an own image may be built for this Mac
            emulate = apple_silicon and (not options.image or official_image_version(options.image) is not None)
            commands.append(docker_command(image, build.gpu, mounts, script, args, extra, user, emulate))
            if not options.print_only:
                memory = docker_memory_gib()
                if memory < MIN_DOCKER_MEMORY_GIB:
                    where = ("in Docker's settings (Docker Desktop: Settings > Resources; Colima: colima start "
                             "--memory 16)") if platform.system() == "Darwin" else "for Docker"
                    info(f"Docker can use only {memory:.1f} GB of memory, and FastSurfer may fail with less than "
                         f"15 GB. Increase the memory {where}.", level="WARNING")
        else:
            if image.endswith(".sif"):
                sif = _absolute(image, cwd)
                if not sif.is_file():
                    raise LauncherError(f"--image {image}: no such file.")
            else:
                sif = options.image_dir / (image.rsplit("/", 1)[-1].replace(":", "-") + ".sif")
                if not sif.is_file():
                    commands.append([runtime, "build", str(sif), f"docker://{image}"])
            commands.append(apptainer_command(runtime, sif, build.gpu, mounts, script, args, extra))
    except LauncherError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    if options.print_only:
        for command in commands:
            print(shlex.join(command))
        return 0
    for command in commands[:-1]:
        info(f"Building the image once: {shlex.join(command)}")
        Path(command[2]).parent.mkdir(parents=True, exist_ok=True)
        if subprocess.run(command).returncode != 0:
            print("ERROR: Building the image failed.", file=sys.stderr)
            return 1
    info(f"Running: {shlex.join(commands[-1])}")
    sys.stdout.flush()
    sys.stderr.flush()
    os.execvp(commands[-1][0], commands[-1])
    return 0  # not reached


if __name__ == "__main__":
    raise SystemExit(main())
