#!/bin/python

# Copyright 2022 Image Analysis Lab, German Center for Neurodegenerative Diseases(DZNE), Bonn
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

# Author: David Kuegler
# June 27th 2023

import argparse
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Tuple, Literal, Sequence, Optional, Dict, overload, Generator, get_args, cast, Union

Target = Literal['runtime', 'build_common', 'build_cpu', 'build_cuda', 'build_freesurfer']
CacheType = Literal["inline", "registry", "local", "gha", "s3", "azblob"]
DeviceType = Literal["cpu", "cuda", "amd"]


class DEFAULTS:
    BUILD_BASE_IMAGE: Dict[DeviceType, str] = defaultdict(lambda: "ubuntu:20.04")
    CONDA_BUILD_IMAGE: Dict[DeviceType, str] = defaultdict(
        lambda: "build_common",
        cpu="build_cpu",
        cuda="build_cuda",
    )
    FREESURFER_BUILD_IMAGE: Dict[DeviceType, str] = defaultdict(lambda: "build_freesurfer")
    RUNTIME_BASE_IMAGE: Dict[DeviceType, str] = defaultdict(
        lambda: "ubuntu:20.04",
        amd="rocm/pytorch",
    )


def docker_image(arg) -> str:
    """Returns a str with the image or raises an error if it is not a valid docker image."""
    from re import match
    # regex from https://stackoverflow.com/questions/39671641/regex-to-parse-docker-tag
    docker_image_regex = r"^(?:(?=[^:\/]{1,253})(?!-)[a-zA-Z0-9-]{1,63}(?<!-)(?:\.(?!-)[a-zA-Z0-9-]{1,63}(?<!-))*(" \
                         r"?::[0-9]{1,5})?/)?((?![._-])(?:[a-z0-9._-]*)(?<![._-])(?:/(?![._-])[a-z0-9._-]*(?<![" \
                         r"._-]))*)(?::(?![.-])[a-zA-Z0-9_.-]{1,128})?$"
    if match(docker_image_regex, arg):
        return arg
    else:
        from argparse import ArgumentTypeError
        raise ArgumentTypeError(f"The image '{arg}' does not look like a valid image name.")


def target(arg) -> Target:
    """Returns a tuple of the target (see `Target`) and a str with the docker image, raises an error if not valid."""
    if not isinstance(arg, str):
        raise TypeError("target is not a string.")
    if arg not in get_args(Target):
        from argparse import ArgumentTypeError
        raise ArgumentTypeError(f"target must be one of {', '.join(get_args(Target))}, but was {arg}.")
    return cast(Target, arg)


class CacheSpec:
    """Class to parse the cache specification."""

    _type: CacheType
    _params: dict

    _azblob = ["account_url", "name", "prefix"]
    _ignore_error = ["ignore-error"]
    _io_params = ["compression", "compression-level", "force-compression", "image-manifest", "oci-mediatypes"]
    _gha = ["scope"]
    _local = ["tag"]
    _prefix = ["blobs_prefix", "manifests_prefix"]
    _registry = ["ref"]
    _s3 = ["bucket", "region", "name", "prefix"]
    CACHE_PARAMETERS: Dict[CacheType, Tuple[Sequence[str], Sequence[str]]] = {
        "inline": ([], []),
        "registry": (_registry + _io_params + _ignore_error, _registry),
        "local": (["dest"] + _local + _io_params + _ignore_error, ["src", "digest"] + _local),
        "gha": (_gha + _ignore_error, _gha),
        "s3": (_s3 + _ignore_error, _s3 + _prefix),
        "azblob": (_azblob + _ignore_error, _azblob + _prefix),
    }

    def __init__(self, arg: str):
        self._params = {}
        for key_value_pair in arg.split(","):
            k, v = key_value_pair.split("=", 2)
            if k.strip().lower() == "type":
                self.type = v.strip()
            else:
                self._params[k.strip().lower()] = v.strip()
        if not hasattr(self, "_type"):
            raise ValueError("No type defined in arg.")

    @property
    def type(self) -> CacheType:
        return self._type

    @type.setter
    def type(self, _type: str):
        from typing import get_args, cast
        if _type.lower() in get_args(CacheType):
            self._type = cast(CacheType, _type)
        else:
            raise ValueError(f"{_type} is not a valid cache type of {', '.join(get_args(CacheType))}")

    def _to_str(self, num: Literal[0, 1], **kwargs) -> str:
        params = {k: v for k, v in self._params.items() if k in self.CACHE_PARAMETERS[self.type][num]}
        params["type"] = self.type
        params.update(kwargs)
        return ",".join("=".join(i) for i in params.items())

    def cache_from_str(self) -> str:
        return self._to_str(1)

    def cache_to_str(self) -> str:
        return self._to_str(0, mode=self._params.get("mode", "min"))

    __repr__ = cache_from_str


def make_parser() -> argparse.ArgumentParser:
    from FastSurferCNN.segstats import HelpFormatter
    parser = argparse.ArgumentParser(
        description="Docker build script for FastSurfer",
        formatter_class=HelpFormatter,
    )

    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "amd"],
        required=True,
        help="""selection of internal build stages to build for a specific platform.<br>
                - cuda: build an internal cuda conda image (== --conda_build_image build_conda) (default)<br>
                - cpu: build an internal cpu conda image (== --conda_build_image build_cpu)<br>
                - amd: shortcut for --runtime_base_image rocm/pytorch --conda_build_image build_common (experimental)""",
    )
    parser.add_argument(
        "--tag",
        type=docker_image,
        metavar="image[:tag]",
        help="""tag build stage/target as <image>[:<tag>]""")
    parser.add_argument(
        "--target",
        default="runtime",
        type=target,
        choices=get_args(Target),
        metavar="target",
        help=f"""target to build (from list of targets below, defaults to {target}):<br>
                 - conda_build_image: "finished" conda build image<br>
                 - freesurfer_build_image: "finished" freesurfer build image<br>
                 - runtime: final fastsurfer runtime image""")
    parser.add_argument(
        "--rm",
        action="store_true",
        help="disables caching, i.e. removes all intermediate images.")
    parser.add_argument(
        "--cache",
        type=CacheSpec,
        help="""cache as defined in https://docs.docker.com/build/cache/backends/ 
                (using --cache-to syntax, parameters are automatically filtered for use in 
                --cache-to and --cache-from), e.g.: --cache type=registry,ref=server/fastbuild,mode=max.""")
    parser.add_argument(
        '--print',
        action="store_true",
        help="Instead of starting processes, write the commands to stdout, so they can be run with "
             "%(prog) ... --print | bash.")
    expert = parser.add_argument_group('Expert options')

    expert.add_argument(
        "--freesurfer_build_image",
        type=docker_image,
        metavar="image[:tag]",
        help="""explicitly specifies an image to copy freesurfer binaries from.
                freesurfer binaries are expected to be in /opt/freesurfer in the image, like the runtime image. By default,
                uses the "build_freesurfer" stage in the Dockerfile (either by building it or from cache, see --cache).""")
    expert.add_argument(
        "--conda_build_image",
        type=docker_image,
        metavar="image[:tag]",
        help="""explicitly specifies an image to copy the python environment from.
                The environment is expected to be in /venv in the image, like the runtime image. By default, uses the
                "build_conda" stage in the Dockerfile (either by building it or from cache, see --cache).""")
    expert.add_argument(
        "--runtime_base_image",
        type=docker_image,
        metavar="image[:tag]",
        help="explicitly specifies the base image to build the runtime image from (default: ubuntu:20.04).")
    expert.add_argument(
        "--build_base_image",
        type=docker_image,
        metavar="image[:tag]",
        help="explicitly specifies the base image to build the build images from (default: ubuntu:20.04).")
    return parser


@overload
def command(
        args: Sequence[str],
        working_directory: Optional[Path] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        poll_interval: float = 1.,
        redirect_stdout: Literal["cache"] = "cache",
        redirect_stderr: Union[Literal["stdout", "cache"], Path] = "cache",
    ) -> Tuple[bytes, bytes, int]:
    ...


@overload
def command(
        args: Sequence[str],
        working_directory: Optional[Path] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        poll_interval: float = 1.,
        redirect_stdout: Union[Literal["stdout", "cache"], Path] = "cache",
        redirect_stderr: Literal["cache"] = "cache",
    ) -> Tuple[bytes, bytes, int]:
    ...


@overload
def command(
        args: Sequence[str],
        working_directory: Optional[Path] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        poll_interval: float = 1.,
        redirect_stdout: Union[Literal["stdout"], Path] = None,
        redirect_stderr: Union[Literal["stdout"], Path] = None,
    ) -> int:
    ...


def command(
        args: Sequence[str],
        working_directory: Optional[Path] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        poll_interval: float = 1.,
        redirect_stdout: Union[Literal["stdout", "cache"], Path] = "cache",
        redirect_stderr: Union[Literal["stdout", "cache"], Path] = "cache",
    ):
    stdout = b''
    stderr = b''
    command_iter_gen = command_iter(args,
                                    working_directory=working_directory,
                                    env=env,
                                    timeout=timeout,
                                    poll_interval=poll_interval,
                                    redirect_stdout="stream" if redirect_stdout == "cache" else redirect_stdout,
                                    redirect_stderr="stream" if redirect_stderr == "cache" else redirect_stderr)
    for out, err in command_iter_gen:
        stdout += out
        stderr += err
    return stdout, stderr, command_iter_gen if "cache" in [redirect_stderr, redirect_stdout] else next(command_iter_gen)


def command_iter(
        args: Sequence[str],
        working_directory: Optional[Path] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        poll_interval: float = 1.,
        redirect_stdout: Union[Literal["stdout", "stream"], Path] = "stream",
        redirect_stderr: Union[Literal["stdout", "stream"], Path] = "stream",
    ) -> Generator[None, Tuple[bytes, bytes], int]:

    from subprocess import Popen, TimeoutExpired, STDOUT, PIPE, run
    redirect_targets = {"stdout": STDOUT, "stream": PIPE}
    kwargs = {}
    if working_directory is not None:
        kwargs["cwd"] = str(working_directory)
    if env is not None:
        kwargs["env"] = env

    before_call = time.perf_counter()

    with Popen(args,
               stdout=redirect_targets.get(redirect_stdout, redirect_stdout),
               stderr=redirect_targets.get(redirect_stderr, redirect_stderr),
               **kwargs) as proc:
        while proc.poll() is None:
            try:
                delta = time.perf_counter() - before_call
                if timeout is None:
                    _timeout = poll_interval
                elif delta > timeout:
                    proc.terminate()
                    raise TimeoutExpired
                else:
                    _timeout = delta - timeout
                result = proc.communicate(timeout=_timeout)
                yield tuple(b'' if _ is None else _ for _ in result)
            except TimeoutExpired:
                if timeout is not None and time.perf_counter() - before_call > timeout:
                    raise
        _stdout = b'' if proc.stdout is None or proc.stdout.closed else proc.stdout.read()
        _stderr = b'' if proc.stderr is None or proc.stderr.closed else proc.stderr.read()
        if _stderr != b'' or _stdout != b'':
            yield _stdout, _stderr
        return proc.returncode


def red(skk):
    return "\033[91m {}\033[00m" .format(skk)


def docker_build_image(
        image_name: str,
        dockerfile: Path,
        working_directory: Optional[Path] = None,
        context: Path = ".",
        run: bool = True,
        **kwargs):
    if run:
        print("Building. This starts with sending the build context to the docker daemon, which may take a while...")
    env = dict(os.environ)
    env["DOCKER_BUILDKIT"] = "1"
    from itertools import chain

    def to_pair(key, values):
        _values = values if isinstance(values, Sequence) and not isinstance(values, (str, bytes)) else [values]
        key_dashed = key.replace("_", "-")
        return list(chain(*[[f"--{key_dashed}"] + ([] if val is None else [val]) for val in _values]))

    params = [to_pair(*a) for a in kwargs.items()]

    args = ["build", "-t", image_name, "-f", str(dockerfile)] + list(chain(*params)) + [str(context)]
    if run:
        from shutil import which
        docker_cmd = which("docker")
        if docker_cmd is None:
            raise FileNotFoundError("Could not locate the docker executable")
        for out, err in command_iter([docker_cmd] + args,
                                     working_directory=working_directory, env=env, redirect_stdout="stream", redirect_stderr="stdout"):
            if out:
                print("stdout: " + out.decode("utf-8"))
            if err:
                print("stderr: " + red(err.decode("utf-8")))
    else:
        print(" ".join(["docker"] + args), end="")


def main(args: object):
    kwargs = {}
    if getattr(args, "cache") is not None:
        kwargs["cache_from"] = getattr(args, "cache").cache_from_str()
        kwargs["cache_to"] = getattr(args, "cache").cache_from_str()
    elif getattr(args, "rm", False):
        kwargs["no-cache"] = None

    target = getattr(args, "target", "runtime")
    if target not in get_args(Target):
        raise ValueError(f"Invalid target: {target}")
    kwargs["target"] = target
    device = getattr(args, "device")
    if device not in get_args(DeviceType):
        raise ValueError(f"Invalid device: {device}")
    kwargs["build_arg"] = []
    for key in ["build_base_image", "runtime_base_image", "freesurfer_build_image", "conda_build_image"]:
        upper_key = key.upper()
        value = getattr(args, key) or getattr(DEFAULTS, upper_key)[device]
        kwargs["build_arg"].append(f"{upper_key}={value}")
    env = dict(os.environ)
    fastsurfer_home = Path(__file__).parent.parent
    env["FASTSURFER_HOME"] = str(fastsurfer_home)
    build_info, _, _ = command(
        ["bash", str(fastsurfer_home / "run_fastsurfer.sh"), "--version", "long"],
        env=env)
    build_info = build_info.decode("utf-8")
    # stdout of run_fastsurfer.sh --version, now just get the version number not the git hash
    version_tag = build_info.split(" ", 1)[0]
    image_tag = getattr(args, "tag") or f"fastsurfer:{version_tag}"
    run = not getattr(args, "print", False)

    build_filename = fastsurfer_home / "BUILD.txt"
    with open(build_filename, "w") as build_file:
        build_file.write(build_info)
    if run:
        print("Version info added to the docker image:")
        print(build_info)

    dockerfile = fastsurfer_home / "Docker" / "Dockerfile"
    docker_build_image(
        image_tag,
        dockerfile,
        working_directory=fastsurfer_home,
        context=fastsurfer_home,
        run=run,
        **kwargs
    )
    if run:
        os.remove(build_filename)
    else:
        print(f" || rm \"{fastsurfer_home / 'BUILD.txt'}\"")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main(make_parser().parse_args()))
