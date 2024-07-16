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
import subprocess
from pathlib import Path
from typing import Tuple, Literal, Sequence, Optional, Dict, get_args, cast, List, Callable, Union
import logging


logger = logging.getLogger(__name__)

Target = Literal['runtime', 'build_common', 'build_conda', 'build_freesurfer',
                 'build_base', 'runtime_cuda']
CacheType = Literal["inline", "registry", "local", "gha", "s3", "azblob"]
AllDeviceType = Literal["cpu", "cuda", "cu116", "cu117", "cu118", "rocm", "rocm5.1.1",
                        "rocm5.4.2"]
DeviceType = Literal["cpu", "cu116", "cu117", "cu118", "rocm5.1.1", "rocm5.4.2"]

CREATE_BUILDER = "Create builder with 'docker buildx create --name fastsurfer'."
CONTAINERD_MESSAGE = (
    "Attestation requires OCI images, which are not supported by the default docker "
    "storage driver (your current storage driver?). Use containerd storage: "
    "https://docs.docker.com/storage/containerd/\n"
)
_WGET_BUILDX_FMT = (
    "wget -qO ~/.docker/cli-plugins/docker-buildx https://github.com/docker"
    "/buildx/releases/download/{0:s}/buildx-{0:s}.{1:s}"
)
INSTALL_BUILDX = (
    f"Install buildx with '{_WGET_BUILDX_FMT.format('<version>', '<platform>')}', "
    f"e.g. '{_WGET_BUILDX_FMT.format('v0.12.1', 'linux-amd64')}.\nYou may need to "
    f"'chmod +x ~/.docker/cli-plugins/docker-buildx'\nSee also "
    f"https://github.com/docker/buildx#manual-download."
)

__import_cache = {}


class DEFAULTS:
    # Here (and in the Literals at the top of the document), we need to update the cuda
    # and rocm versions, if pytorch comes with new versions.
    # torch 1.12.0 comes compiled with cu113, cu116, rocm5.0 and rocm5.1.1
    # torch 2.0.1 comes compiled with cu117, cu118, and rocm5.4.2
    MapDeviceType: Dict[AllDeviceType, DeviceType] = dict(
        ((d, d) for d in get_args(DeviceType)),
        rocm="rocm5.1.1",
        cuda="cu117",
    )
    BUILD_BASE_IMAGE = "ubuntu:22.04"
    RUNTIME_BASE_IMAGE = "ubuntu:22.04"
    FREESURFER_BUILD_IMAGE = "build_freesurfer"
    CONDA_BUILD_IMAGE = "build_conda"


def docker_image(arg) -> str:
    """
    Returns a str with the image.

    Raises
    ======
    ArgumentTypeError
        if it is not a valid docker image.
    """
    from re import match
    # regex from https://stackoverflow.com/questions/39671641/regex-to-parse-docker-tag
    pattern = r"^(?:(?=[^:\/]{1,253})(?!-)[a-zA-Z0-9-]{1,63}(?<!-)" \
              r"(?:\.(?!-)[a-zA-Z0-9-]{1,63}(?<!-))*(?::[0-9]{1,5})?/)?" \
              r"((?![._-])(?:[a-z0-9._-]*)(?<![._-])(?:/(?![._-])[a-z0-9._-]*" \
              r"(?<![._-]))*)(?::(?![.-])[a-zA-Z0-9_.-]{1,128})?$"
    if match(pattern, arg):
        return arg
    else:
        raise argparse.ArgumentTypeError(
            f"The image '{arg}' does not look like a valid image name."
        )


def target(arg) -> Target:
    """Returns a tuple of the target (see `Target`) and a str with the docker image.

    Raises
    ======
    ArgumentTypeError
        if not valid."""
    if isinstance(arg, str) and arg in get_args(Target):
        return cast(Target, arg)
    else:
        raise argparse.ArgumentTypeError(
            f"target must be one of {', '.join(get_args(Target))}, but was {arg}."
        )


class CacheSpec:
    """Class to parse the cache specification."""

    _type: CacheType
    _params: dict

    CACHE_PARAMETERS: Dict[CacheType, Tuple[List[str], List[str]]] = {
        "inline": ([], []),
        "registry": (
            ["ref", "compression", "compression-level", "force-compression",
             "image-manifest", "oci-mediatypes", "ignore-error"],
            ["ref"]),
        "local": (
            ["dest", "tag", "compression", "compression-level", "force-compression",
             "image-manifest", "oci-mediatypes", "ignore-error"],
            ["src", "digest", "tag"]),
        "gha": (["scope", "ignore-error"], ["scope"]),
        "s3": (
            ["bucket", "region", "name", "prefix", "ignore-error"],
            ["bucket", "region", "name", "prefix", "blobs_prefix", "manifests_prefix"]),
        "azblob": (
            ["account_url", "name", "prefix", "ignore-error"],
            ["account_url", "name", "prefix", "blobs_prefix", "manifests_prefix"]),
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
            raise ValueError(f"{_type} is not a valid cache type of "
                             f"{', '.join(get_args(CacheType))}")

    def to_str(self, from_str_format: bool, **kwargs) -> str:
        valid_parameters = self.CACHE_PARAMETERS[self.type][int(from_str_format)]
        params = {k: v for k, v in self._params.items() if k in valid_parameters}
        params["type"] = self.type
        params.update(kwargs)
        return ",".join("=".join(i) for i in params.items())

    def format_cache_from(self) -> str:
        return self.to_str(True)

    def format_cache_to(self) -> str:
        return self.to_str(False, mode=self._params.get("mode", "min"))

    __repr__ = format_cache_from


def make_parser() -> argparse.ArgumentParser:
    try:
        from FastSurferCNN.segstats import HelpFormatter
    except ImportError:
        from argparse import HelpFormatter
    parser = argparse.ArgumentParser(
        description="Docker build script for FastSurfer",
        formatter_class=HelpFormatter,
    )

    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "cu117", "cu118", "rocm", "rocm5.4.2"],
        required=True,
        help="""selection of internal build stages to build for a specific platform.<br>
                - cuda: defaults to cu118, cuda 11.8<br>
                - cpu: only cpu support<br>
                - rocm: defaults to rocm5.4.2 (experimental)""",
    )
    parser.add_argument(
        "--tag",
        type=docker_image,
        dest="image_tag",
        metavar="image[:tag]",
        help="""tag build stage/target as <image>[:<tag>]""",
    )
    parser.add_argument(
        "--target",
        default="runtime",
        type=target,
        choices=get_args(Target),
        metavar="target",
        help=f"""target to build (from list of targets below, defaults to runtime):<br>
                 - build_conda: "finished" conda build image<br>
                 - build_freesurfer: "finished" freesurfer build image<br>
                 - runtime: final fastsurfer runtime image""",
    )
    cache_kwargs = {}
    if "FASTSURFER_BUILD_CACHE" in os.environ:
        try:
            cache_kwargs = {
                "default": CacheSpec(os.environ["FASTSURFER_BUILD_CACHE"])
            }
        except ValueError as e:
            logger.warning(
                f"ERROR while parsing the environment variable 'FASTSURFER_BUILD_CACHE' "
                f"{os.environ['FASTSURFER_BUILD_CACHE']} (ignoring this environment "
                f"variable): {e.args[0]}"
            )
    parser.add_argument(
        "--cache",
        type=CacheSpec,
        help=f"""cache as defined in https://docs.docker.com/build/cache/backends/ 
                 (using --cache-to syntax, parameters are automatically filtered for use 
                 in --cache-to and --cache-from), e.g.: 
                 --cache type=registry,ref=server/fastbuild,mode=max.
                 Will default to the environment variable FASTSURFER_BUILD_CACHE: 
                 {cache_kwargs.get('default', 'N/A')}""",
        **cache_kwargs,
    )
    parser.add_argument(
        "--dry_run",
        "--print",
        action="store_true",
        help="Instead of starting processes, write the commands to stdout, so they can "
             "be dry_run with 'build.py ... --dry_run | bash'.",
    )
    parser.add_argument(
        "--tag_dev",
        action="store_true",
        help="Also tag the resulting image as 'fastsurfer:dev'.",
    )
    parser.add_argument(
        "--save_image",
        dest="image_path",
        default=None,
        help="Export the image to a to a tarball.",
    )
    parser.add_argument(
        "--singularity",
        type=Path,
        default=None,
        help="Specify a singularity file name to build a singularity image into.",
    )

    expert = parser.add_argument_group('Expert options')

    parser.add_argument(
        "--attest",
        action="store_true",
        help="add sbom and provenance attestation (requires docker-container buildkit "
             "builder created with 'docker buildx create')",
    )
    parser.add_argument(
        "--action",
        choices=("load", "push"),
        default="load",
        help="Which action to perform after building the image (if a docker-container "
             "is detected): "
             "'load' loads the image into the current docker context (default), "
             "'push' pushes the image to the registry (needs --tag <registry>/"
             "<name+maybe organization>:<tag>)",
    )
    expert.add_argument(
        "--freesurfer_build_image",
        type=docker_image,
        metavar="image[:tag]",
        help="""explicitly specifies an image to copy freesurfer binaries from.
                freesurfer binaries are expected to be in /opt/freesurfer in the image, 
                like the runtime image. By default, uses the "build_freesurfer" stage in 
                the Dockerfile (either by building it or from cache, see --cache).""")
    expert.add_argument(
        "--conda_build_image",
        type=docker_image,
        metavar="image[:tag]",
        help="""explicitly specifies an image to copy the python environment from.
                The environment is expected to be in /venv in the image, like the 
                runtime image. By default, uses the "build_conda" stage in the 
                Dockerfile (either by building it or from cache, see --cache).""")
    expert.add_argument(
        "--runtime_base_image",
        type=docker_image,
        metavar="image[:tag]",
        help="explicitly specifies the base image to build the runtime image from "
             "(default: ubuntu:22.04).")
    expert.add_argument(
        "--build_base_image",
        type=docker_image,
        metavar="image[:tag]",
        help="explicitly specifies the base image to build the build images from "
             "(default: ubuntu:22.04).")

    expert.add_argument(
        "--debug",
        action="store_true",
        help="enables the DEBUG build flag."
    )
    return parser


def red(skk):
    return "\033[91m {}\033[00m" .format(skk)


def get_builder(
        Popen,
        builder_type: str,
        require_builder_type: bool = False,
) -> tuple[bool, str]:
    """Get the builder to build the fastsurfer image."""
    from subprocess import PIPE
    from re import compile

    buildx_binfo = Popen(["docker", "buildx", "ls"], stdout=PIPE, stderr=PIPE).finish()
    header, *lines = buildx_binfo.out_str("utf-8").strip().split("\n")
    header_pattern = compile("\\S+\\s*")
    fields = {}
    alternative_builder = "use_default"
    pos = 0
    while pos < len(header) and (match := header_pattern.search(header, pos)):
        start, pos = match.span()
        fields[match.group().strip()] = slice(start, pos)
    builders = {line[fields["NAME/NODE"]].strip(): line[fields["DRIVER/ENDPOINT"]].strip()
                for line in lines if not line.startswith(" ")}
    default_builders = [name for name in builders.keys() if name.endswith("*")]
    if len(default_builders) != 1:
        raise RuntimeError("Could not find default builder of buildx")
    default_builder = default_builders[0][:-1].strip()
    builders[default_builder] = builders[default_builders[0]]
    del builders[default_builders[0]]
    builder_is_correct_type = builders[default_builder] == builder_type
    default_builder_is_correct_type = builder_is_correct_type
    if not builder_is_correct_type:
        # if the default builder is a docker builder (which may not support features)
        # see if there is an alternative builder named "fastsurfer*"
        for builder in builders.keys():
            if builder.startswith("fastsurfer") and builders[builder] == builder_type:
                # set the default_builder to this (prefered) builder
                alternative_builder = builder
                break
    # update is_correct_type
    if alternative_builder != "use_default":
        builder_is_correct_type = builders[alternative_builder] == builder_type
    if not builder_is_correct_type and require_builder_type:
        # did not find an appropriate builder, but is required!!
        raise RuntimeError(
            "Could not find an appropriate builder from the current builder "
            "(see docker buildx use) or builders named fastsurfer* (searching for "
            f"a builder of type {builder_type}, docker builders may not be supported "
            f"with the selected export settings. {CREATE_BUILDER}"
        )
    return default_builder_is_correct_type, alternative_builder


def docker_build_image(
        image_name: str,
        dockerfile: Path,
        working_directory: Optional[Path] = None,
        context: Path | str = ".",
        dry_run: bool = False,
        attestation: bool = False,
        action: Literal["load", "push"] = "load",
        image_path: Path | str | None = None,
        **kwargs) -> None:
    """
    Build a docker image.

    Parameters
    ----------
    image_name : str
        Name / target tag of the image.
    dockerfile : Path, str
        Path to the Dockerfile.
    working_directory : Path, str, optional
        Path o the working directory to perform the build operation (default: inherit).
    context : Path, str, optional
        Base path to the context folder to build the docker image from (default: '.').
    dry_run : bool, optional
        Whether to actually trigger the build, or just print the command to the console
        (default: False => actually build).
    cache_to : str, optional
        Forces usage of buildx over build, use docker build caching as in the --cache-to
        argument to docker buildx build.
    attestation : bool, default=False
        Whether to create sbom and provenance attestation
    action : "load", "push", default="load"
        The operation to perform after the image is built (only if a docker-container
        builder is detected).
    image_path : Path, str, optional
        A path to save the image to.

    Additional kwargs add additional build flags to the build command in the following
    manner: "_" is replaced by "-" in the keyword name and each sequence entry is passed
    with its own flag, e.g. `docker_build_image(..., build_arg=["TEST=1", "VAL=2"])` is
    translated to `docker [buildx] build ... --build-arg TEST=1 --build-arg VAL=2`.
    """
    from itertools import chain, repeat
    from shutil import which
    from subprocess import PIPE

    from FastSurferCNN.utils.run_tools import Popen

    logger.info("Building. This starts with sending the build context to the docker "
                "daemon, which may take a while...")
    extra_env = {"DOCKER_BUILDKIT": "1"}

    docker_cmd = which("docker")
    if docker_cmd is None:
        raise FileNotFoundError("Could not locate the docker executable")

    if action not in ("load", "push"):
        raise ValueError(f"Invalid Value for 'action' {action}, must be load or push.")

    def to_pair(key, values):
        if isinstance(values, Sequence) and isinstance(values, (str, bytes)):
            values = [values]
        key_dashed = key.replace("_", "-")
        # concatenate the --key_dashed value pairs
        return list(chain(*zip(repeat(f"--{key_dashed}"), values)))

    kw = {"stdout": PIPE, "stderr": PIPE}
    _buildx = Popen([docker_cmd, "buildx", "version"], **kw)
    _storage = Popen([docker_cmd, "info", "-f", "{{.DriverStatus}}"], **kw)
    has_buildx = "'buildx' is not a docker command" not in _buildx.finish().err_str()
    has_storage = "io.containerd.snapshotter" in _storage.finish().out_str()

    def is_inline_cache(cache_kw):
        inline_cache = "type=inline"
        all_inline_cache = (None, "", inline_cache)
        return kwargs.get(cache_kw, inline_cache) not in all_inline_cache

    # require buildx for sbom and provenance and cache != inline
    require_container = (attestation or
                         any(is_inline_cache(f"cache_{c}") for c in ("to", "from")))
    import_after_args = []
    if dest := image_path or "":
        dest = f",dest={dest}"
    if not has_buildx:
        # only standard build environment arguments available
        if require_container:
            # not supported with builder != docker-container
            raise RuntimeError(
                "Using --cache_{from,to} or attestation requires docker buildx and a "
                f"docker-container builder.\n{INSTALL_BUILDX}\n{CREATE_BUILDER}"
            )
        if action == "push" or not dest:
            raise RuntimeError(
                "The legacy docker builder does not support pushing or exporting the "
                "image."
            )
        args = ["build"]
        kwargs_to_exclude = [f"cache_{c}" for c in ("to", "from")]
    else:
        # buildx argument construction
        args = ["buildx", "build"]
        # raises RuntimeError, if a docker-container builder is required, but not found
        default_builder_is_container, alternative_builder = get_builder(
            Popen,
            "docker-container",
            require_container,
        )
        if has_storage or action == "push":
            # with containerd storage driver or pushing to registry
            image_type = f"oci{dest}"
            # both support attestation no problem
        elif attestation and dest:
            # also implicitly action == load
            logger.warning(
                f"{CONTAINERD_MESSAGE}The build script will save the image to "
                f"{image_path} (which will contain the attestation manifest files, "
                f"but the imported image {image_name} will not."
            )
            image_type = f"oci{dest}"
            if dry_run:
                print(f"mkdir -p {Path(image_path).parent} && ", sep="")
            else:
                Path(image_path).parent.mkdir(exist_ok=True)
            import_after_args = ["image", "import", image_path, image_name]
        elif attestation:
            # also implicitly action == load
            raise RuntimeError(
                f"{CONTAINERD_MESSAGE}Alternatively, save the image to preserve the "
                f"manifest files in that file."
            )
        else:
            # no attestation, docker builder supports this format
            image_type = f"docker{dest}"

        args.extend(["--output", f"type={image_type},name={image_name}"])
        if not bool(import_after_args):
            args.append(f"--{action}")
        if attestation:
            args.extend([
                "--attest", "type=sbom",
                "--attest", "type=provenance",
            ])
        if not default_builder_is_container:
            args.extend(["--builder", alternative_builder])

        kwargs_to_exclude = []

    params = [to_pair(k, v) for k, v in kwargs.items() if k not in kwargs_to_exclude]
    # arguments for standard build and buildx
    args.extend([
        "-t", image_name,
        "-f", str(dockerfile),
    ])
    args.extend(chain(*params))
    args.append(str(context))

    if dry_run:
        extra_environment = [f"{k}={v}" for k, v in extra_env.items()]
        print(" ".join(extra_environment + [docker_cmd] + args), sep="")
        if import_after_args:
            print(" && " + " ".join([docker_cmd] + import_after_args), sep="")
    else:
        env = dict(os.environ)
        env.update(extra_env)

        def forward_output_to_logger(process):
            for msg in process:
                if msg.out:
                    logger.info("stdout: " + msg.out.decode("utf-8"))
                if msg.err:
                    logger.info("stderr: " + red(msg.err.decode("utf-8")))

        with Popen([docker_cmd] + args + ["--progress=plain"],
                   cwd=working_directory, env=env, stdout=subprocess.PIPE) as proc:
            forward_output_to_logger(proc)
        if import_after_args:
            with Popen([docker_cmd] + import_after_args,
                       cwd=working_directory, env=env, stdout=subprocess.PIPE) as proc:
                forward_output_to_logger(proc)


def singularity_build_image(
        image_name: str,
        singularity_image: Path,
        working_directory: Optional[Path] = None,
        dry_run: bool = False,
):
    """
    Build the singularity image from the docker image.

    Parameters
    ----------
    image_name : str
        The name of the docker image to build the singularity image from.
    singularity_image : Path
        The path and file of the singularity image to build.
    working_directory : Path, str, optional
        Path o the working directory to perform the build operation (default: inherit).
    dry_run : bool, default=False
        Whether to build from python or to print the command to stdout.
    """
    from shutil import which

    # Create the folder for the singularity image
    singularity_image.parent.mkdir(exist_ok=True)
    args = [
        which("singularity"),
        "build",
        "--force",
        str(singularity_image),
        f"docker-daemon://{image_name}",
    ]
    if dry_run:
        print(" ".join([" &&"] + args), sep="")
    else:
        from FastSurferCNN.utils.run_tools import Popen
        with Popen(args,
                   cwd=working_directory, stdout=subprocess.PIPE) as proc:
            for msg in proc:
                if msg.out:
                    logger.info("stdout: " + msg.out.decode("utf-8"))
                if msg.err:
                    logger.info("stderr: " + red(msg.err.decode("utf-8")))


def main(
        device: DeviceType,
        cache: Optional[CacheSpec] = None,
        target: Target = "runtime",
        debug: bool = False,
        image_tag: Optional[str] = None,
        dry_run: bool = False,
        tag_dev: bool = True,
        fastsurfer_home: Optional[Path] = None,
        **keywords,
        ) -> int | str:
    from FastSurferCNN.version import has_git, main as version
    kwargs: Dict[str, Union[str, List[str]]] = {}
    if cache is not None:
        if not isinstance(cache, CacheSpec):
            cache = CacheSpec(cache)
        logger.info(f"cache: {cache}")
        kwargs["cache_from"] = cache.format_cache_from()
        kwargs["cache_to"] = cache.format_cache_from()

    fastsurfer_home = Path(fastsurfer_home) if fastsurfer_home else default_home()

    if target not in get_args(Target):
        raise ValueError(f"Invalid target: {target}")
    if device not in get_args(AllDeviceType):
        raise ValueError(f"Invalid device: {device}")
    if keywords.get("action", "load") == "push":
        kwargs["action"] = "push"
    # special case to add extra environment variables to better support AWS and ROCm
    if device.startswith("cu") and target == "runtime":
        target = "runtime_cuda"
    kwargs["target"] = target
    kwargs["build_arg"] = [f"DEVICE={DEFAULTS.MapDeviceType.get(device, 'cpu')}"]
    if debug:
        kwargs["build_arg"].append(f"DEBUG=true")
    build_arg_list = [
        "build_base_image",
        "runtime_base_image",
        "freesurfer_build_image",
        "conda_build_image",
    ]
    for key in build_arg_list:
        upper_key = key.upper()
        value = keywords.get(key) or getattr(DEFAULTS, upper_key)
        kwargs["build_arg"].append(f"{upper_key}={value}")
    #    kwargs["build_arg"] = " ".join(kwargs["build_arg"])

    build_filename = fastsurfer_home / "BUILD.info"
    if has_git():
        version_sections = "+git"
    else:
        # try creating the build file without git info
        version_sections = ""
        logger.warning(
            "Failed to create the git_status section in the BUILD.info file. "
            "The resulting build file will not have valid git information, so "
            "the version command of FastSurfer in the image will not complete."
        )

    with open(build_filename, "w") as build_file, \
            open(fastsurfer_home / "pyproject.toml") as project_file:
        ret_version = version(
            version_sections,
            project_file=project_file,
            file=build_file,
            build_cache=False,
        )
    if ret_version != 0:
        return f"Creating the version file failed with message: {ret_version}"

    with open(build_filename, "r") as build_file:
        from FastSurferCNN.version import parse_build_file
        build_info = parse_build_file(build_file)

    version_tag = build_info["version_tag"]
    image_prefix = ""
    if device != "cuda":
        image_prefix = f"{device}-"
    # image_tag is None or ""
    if not bool(image_tag):
        image_tag = f"fastsurfer:{image_prefix}{version_tag}".replace("+", "_")
        logger.info(f"No image name/tag provided, auto-generated tag: {image_tag}")

    attestation = bool(keywords.get("attest"))
    if tag_dev:
        kwargs["tag"] = f"fastsurfer:dev{image_prefix}"
    if keywords.get("image_path", False):
        kwargs["image_path"] = keywords["image_path"]

    if not dry_run:
        logger.info("Version info added to the docker image:")
        logger.info(build_info["content"])

    dockerfile = fastsurfer_home / "Docker" / "Dockerfile"
    try:
        docker_build_image(
            image_tag,
            dockerfile,
            working_directory=fastsurfer_home,
            context=fastsurfer_home,
            dry_run=dry_run,
            attestation=attestation,
            **kwargs,
        )
        if singularity := keywords.get("singularity", None):
            singularity_build_image(
                image_tag,
                Path(singularity),
                dry_run=dry_run,
            )
        print("")
    except RuntimeError as e:
        return e.args[0]
    return 0


def default_home() -> Path:
    """
    Find the fastsurfer path.

    Returns
    -------
    Path
        The FASTSURFER_HOME-path.
    """
    if "FASTSURFER_HOME" in os.environ:
        return Path(os.environ["FASTSURFER_HOME"])
    else:
        return Path(__file__).parent.parent


if __name__ == "__main__":
    import sys
    logging.basicConfig(stream=sys.stdout)
    arguments = make_parser().parse_args()

    # make sure the code can run without FastSurfer being in PYTHONPATH
    fastsurfer_home = default_home()
    if str(fastsurfer_home) not in sys.path:
        sys.path.append(str(fastsurfer_home))

    logger.setLevel(logging.WARN if arguments.dry_run else logging.INFO)
    sys.exit(main(**vars(arguments), fastsurfer_home=fastsurfer_home))
