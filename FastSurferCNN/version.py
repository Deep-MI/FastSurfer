#!/bin/python

import argparse
import re
import shutil
import subprocess
from io import TextIOWrapper
from pathlib import Path
from typing import Any, Dict, Optional, Union


class DEFAULTS:
    PROJECT_ROOT = Path(__file__).parent.parent
    BUILD_TXT = PROJECT_ROOT / "BUILD.info"
    PROJECT_TOML = PROJECT_ROOT / "pyproject.toml"
    VERSION_SECTIONS = {
        "checkpoints": ("+checkpoints", "checkpoints:"),
        "git_status": ("+git", "git status:"),
        "pypackages": ("+pip", "python packages:"),
    }


def section(arg: str) -> str:
    """
    Validate the argument is a valid sections string.

    A valid sections string is either 'all' or a concatenated list of '+checkpoints',
    '+git', and '+pip', e.g. '+git+checkpoints'. The order does not matter.

    Parameters
    ----------
    arg : str
        The input string to be validated.

    Returns
    -------
    str
        The validated sections string.
    """
    from re import match

    if arg == "all":
        return "+checkpoints+git+pip"
    elif match("^(\\+branch|\\+checkpoints|\\+git|\\+pip)*$", arg):
        return arg
    else:
        raise argparse.ArgumentTypeError(
            "The section argument must be 'all', or any combination of "
            "'+branch', '+checkpoints', '+git' and '+pip'."
        )


def make_parser():
    """
    Generate the argument parser for the version script.

    Returns
    -------
    argparse.ArgumentParser
        The argument parser for the version script.
    """
    parser = argparse.ArgumentParser(
        description="Helper script to read and write version information"
    )
    parser.add_argument(
        "--sections",
        default="",
        type=section,
        help="Sections to include from +checkpoints, +git, +pip. If not passed, will "
        "only have the version number.",
    )
    parser.add_argument(
        "--build_cache",
        type=argparse.FileType("r"),
        help=f"File to read version info to read from (default: {DEFAULTS.BUILD_TXT}).",
    )
    parser.add_argument(
        "--project_file",
        type=argparse.FileType("r"),
        help=f"File to project detail / version info from (default: {DEFAULTS.PROJECT_TOML}).",
    )
    parser.add_argument(
        "-o",
        "--file",
        default=None,
        type=argparse.FileType("w"),
        help=f"File to write version info to (default: write to stdout).",
    )
    parser.add_argument(
        "--prefer_cache",
        action="store_true",
        help="Avoid running commands and only read the build file ",
    )
    return parser


def print_build_file(
    version: str,
    git_hash: str = "",
    git_branch: str = "",
    git_status: str = "",
    checkpoints: str = "",
    pypackages: str = "",
    file: Optional[TextIOWrapper] = None,
) -> None:
    """
    Format and print the build file.

    Parameters
    ----------
    version : str
        The version number to print.
    git_hash : str, optional
        The git hash associated with the build, may be empty.
    git_branch : str, optional
        The git branch associated with the build, may be empty.
    git_status : str, optional
        The md5sums of the checkpoint files, leave empty to skip.
    checkpoints : str, optional
        The md5sums of the checkpoint files, leave empty to skip.
    pypackages : str, optional
        The md5sums of the checkpoint files, leave empty to skip.
    file : TextIOWrapper, optional
        A file-like object to write the build file to, default: stdout.

    Notes
    -----
    See also main.
    """

    if file is None:
        file = sys.stdout

    def print_header(section_name: str) -> None:
        section_info = DEFAULTS.VERSION_SECTIONS[section_name]
        print("=" * 10, section_info[1], "=" * 10, sep="\n", file=file)

    version_line = version
    if git_hash:
        version_line += "+" + git_hash
    if git_branch:
        version_line += f" ({git_branch})"
    print(version_line, end="\n", file=file)
    if git_status:
        print_header("git_status")
        print(git_status, file=file)
    if checkpoints:
        print_header("checkpoints")
        print(checkpoints, file=file)
    if pypackages:
        print_header("pypackages")
        print(pypackages, file=file)


def main(
    sections: str = "",
    project_file: Optional[TextIOWrapper] = None,
    build_cache: Optional[TextIOWrapper] = None,
    file: Optional[TextIOWrapper] = None,
    prefer_cache: bool = False,
) -> Union[str, int]:
    """
    Print version info to stdout or file.

    Prints/writes version info of FastSurfer in the style:
    ```
    <version number>[+<git hash>][ (<git active branch>)]
    [==========
    checkpoints:
    ==========
    <md5sum> <file>
    ...]
    [==========
    git status:
    ==========
    <git status text>]
    [==========
    python packages:
    ==========
    Package         Version    Location   [Installer]
    <package name>  <version>  <path>     <pip|conda>
    ...]
    ```

    $PROJECT_ROOT is the root directory of the project determined as the parent to this
    file's directory.

    Parameters
    ----------
    sections : str
        String describing which sections the output should include. Can be 'all' or a
        concatenated list of '+branch', '+checkpoints', '+git', and '+pip', e.g.
        '+git+checkpoints'.
        The order does not matter, '+checkpoints', '+git' or '+pip' also implicitly
        activate '+branch'.
    project_file : TextIOWrapper, optional
        A file-like object to read the projects toml file, with the '[project]' section
        with a 'version' attribute. Defaults to $PROJECT_ROOT/pyproject.toml.
    build_cache : TextIOWrapper, optional
        A file-like object to read cached version information, the format should be
        formatted like the output of `main`. Defaults to $PROJECT_ROOT/BUILD.info.
    file : TextIOWrapper, optional
        A file-like object to write the output to, defaults to stdout if None or not
        passed.
    prefer_cache : bool, default=False
        Whether to prefer information from the `build_cache` over online generation.

    Returns
    -------
    int or str
        Returns 0, if the function was successful, a error message if a problem occurred.
    """
    has_git = (
        shutil.which("git") is not None and (DEFAULTS.PROJECT_ROOT / ".git").is_dir()
    )
    has_build_cache = build_cache is not None or DEFAULTS.BUILD_TXT.is_file()

    if prefer_cache and not has_build_cache:
        return (
            "Trying to force the use cached version information, but no build information file "
            f"was passed found at the default location ({DEFAULTS.BUILD_TXT})."
        )

    if sections == "all":
        sections = "+checkpoints+git+pip"

    from FastSurferCNN.utils.run_tools import Popen, PyPopen

    build_cache_required = prefer_cache
    kw_root = {"cwd": DEFAULTS.PROJECT_ROOT, "stdout": subprocess.PIPE}
    futures = {}

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor() as pool:
        futures["version"] = pool.submit(read_and_close_version, project_file)
        # if we do not have git, try VERSION file else git sha and branch
        if has_git and not prefer_cache:
            futures["git_hash"] = Popen(
                ["git", "rev-parse", "--short", "HEAD"], **kw_root
            ).as_future(pool)
            if sections != "":
                futures["git_branch"] = Popen(
                    ["git", "branch", "--show-current"], **kw_root
                ).as_future(pool)
            if "+git" in sections:
                futures["git_status"] = pool.submit(
                    filter_git_status, Popen(["git", "status", "-s", "-b"], **kw_root)
                )
        else:
            build_cache_required = True

        if build_cache_required:
            futures["build_cache"] = pool.submit(parse_build_file, build_cache)

        if "+checkpoints" in sections and not prefer_cache:

            def calculate_md5_for_checkpoints() -> "MessageBuffer":
                from glob import glob

                files = glob(str(DEFAULTS.PROJECT_ROOT / "checkpoints" / "*"))
                shorten = len(str(DEFAULTS.PROJECT_ROOT)) + 1
                files = [f[shorten:] for f in files]
                return Popen(["md5sum"] + files, **kw_root).finish()

            futures["checkpoints"] = pool.submit(calculate_md5_for_checkpoints)

        if "+pip" in sections and not prefer_cache:
            futures["pypackages"] = PyPopen(
                ["-m", "pip", "list", "--verbose"], **kw_root
            ).as_future(pool)

    if build_cache_required:
        build_cache = futures.pop("build_cache").result()
    else:
        build_cache = {}

    build_file_kwargs = {}

    try:
        version = futures.pop("version").result()
    except IOError:
        version = build_cache["version_no"]

    def __future_or_cache(
        key: str, futures: Dict[str, Any], cache: Dict[str, Any]
    ) -> str:
        future = futures.get(key, None)
        if future:
            returnmsg = future.result()
            if isinstance(returnmsg, str):
                return returnmsg
            elif returnmsg.retcode != 0:
                raise RuntimeError(
                    f"The calculation/determination of {key} has failed."
                )
            return returnmsg.out_str("utf-8").strip()
        elif key in cache:
            # fill from cache
            return cache[key]
        else:
            add_msg = ""
            if prefer_cache:
                add_msg = " The cached build file seems to not contain this info?"
            # ERROR
            raise RuntimeError(f"Could not find a valid value for {key}!" + add_msg)

    try:
        build_file_kwargs["git_hash"] = __future_or_cache(
            "git_hash", futures, build_cache
        )
        if sections != "":
            build_file_kwargs["git_branch"] = __future_or_cache(
                "git_branch", futures, build_cache
            )
        for key in ("git_status", "checkpoints", "pypackages"):
            if DEFAULTS.VERSION_SECTIONS[key][0] in sections:
                # stuff that is needed
                build_file_kwargs[key] = __future_or_cache(key, futures, build_cache)

        print_build_file(version, **build_file_kwargs, file=file)
    except RuntimeError as e:
        return e.args[0]

    return 0


def parse_build_file(build_file: Optional[TextIOWrapper]) -> Dict[str, str]:
    """Read and parse a build file (same as output of `main`).

    Read and parse a file with version information in the format that is also the
    output of the `main` function. The format is documented in `main`.

    Parameters
    ----------
    build_file : TextIOWrapper, optional
        File-like object, will be closed.

    Returns
    -------
    dict
        Dictionary with keys 'version_line', 'version', 'git_hash', 'git_branch',
        'checkpoints', 'git_status', and 'pip'. The last 3 are optional and may
        be missing depending on the content of the file.

    Notes
    -----
    See also main.
    """
    file_cache: Dict[str, str] = {}
    try:
        if build_file is None:
            build_file = open(DEFAULTS.BUILD_TXT, "r")
        file_cache["content"] = "".join(build_file.readlines())
    finally:
        build_file.close()
    section_pattern = re.compile("\n={3,}\n")
    file_cache["version_line"], *rest = section_pattern.split(file_cache["content"], 1)
    version_regex = re.compile(
        "([a-zA-Z.0-9\\-]+)(\\+([0-9A-Fa-f]+))?(\\s+\\(([^)]+)\\))?\\s*"
    )
    hits = version_regex.search(file_cache["version_line"])
    if hits is None:
        raise RuntimeError(
            "The build file has invalid formatting, version tag not " "recognized!",
            f"First line was '{file_cache['version_line']}' and did "
            f"not fit the pattern '{version_regex.pattern}.",
        )
    (
        file_cache["version"],
        _,
        file_cache["git_hash"],
        _,
        file_cache["git_branch"],
    ) = hits.groups("")
    if file_cache["git_hash"]:
        file_cache["version_tag"] = file_cache["version"] + "+" + file_cache["git_hash"]
    else:
        file_cache["version_tag"] = file_cache["version"]

    def get_section_name_by_header(header: str) -> Optional[str]:
        for name, info in DEFAULTS.VERSION_SECTIONS.items():
            if info[1] == header:
                return name

    while len(rest) > 0:
        section_header, section_content, *rest = section_pattern.split(rest[0], 2)
        section_name = get_section_name_by_header(section_header)
        if section_name:
            file_cache[section_name] = section_content
    return file_cache


def read_version_from_project_file(project_file: TextIOWrapper) -> str:
    """
    Read the version entry from the pyproject file.

    Searches for the [project] section in project_file, therein the version attribute.
    Extracts the Value. The file pointer is right after the version attribute at return
    time.

    Parameters
    ----------
    project_file : TextIOWrapper
        File pointer to the project file to read from.

    Returns
    -------
    str
        The version string.
    """
    project_pattern = re.compile(r"\[project]")
    version_pattern = re.compile('version\\s*=\\s*(\\")?([^\\"]+)\\1')
    version = "unspecified"

    seek_to_project = True
    seek_to_version = True
    line = "before"
    while seek_to_project and line != "":
        line = project_file.readline()
        seek_to_project = not project_pattern.match(line)
    while seek_to_version and line != "":
        line = project_file.readline()
        hits = version_pattern.search(line)
        if hits is not None:
            seek_to_version = False
            version = hits.group(2)
            if version[0] == '"':
                version = version.strip('"')
    return version


def filter_git_status(git_process: "FastSurferCNN.utils.run_tools.Popen") -> str:
    """
    Filter the output of a running git status process.

    Parameters
    ----------
    git_process : FastSurferCNN.utils.run_tools.Popen
        The Popen process object that will return the git status output.

    Returns
    -------
    str
        The git status string filtered to exclude lines containing "__pycache__".
    """
    from FastSurferCNN.utils.run_tools import Popen

    finished_process = git_process.finish()
    if finished_process.retcode != 0:
        raise RuntimeError("Failed git status command")
    git_status_text = finished_process.out_str("utf-8")
    return "\n".join(
        filter(lambda x: "__pycache__" not in x, git_status_text.split("\n"))
    )


def read_and_close_version(project_file: Optional[TextIOWrapper] = None) -> str:
    """
    Read and close the version from the pyproject file. Also fill default.

    Always closes the file pointer.

    Parameters
    ----------
    project_file : Optional[TextIOWrapper] = None
        Project file.

    Returns
    -------
    str
        The version read from the pyproject file.

    Notes
    -----
    See also FastSurferCNN.version.read_version_from_project_file
    """
    if project_file is None:
        project_file = open(DEFAULTS.PROJECT_TOML, "r")
    try:
        version = read_version_from_project_file(project_file)
    finally:
        project_file.close()
    return version


if __name__ == "__main__":
    import sys

    args = make_parser().parse_args()
    sys.exit(main(**vars(args)))
