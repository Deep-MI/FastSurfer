#!/bin/python

import argparse
import io
import re
import shutil
import subprocess
from collections.abc import Iterable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from functools import lru_cache
from hashlib import md5
from os import PathLike
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, TextIO, TypedDict, cast, get_args


class DEFAULTS:
    PROJECT_ROOT = Path(__file__).parent.parent
    BUILD_TXT = PROJECT_ROOT / "BUILD.info"
    PROJECT_TOML = PROJECT_ROOT / "pyproject.toml"
    VERSION_SECTIONS = {
        "checkpoints": ("+checkpoints", "checkpoints:"),
        "git_status": ("+git", "git status:"),
        "pypackages": ("+pip", "python packages:"),
    }


# the first line of the version info, e.g. "2.6.0+1dc02473 (stable)", see main
VERSION_LINE = re.compile(r"^(\d+\.\d+\.\d+[^\s+]*|N/A)(\+([0-9A-Fa-f]+))?( \(([^)]*)\))?$")
# the header of each invocation in a build log, which run_fastsurfer.sh appends after the version info
INVOCATION_HEADER = "invocation:"


class _RequiredVersionDict(TypedDict):
    """
    Dictionary with keys 'version_line', 'version', 'git_hash', 'git_branch'.
    """
    git_branch: str
    git_hash: str
    version: str
    version_line: str


class _OptionalVersionDict(TypedDict, total=False):
    """
    Dictionary with optional keys 'checkpoints', 'git_status', and 'pypackages'.
    """
    content: str
    checkpoints: str
    git_status: str
    pypackages: str
    version_tag: str


class VersionDict(_RequiredVersionDict, _OptionalVersionDict):
    """
    Dictionary with keys 'version_line', 'version', 'git_hash', 'git_branch',
    'checkpoints', 'git_status', and 'pypackages'. The last 3 are optional and may
    be missing depending on the content of the file.
    """
    pass


VersionDictKeys = Literal[
    "git_branch",
    "git_hash",
    "version",
    "version_line",
    "checkpoints",
    "git_status",
    "pypackages",
]


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
            "The section argument must be 'all', or any combination of '+branch', '+checkpoints', '+git' and '+pip'."
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
        help="File to write version info to (default: write to stdout).",
    )
    parser.add_argument(
        "--append",
        type=Path,
        default=None,
        help="Build log to append the version info to, unless it is the same as the last version info there; prints "
             "a warning if only the environment differs, e.g. the python packages.",
    )
    parser.add_argument(
        "--compare",
        type=Path,
        default=None,
        help="Build log to compare this version with: if the last version recorded there is another major or minor "
             "version, print its version line and exit with 3; another patch release, development version or "
             "commit, exit with 4.",
    )
    parser.add_argument(
        "--prefer_cache",
        action="store_true",
        help="Avoid running commands and only read the build file.",
    )
    return parser


@lru_cache
def has_git():
    """
    Determine whether FastSurfer is installed as a git directory.

    Returns
    -------
    bool
        Whether git commands on the FastSurfer dir likely work.
    """
    return shutil.which("git") is not None and (DEFAULTS.PROJECT_ROOT / ".git").is_dir()


def print_build_file(
    version: str,
    git_hash: str = "",
    git_branch: str = "",
    git_status: str = "",
    checkpoints: str = "",
    pypackages: str = "",
    file: TextIO | None = None,
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
    file : TextIO, optional
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
    project_file: TextIO | None = None,
    build_cache: TextIO | Literal[False] | None = None,
    file: TextIO | None = None,
    prefer_cache: bool = False,
) -> str | int:
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
    <package name>  <version>  <path>     <pip|...>
    ...]
    ```

    $PROJECT_ROOT is the root directory of the project determined as the parent to this file's directory.

    Parameters
    ----------
    sections : str
        String describing which sections the output should include. Can be 'all' or a concatenated list of '+branch',
        '+checkpoints', '+git', and '+pip', e.g. '+git+checkpoints'.
        The order does not matter, '+checkpoints', '+git' or '+pip' also implicitly activate '+branch'.
    project_file : TextIO, optional
        A file-like object to read the projects toml file, with the '[project]' section with a 'version' attribute.
        Defaults to $PROJECT_ROOT/pyproject.toml.
    build_cache : False, TextIO, optional
        A file-like object to read cached version information, the format should be formatted like the output of `main`.
        If build_cache is None, it defaults to $PROJECT_ROOT/BUILD.info; if it is False, it is ignored.
    file : TextIO, optional
        A file-like object to write the output to, defaults to stdout if None or not passed.
    prefer_cache : bool, default=False
        Whether to prefer information from the `build_cache` over online generation.

    Returns
    -------
    int or str
        Returns 0, if the function was successful, or an error message. A failed git status, checkpoints or python
        packages section is left out of the output, which still contains the others.
    """
    has_build_cache = False
    # ignore build_cache, if it is False
    if build_cache is not False:
        # if build_cache is not False, use the passed value or the default
        has_build_cache = build_cache is not None or DEFAULTS.BUILD_TXT.is_file()

    if prefer_cache and not has_build_cache:
        return (
            "Trying to force the use of cached version information (--prefer_cache), but no build information file was "
            f"passed found at the default location ({DEFAULTS.BUILD_TXT})."
        )

    if sections == "all":
        sections = "+checkpoints+git+pip"

    from FastSurferCNN.utils.run_tools import MessageBuffer, Popen, PyPopen

    build_cache_required = prefer_cache
    kw_root = {"cwd": DEFAULTS.PROJECT_ROOT, "stdout": subprocess.PIPE}
    futures: dict[str, Future[str | MessageBuffer | VersionDict]] = {}

    with ThreadPoolExecutor() as pool:
        futures["version"] = pool.submit(read_and_close_version, project_file)
        # if we do not have git, try VERSION file else git sha and branch
        if has_git() and not prefer_cache:
            git_rev_parse_cmd = ["git", "rev-parse", "--short", "HEAD"]
            futures["git_hash"] = Popen(git_rev_parse_cmd, **kw_root).as_future(pool, timeout=10.0)
            if sections != "":
                git_branch_current_cmd = ["git", "branch", "--show-current"]
                futures["git_branch"] = Popen(git_branch_current_cmd, **kw_root).as_future(pool, timeout=10.0)
            if "+git" in sections:
                futures["git_status"] = pool.submit(filter_git_status, Popen(["git", "status", "-sb"], **kw_root))
        else:
            # we go not have git, try loading the build cache
            build_cache_required = True

        if build_cache_required and build_cache is not False:
            futures["build_cache"] = pool.submit(parse_build_file, build_cache)

        if "+checkpoints" in sections and not prefer_cache:

            def calculate_md5_for_checkpoints() -> "MessageBuffer":
                files = list(map(str, cast(Iterable[Path], DEFAULTS.PROJECT_ROOT.glob("checkpoints/*"))))
                if len(files) == 0:
                    return MessageBuffer(out=b"", err=b"No checkpoints found.", retcode=0, runtime=0.0)
                # hashlib rather than the md5sum binary, which macOS did not ship until recently:
                # the checkpoints are hundreds of MB, so read them in chunks. The output format is
                # md5sum's, two spaces between hash and path, so BUILD.info is unchanged.
                start = perf_counter()
                lines = []
                for checkpoint_file in files:
                    digest = md5(usedforsecurity=False)
                    with open(checkpoint_file, "rb") as checkpoint:
                        while chunk := checkpoint.read(1 << 22):
                            digest.update(chunk)
                    lines.append(f"{digest.hexdigest()}  {checkpoint_file}\n")
                out = "".join(lines).encode("utf-8")
                return MessageBuffer(out=out, err=b"", retcode=0, runtime=perf_counter() - start)

            futures["checkpoints"] = pool.submit(calculate_md5_for_checkpoints)

        if "+pip" in sections and not prefer_cache:
            pip_command = "-m pip list --verbose --no-cache-dir --no-color --disable-pip-version-check"
            futures["pypackages"] = PyPopen(pip_command.split(" "), **kw_root).as_future(pool, timeout=10.0)

    if build_cache_required and build_cache is not False:
        build_cache: VersionDict = cast(VersionDict, futures.pop("build_cache").result())
    else:
        build_cache: VersionDict = get_default_version_info()

    build_file_kwargs = {}

    try:
        version = cast(str, futures.pop("version").result())
    except OSError:
        version = build_cache["version"]

    def __future_or_cache(key: VersionDictKeys, futures: dict[str, Future[Any]], cache: VersionDict) -> str:
        future: None | Future[Any] = futures.get(key, None)
        if future is not None:
            returnmsg = future.result()
            if isinstance(returnmsg, str):
                return returnmsg
            elif returnmsg.retcode != 0:
                raise RuntimeError(f"The calculation/determination of {key} has failed.")
            return returnmsg.out_str("utf-8").strip()
        elif key in cache:
            # fill from cache
            return cache[key]
        else:
            add_msg = ""
            if key == "git_status":
                add_msg += (" --sections all or --sections with +git require a build "
                            "cache file or a FastSurfer git directory and git.")
            if prefer_cache:
                add_msg += " The cached build file seems to not contain this info?"
            # ERROR
            raise RuntimeError(f"Could not find a valid value for {key}!" + add_msg)

    try:
        build_file_kwargs["git_hash"] = __future_or_cache("git_hash", futures, build_cache)
        if sections != "":
            build_file_kwargs["git_branch"] = __future_or_cache("git_branch", futures, build_cache)
    except RuntimeError as e:
        return e.args[0]

    # a section that fails is left out and reported, the others are still written
    errors = []
    keys: Sequence[VersionDictKeys] = ("git_status", "checkpoints", "pypackages")
    for key in keys:
        if DEFAULTS.VERSION_SECTIONS[key][0] in sections:
            try:
                build_file_kwargs[key] = __future_or_cache(key, futures, build_cache)
            except RuntimeError as e:
                errors.append(e.args[0])

    print_build_file(version, **build_file_kwargs, file=file)
    return " ".join(errors) if errors else 0


def get_default_version_info() -> VersionDict:
    """
    Get the blank version information.

    Returns
    -------
    VersionDict
        A dictionary with blank version information.
    """
    return {
        "version_line": "N/A",
        "version": "N/A",
        "git_hash": "0000000",
        "git_branch": "release",
    }


def parse_build_file(build_file: TextIO | io.StringIO | None) -> VersionDict:
    """Read and parse a build file (same as output of `main`).

    Read and parse a file with version information in the format that is also the output of the `main` function. The
    format is documented in `main`.

    Parameters
    ----------
    build_file : TextIO, io.StringIO, optional
        File-like object, will be closed.

    Returns
    -------
    VersionDict
        Dictionary with keys 'version_line', 'version', 'git_hash', 'git_branch', 'checkpoints', 'git_status', and
        'pypackages'. The last 3 are optional and may be missing depending on the content of the file.

    Notes
    -----
    See also main.
    """
    file_cache: VersionDict = get_default_version_info()
    if build_file is None:
        try:
            _build_file = open(DEFAULTS.BUILD_TXT)
        except FileNotFoundError:
            return get_default_version_info()
    else:
        _build_file = build_file
    if isinstance(_build_file, io.StringIO) and hasattr(_build_file, "getvalue"):
        file_cache["content"] = _build_file.getvalue()
    else:
        file_cache["content"] = "".join(_build_file.read())
    if not _build_file.closed:
        _build_file.close()
    section_pattern = re.compile("\n={3,}\n")
    file_cache["version_line"], *rest = section_pattern.split(file_cache["content"], 1)
    version_regex = re.compile("([a-zA-Z.0-9\\-]+)(\\+([0-9A-Fa-f]+))?(\\s+\\(([^)]+)\\))?\\s*")
    hits = version_regex.search(file_cache["version_line"])
    if hits is None:
        filename = getattr(build_file, "name", "<unnamed file>")
        raise RuntimeError(
            f"The build file {filename} has invalid formatting, version tag not recognized! First line was "
            f"'{file_cache['version_line']}' and did not fit the pattern '{version_regex.pattern}'.",
        )
    file_cache["version"], _, file_cache["git_hash"], _, file_cache["git_branch"] = hits.groups("")

    file_cache["version_tag"] = file_cache["version"] + (f"+{file_cache['git_hash']}" if file_cache["git_hash"] else "")

    def get_section_name_by_header(header: str) -> str | None:
        for name, info in DEFAULTS.VERSION_SECTIONS.items():
            if info[1] == header:
                return name
        return None

    while len(rest) > 0:
        section_header, section_content, *rest = section_pattern.split(rest[0], 2)
        section_name = get_section_name_by_header(section_header)
        if section_name and section_name in get_args(VersionDictKeys):
            file_cache[cast(VersionDictKeys, section_name)] = section_content
    return file_cache


def last_version_block(build_log: str) -> str | None:
    """
    Return the last version info of a build log, from its last version line to the invocation after it.

    A build log has the version info of each version that processed a subject, each followed by the
    invocations that ran with it.

    Parameters
    ----------
    build_log : str
        The content of the build log.

    Returns
    -------
    str, optional
        The last version info, None if the build log has none.
    """
    lines = build_log.splitlines()
    starts = [i for i, line in enumerate(lines) if VERSION_LINE.match(line)]
    if not starts:
        return None
    block = []
    for line in lines[starts[-1]:]:
        if line == INVOCATION_HEADER:
            break
        block.append(line)
    # the separator above the invocation header, and blank lines
    while block and block[-1] in ("", "=" * 10):
        block.pop()
    return "\n".join(block)


def version_difference(version_line: str, other_version_line: str) -> Literal["", "patch", "minor"]:
    """
    Return how two version lines differ, whatever the branch.

    Abbreviated commit hashes differ in length between repositories, so one has to start with the other.

    Parameters
    ----------
    version_line, other_version_line : str
        Version lines, e.g. "2.6.0+1dc02473 (stable)".

    Returns
    -------
    "", "patch", "minor"
        "" for the same version and commit, "minor" if the major or minor version differs or a version is not
        recognized, else "patch" (another patch release, development version or commit).
    """
    hits, other_hits = VERSION_LINE.match(version_line), VERSION_LINE.match(other_version_line)
    if hits is None or other_hits is None:
        return "minor"
    (version, _, git_hash, *_), (other_version, _, other_hash, *_) = hits.groups(""), other_hits.groups("")
    if version == other_version and (git_hash.startswith(other_hash) or other_hash.startswith(git_hash)):
        return ""
    major_minor, other_major_minor = (re.match(r"\d+\.\d+", v) for v in (version, other_version))
    if major_minor is None or other_major_minor is None or major_minor.group() != other_major_minor.group():
        return "minor"
    return "patch"


def append_build_log(build_log: Path, version_info: str) -> str | None:
    """
    Append the version info to the build log, unless it is the last version info there.

    Parameters
    ----------
    build_log : Path
        The build log.
    version_info : str
        The version info, as written by `main`.

    Returns
    -------
    str, optional
        A warning if the version is the same as the last one in the build log, but the environment differs.
    """
    content = build_log.read_text() if build_log.is_file() else ""
    previous = last_version_block(content)
    version_info = version_info.rstrip("\n")
    if previous == version_info:
        return None
    with open(build_log, "a") as file:
        file.write(("\n" if content else "") + version_info + "\n")
    if previous is not None and not version_difference(previous.split("\n", 1)[0], version_info.split("\n", 1)[0]):
        return (f"WARNING: This FastSurfer version runs in another environment than the last run on this subject, "
                f"e.g. with other python packages or checkpoints; {build_log} records both.")
    return None


def read_version_from_project_file(project_file: TextIO) -> str:
    """
    Read the version entry from the pyproject file.

    Searches for the [project] section in project_file, therein the version attribute.
    Extracts the Value. The file pointer is right after the version attribute at return
    time.

    Parameters
    ----------
    project_file : TextIO
        File pointer to the project file to read from.

    Returns
    -------
    str
        The version string.
    """
    project_pattern = re.compile(r"\[project]")
    version_pattern = re.compile('version\\s*=\\s*([\\"\']?)([^\\"]+)\\1')
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


def filter_git_status(git_process) -> str:
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
    finished_process = git_process.finish(timeout=10.0)
    if finished_process.retcode != 0:
        raise RuntimeError("Failed git status command")
    git_status_text = finished_process.out_str("utf-8")
    return "\n".join(
        filter(lambda x: "__pycache__" not in x, git_status_text.split("\n"))
    )


def read_and_close_version(project_file: TextIO | PathLike | None = None) -> str:
    """
    Read and close the version from the pyproject file. Also fill default.

    Always closes the file pointer.

    Parameters
    ----------
    project_file : TextIO, PathLike, optional
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
        _project_file = open(DEFAULTS.PROJECT_TOML)
    elif hasattr(project_file, "read"):
        _project_file = cast(TextIO, project_file)
    elif isinstance(project_file, PathLike | Path):
        _project_file = open(project_file)
    else:
        raise TypeError("project_file must be TextIO, PathLike, or None!")
    try:
        version = read_version_from_project_file(_project_file)
    finally:
        _project_file.close()
    return version


if __name__ == "__main__":
    import sys

    kwargs = vars(make_parser().parse_args())
    append, compare = kwargs.pop("append"), kwargs.pop("compare")
    if append is None and compare is None:
        sys.exit(main(**kwargs))
    buffer = io.StringIO()
    returncode = main(**(kwargs | {"file": buffer}))
    if compare is not None:
        if returncode != 0:
            sys.exit(returncode)
        previous = last_version_block(compare.read_text()) if compare.is_file() else None
        difference = previous and version_difference(previous.split("\n", 1)[0], buffer.getvalue().strip())
        if difference:
            print(previous.split("\n", 1)[0])
        sys.exit({"minor": 3, "patch": 4}.get(difference, 0))
    warning = append_build_log(append, buffer.getvalue())
    if warning:
        print(warning)
    sys.exit(returncode)
