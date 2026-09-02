"""
Check that brun_fastsurfer actually works under the bash macOS ships.

These three paths used to abort with "requires at minimum bash version 4", because they built arrays
with ``mapfile``. They are only meaningful where ``/bin/bash`` really is 3.x, i.e. on macOS, so they
skip elsewhere; the static counterpart in test_bash4_lint.py runs everywhere.

They substitute a stub for run_fastsurfer.sh via ``--run_fastsurfer``, so no image is processed.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

FASTSURFER_HOME = Path(__file__).parent.parent.parent
SYSTEM_BASH = "/bin/bash"


def _system_bash_major() -> int:
    out = subprocess.run([SYSTEM_BASH, "-c", "echo ${BASH_VERSINFO[0]}"], capture_output=True, text=True, check=True)
    return int(out.stdout.strip())


requires_bash3 = pytest.mark.skipif(
    sys.platform != "darwin" or not Path(SYSTEM_BASH).exists() or _system_bash_major() != 3,
    reason=f"needs {SYSTEM_BASH} to be bash 3.x, which is the macOS case this guards",
)


@pytest.fixture
def stub(tmp_path: Path) -> Path:
    """A stand-in for run_fastsurfer.sh that logs when it starts and stops, and does nothing else.

    It records the --t1 it was handed as well as the --sid, so a test can tell whether the path
    survived the subject-line parsing intact.
    """
    script = tmp_path / "stub.sh"
    script.write_text(
        "#!/bin/bash\n"
        'sid=""\n'
        't1=""\n'
        'while [[ "$#" -gt 0 ]] ; do\n'
        '  if [[ "$1" == "--sid" ]] ; then sid="$2" ; fi\n'
        '  if [[ "$1" == "--t1" ]] ; then t1="$2" ; fi\n'
        "  shift\n"
        "done\n"
        'echo "$(date +%s) START $sid $t1" >> "$BT_LOG"\n'
        "sleep 1\n"
        'echo "$(date +%s) END $sid $t1" >> "$BT_LOG"\n'
    )
    script.chmod(0o755)
    return script


@pytest.fixture
def subjects(tmp_path: Path) -> str:
    """Three subject specs, `<id>=<t1>`, whose images exist so brun does not skip them."""
    lines = []
    for name in ("subj1", "subj2", "subj3"):
        image = tmp_path / f"{name}.nii.gz"
        image.write_bytes(b"")
        lines.append(f"{name}={image}")
    return "\n".join(lines) + "\n"


def _run_brun(stub: Path, log: Path, tmp_path: Path, *args: str, stdin: str | None = None):
    """Run brun_fastsurfer under bash 3.2 and require it to succeed.

    Every caller expects success, so the status is checked here rather than in each of them: the
    log assertions alone would pass a regression that starts each stub and then exits nonzero.
    Note brun_fastsurfer.sh ends with an unconditional `exit 0`, so what this really guards is the
    early-exit paths, e.g. the "No subjects specified" and "Could not parse the line" errors.
    """
    env = dict(os.environ, BT_LOG=str(log))
    result = subprocess.run(
        [
            SYSTEM_BASH,
            str(FASTSURFER_HOME / "brun_fastsurfer.sh"),
            "--sd",
            str(tmp_path / "out"),
            "--run_fastsurfer",
            str(stub),
            *args,
        ],
        input=stdin,
        capture_output=True,
        text=True,
        env=env,
        cwd=FASTSURFER_HOME,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"brun_fastsurfer.sh exited {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    return result


def _started(log: Path) -> list[str]:
    if not log.exists():
        return []
    return [line.split()[2] for line in log.read_text().splitlines() if " START " in line]


def _t1_paths(log: Path) -> dict[str, str]:
    """The --t1 each subject was started with, keyed by subject id."""
    started = {}
    for line in log.read_text().splitlines():
        parts = line.split(None, 3)
        if len(parts) == 4 and parts[1] == "START":
            started[parts[2]] = parts[3]
    return started


def _peak_concurrency(log: Path) -> int:
    """The largest number of stubs that were running at once.

    In log order, not sorted by the timestamp. The stubs append to one file, so the file is already
    in event order, whereas the timestamps have a one-second resolution and sorting ties puts every
    END before every START in the same second. That systematically understates the peak, which is
    the direction that would let broken job accounting pass: a real peak of three reads as two when
    the third START shares a second with an earlier END.
    """
    current = peak = 0
    for line in log.read_text().splitlines():
        current += 1 if " START " in line else -1
        peak = max(peak, current)
    return peak


@requires_bash3
def test_subject_list(stub: Path, tmp_path: Path, subjects: str):
    """--subject_list reads every subject; it used to abort with 'requires bash version 4'."""
    log = tmp_path / "log.txt"
    listfile = tmp_path / "subjects.txt"
    listfile.write_text(subjects)
    result = _run_brun(stub, log, tmp_path, "--subject_list", str(listfile), "--parallel", "max")
    # sorted: with --parallel max the three start concurrently, so the order is not deterministic
    assert sorted(_started(log)) == ["subj1", "subj2", "subj3"], result.stdout + result.stderr


@requires_bash3
def test_subjects_via_stdin(stub: Path, tmp_path: Path, subjects: str):
    """Subjects on stdin are read; this also used to abort."""
    log = tmp_path / "log.txt"
    result = _run_brun(stub, log, tmp_path, "--parallel", "max", stdin=subjects)
    assert sorted(_started(log)) == ["subj1", "subj2", "subj3"], result.stdout + result.stderr


@requires_bash3
def test_parallel_limit_is_enforced(stub: Path, tmp_path: Path, subjects: str):
    """--parallel <n> throttles rather than being refused, and rather than silently not throttling.

    An empty running-jobs array would spawn everything at once, so asserting the peak is what
    actually tests the job accounting, not just that the subjects ran.
    """
    log = tmp_path / "log.txt"
    listfile = tmp_path / "subjects.txt"
    listfile.write_text(subjects)
    result = _run_brun(stub, log, tmp_path, "--subject_list", str(listfile), "--parallel", "2")
    assert len(_started(log)) == 3, result.stdout + result.stderr
    # exactly 2, not at most 2: a peak of 1 would mean it serialised the run instead of throttling
    # it, which is also broken job accounting, so the upper bound alone would let that through
    assert _peak_concurrency(log) == 2, f"expected exactly 2 concurrent, log:\n{log.read_text()}"


@requires_bash3
def test_subject_list_is_cleaned_up_and_complete(stub: Path, tmp_path: Path):
    """A list file with the messy bits the sed cleanup exists for, and no final newline.

    Four things at once, each of which used to lose or corrupt a subject on macOS:
    a CRLF line ending, a blank line, a line with trailing spaces, a t1 path ending in "s" (BSD sed
    reads GNU's \\s as the letter s, so it truncated that path), and a final line with no newline
    after it (plain `read` reports failure at EOF and would drop it, where mapfile kept it).
    """
    log = tmp_path / "log.txt"
    images = {name: tmp_path / name for name in ("t1.mgz", "t2.mgz", "images", "t4.mgz")}
    for image in images.values():
        image.write_bytes(b"")
    listfile = tmp_path / "subjects.txt"
    listfile.write_bytes(
        (
            f"crlf={images['t1.mgz']}\r\n"
            "\n"
            f"spaces={images['t2.mgz']}   \n"
            f"trailing_s={images['images']}\n"
            # deliberately no newline after the last line
            f"unterminated={images['t4.mgz']}"
        ).encode()
    )
    result = _run_brun(stub, log, tmp_path, "--subject_list", str(listfile), "--parallel", "max")
    started = _t1_paths(log)
    assert sorted(started) == ["crlf", "spaces", "trailing_s", "unterminated"], result.stdout + result.stderr
    # each t1 must have arrived intact, in particular the one ending in "s"
    for sid, expected in (
        ("crlf", images["t1.mgz"]),
        ("spaces", images["t2.mgz"]),
        ("trailing_s", images["images"]),
        ("unterminated", images["t4.mgz"]),
    ):
        assert started[sid] == str(expected), f"{sid} got t1 {started[sid]!r}, expected {str(expected)!r}"


@requires_bash3
def test_quoted_t1_paths_reach_run_fastsurfer(stub: Path, tmp_path: Path):
    """A t1 path containing a space works, in each of the three ways of writing one.

    The tokenizer matches the source text of a token, so before unquote() the quotes and backslashes
    were still attached: --t1 received "'/d/a b.mgz'", which is not the name of any file. Unquoted,
    the path is also the one case where a subject line legitimately contains a space, so it is worth
    asserting the exact argv rather than only that the subject ran.
    """
    log = tmp_path / "log.txt"
    spaced = tmp_path / "a b.mgz"
    apostrophe = tmp_path / "it's.mgz"
    plain = tmp_path / "plain.mgz"
    # a name with a real backslash in it, to pin down the double-quote rule: inside double quotes a
    # backslash only escapes $ ` " and \, so "a\ b.mgz" has to keep its backslash, as in the shell
    backslashed = tmp_path / "a\\ b.mgz"
    for image in (spaced, apostrophe, plain, backslashed):
        image.write_bytes(b"")
    listfile = tmp_path / "subjects.txt"
    listfile.write_text(
        f"plain={plain}\n"
        f"escaped={tmp_path}/a\\ b.mgz\n"
        f"single='{spaced}'\n"
        f'double="{spaced}"\n'
        f"apostrophe=\"{apostrophe}\"\n"
        f'backslash="{backslashed}"\n'
    )
    result = _run_brun(stub, log, tmp_path, "--subject_list", str(listfile), "--parallel", "max")
    started = _t1_paths(log)
    expected = {
        "plain": str(plain),
        "escaped": str(spaced),
        "single": str(spaced),
        "double": str(spaced),
        "apostrophe": str(apostrophe),
        "backslash": str(backslashed),
    }
    assert sorted(started) == sorted(expected), result.stdout + result.stderr
    for sid, path in expected.items():
        assert started[sid] == path, f"{sid} got t1 {started[sid]!r}, expected {path!r}"
        # the point of the exercise: the path it was handed is one that actually exists
        assert Path(started[sid]).is_file(), f"{sid} got a t1 that is not a file: {started[sid]!r}"


@requires_bash3
def test_fs_time_runs_under_bash32(tmp_path: Path):
    """fs_time works when recon-surf is driven by bash 3.2, where GNU time could not run at all.

    Everything else about fs_time is platform independent and lives in test_fs_time.py, which runs
    on linux too; this only pins down that the bash 3.2 caller can use it.
    """
    log = tmp_path / "exectime.log"
    script = (
        f'source "{FASTSURFER_HOME}/recon_surf/functions.sh" > /dev/null 2>&1\n'
        f'if [[ -z "$timecmd" ]] ; then echo "PROBE-FAILED" ; exit 1 ; fi\n'
        f'time_it "{log}" sleep 1\n'
    )
    result = subprocess.run(
        [SYSTEM_BASH, "-c", script],
        capture_output=True,
        text=True,
        env=dict(os.environ, FASTSURFER_HOME=str(FASTSURFER_HOME)),
        timeout=120,
    )
    assert "PROBE-FAILED" not in result.stdout, "functions.sh could not use fs_time"
    assert log.exists(), result.stdout + result.stderr
    assert " e 1." in log.read_text(), f"expected ~1 s elapsed, got:\n{log.read_text()}"
