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
    """A stand-in for run_fastsurfer.sh that logs when it starts and stops, and does nothing else."""
    script = tmp_path / "stub.sh"
    script.write_text(
        "#!/bin/bash\n"
        'sid=""\n'
        'while [[ "$#" -gt 0 ]] ; do\n'
        '  if [[ "$1" == "--sid" ]] ; then sid="$2" ; fi\n'
        "  shift\n"
        "done\n"
        'echo "$(date +%s) START $sid" >> "$BT_LOG"\n'
        "sleep 1\n"
        'echo "$(date +%s) END $sid" >> "$BT_LOG"\n'
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
    env = dict(os.environ, BT_LOG=str(log))
    return subprocess.run(
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


def _started(log: Path) -> list[str]:
    if not log.exists():
        return []
    return [line.split()[2] for line in log.read_text().splitlines() if " START " in line]


def _peak_concurrency(log: Path) -> int:
    events = []
    for line in log.read_text().splitlines():
        stamp, kind, _ = line.split(None, 2)
        events.append((int(stamp), 1 if kind == "START" else -1))
    events.sort()
    current = peak = 0
    for _, delta in events:
        current += delta
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
    assert _peak_concurrency(log) <= 2, f"expected at most 2 concurrent, log:\n{log.read_text()}"


@requires_bash3
def test_fs_time_works(tmp_path: Path):
    """fs_time reports timings under bash 3.2, where the GNU-time version could not run at all.

    functions.sh probes fs_time and silently disables per-command timings if it fails, which is what
    happened on every macOS run.
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
    entry = log.read_text()
    # the elapsed field is what the log is for, and what extract_recon_surf_time_info.py reads
    assert " e 1." in entry, f"expected ~1 s elapsed, got:\n{entry}"
