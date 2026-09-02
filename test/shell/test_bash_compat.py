"""
Keep the shipped shell scripts runnable under bash 3.2.

macOS ships bash 3.2.57 as ``/bin/bash`` (Apple froze it at the last GPLv2 release), and every
shipped script selects that interpreter through its own ``#!/bin/bash`` shebang, whatever the user's
login shell is. A bash-4-only builtin therefore breaks FastSurfer on macOS while passing every test
we run on Linux, where bash is 5.x.

``test_no_bash4_only_constructs`` is the cheap guard and runs everywhere. The functional tests
exercise the paths that used to be refused on macOS and are only meaningful where ``/bin/bash`` is
actually 3.2, so they skip elsewhere.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

FASTSURFER_HOME = Path(__file__).parent.parent.parent

# Everything that runs under a *macOS* bash: the pipeline entry points, the recon_surf scripts they
# call, the macOS build, and the scripts that run at install time or ship inside the package.
# Deliberately excluded: tools/Docker/entrypoint.sh and tools/export_pip-r.sh (run inside the linux
# image) and CerebNet/datasets/realistic_deformations.sh (a training helper), none of which macOS
# ever executes.
SHIPPED_SCRIPTS = (
    ["run_fastsurfer.sh", "brun_fastsurfer.sh", "srun_fastsurfer.sh", "long_fastsurfer.sh",
     "stools.sh"]
    + [str(p.relative_to(FASTSURFER_HOME)) for p in sorted((FASTSURFER_HOME / "recon_surf").glob("*.sh"))]
    + ["tools/build/install_fs_pruned.sh", "tools/build/link_fs.sh",
       "tools/macos_build/build_release_package.sh"]
    # rendered by the build into the installed package, so they run on the user's machine
    + ["tools/macos_build/macos_setup_fastsurfer.sh.template",
       "tools/macos_build/scripts/postinstall.sh.template",
       "tools/macos_build/scripts/preinstall.sh.template"]
)

# Constructs that bash 3.2 does not have. The value is what to use instead, quoted in the failure.
BASH4_ONLY = {
    r"\bmapfile\b": "while IFS= read -r line ; do arr+=(\"$line\") ; done < <(...)",
    r"\breadarray\b": "while IFS= read -r line ; do arr+=(\"$line\") ; done < <(...)",
    r"\b(declare|local)\s+-A\b": "parallel indexed arrays, or a delimited string",
    r"\b(declare|local)\s+-n\b": "pass the value, or eval",
    r"\bcoproc\b": "a named pipe (mkfifo)",
    r"\bwait\s+-n\b": "wait for a specific pid from `jobs -pr`",
    r"\$\{[A-Za-z_][A-Za-z_0-9]*\^\^": "tr '[:lower:]' '[:upper:]'",
    r"\$\{[A-Za-z_][A-Za-z_0-9]*,,": "tr '[:upper:]' '[:lower:]'",
    r"&>>": "'>> file 2>&1'",
}


def _code_lines(path: Path):
    """Yield (lineno, text) for lines that are not whole-line comments."""
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.lstrip().startswith("#"):
            yield number, line


@pytest.mark.parametrize("script", SHIPPED_SCRIPTS)
def test_no_bash4_only_constructs(script: str):
    """No bash 4+ construct in a script that runs under macOS's bash 3.2."""
    path = FASTSURFER_HOME / script
    if not path.exists():
        pytest.skip(f"{script} does not exist")
    findings = [
        f"  {script}:{number}: {line.strip()}\n    use instead: {alternative}"
        for number, line in _code_lines(path)
        for pattern, alternative in BASH4_ONLY.items()
        if re.search(pattern, line)
    ]
    assert not findings, (
        "bash 4+ construct in a script that runs under macOS's bash 3.2:\n" + "\n".join(findings)
    )


# ---------------------------------------------------------------------------------------------
# Functional: the paths that were refused on bash 3.2 before they stopped using mapfile.

SYSTEM_BASH = "/bin/bash"


def _system_bash_major() -> int:
    out = subprocess.run(
        [SYSTEM_BASH, "-c", "echo ${BASH_VERSINFO[0]}"], capture_output=True, text=True, check=True
    )
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
        '#!/bin/bash\n'
        'sid=""\n'
        'while [[ "$#" -gt 0 ]] ; do\n'
        '  if [[ "$1" == "--sid" ]] ; then sid="$2" ; fi\n'
        '  shift\n'
        'done\n'
        'echo "$(date +%s) START $sid" >> "$BT_LOG"\n'
        'sleep 1\n'
        'echo "$(date +%s) END $sid" >> "$BT_LOG"\n'
    )
    script.chmod(0o755)
    return script


def _run_brun(stub: Path, log: Path, tmp_path: Path, *args: str, stdin: str | None = None):
    env = dict(os.environ, BT_LOG=str(log))
    return subprocess.run(
        [SYSTEM_BASH, str(FASTSURFER_HOME / "brun_fastsurfer.sh"),
         "--sd", str(tmp_path / "out"), "--run_fastsurfer", str(stub), *args],
        input=stdin, capture_output=True, text=True, env=env, cwd=FASTSURFER_HOME, timeout=300,
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


@pytest.fixture
def subjects(tmp_path: Path) -> str:
    """Three subject specs, `<id>=<t1>`, whose images exist so brun does not skip them."""
    lines = []
    for name in ("subj1", "subj2", "subj3"):
        image = tmp_path / f"{name}.nii.gz"
        image.write_bytes(b"")
        lines.append(f"{name}={image}")
    return "\n".join(lines) + "\n"


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
