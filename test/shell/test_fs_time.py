"""
Check recon_surf/fs_time, the per-command resource reporter.

It used to be a bash frontend to ``/usr/bin/time -f``, which is a GNU extension, so it could not
work on macOS at all: functions.sh probes it and silently drops every per-command timing when the
probe fails. It is python now, and these tests are deliberately platform independent, because the
implementation branches on ``sys.platform`` for the fields the kernel does not populate and for the
units of ru_maxrss. Both branches need to be exercised, so this file runs on linux and on macOS.

The bash-3.2-specific tests live in test_brun_bash32.py and only run on macOS.
"""

import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pytest

FASTSURFER_HOME = Path(__file__).parent.parent.parent
FS_TIME = FASTSURFER_HOME / "recon_surf" / "fs_time"
# fields the reporter prints as "." on this platform, because the OS does not measure them for a
# child process. Keep in step with UNAVAILABLE in fs_time.
UNMEASURED = ("I", "O", "W") if sys.platform == "darwin" else ()


def run_fs_time(*args: str, env: dict[str, str] | None = None):
    """Run fs_time with FASTSURFER_HOME set, and return the completed process."""
    return subprocess.run(
        [str(FS_TIME), *args],
        capture_output=True,
        text=True,
        env=dict(os.environ, FASTSURFER_HOME=str(FASTSURFER_HOME), **(env or {})),
        timeout=120,
    )


def parse_fields(line: str) -> dict[str, str]:
    """The `<name> <value>` pairs after `N <nargs>`, as a mapping."""
    parts = line.split()
    tail = parts[parts.index("N") + 2 :]
    return dict(zip(tail[::2], tail[1::2], strict=False))


def test_reports_elapsed_time():
    """The elapsed field is the point of the whole thing, and what the log analyser reads."""
    result = run_fs_time("-k", "@#@FSTIME", "--no-load", "sleep", "1")
    assert result.returncode == 0, result.stderr
    fields = parse_fields(result.stderr.strip())
    assert 1.0 <= float(fields["e"]) < 3.0, result.stderr


def test_timestamp_is_the_command_start():
    """Field 2 is documented as the onset of execution, and the analyser adds the duration to it.

    Reading the clock after waiting for the child would put the end time here instead, which shifts
    every interval that extract_recon_surf_time_info.py derives forward by a whole command runtime.
    """
    before = datetime.now().replace(microsecond=0)
    result = run_fs_time("-k", "@#@FSTIME", "--no-load", "sleep", "3")
    after = datetime.now()
    assert result.returncode == 0, result.stderr
    stamp = datetime.strptime(result.stderr.split()[1], "%Y:%m:%d:%H:%M:%S")
    elapsed = float(parse_fields(result.stderr.strip())["e"])
    # the stamp belongs at the start, so it must not have drifted into the second half of the run
    assert before <= stamp, f"stamp {stamp} predates the run that started at {before}"
    assert (stamp - before).total_seconds() < elapsed, (
        f"stamp {stamp} looks like the end time, not the start: the run took {elapsed}s between {before} and {after}"
    )


def test_field_layout_is_the_documented_contract():
    """extract_recon_surf_time_info.py parses this line positionally, so the order is fixed."""
    result = run_fs_time("-k", "@#@FSTIME", "--no-load", "echo", "hello")
    parts = result.stderr.strip().split()
    assert parts[0] == "@#@FSTIME"
    datetime.strptime(parts[1], "%Y:%m:%d:%H:%M:%S")  # raises if the stamp shape changed
    assert parts[2] == "echo"
    assert parts[3] == "N" and parts[4] == "1"
    assert parts[5] == "e", f"the analyser asserts field 5 is 'e', got {parts[5]!r}"
    float(parts[6])


def test_unmeasured_fields_are_dots_only_where_unmeasured():
    """A field the OS does not fill is reported as ".", not as a misleading 0.

    This is the platform branch: macOS counts neither block IO nor swaps for a child, linux counts
    block IO. Running this on one platform only would leave the other branch unverified.
    """
    result = run_fs_time("-k", "@#@FSTIME", "--no-load", "echo", "hello")
    fields = parse_fields(result.stderr.strip())
    for name in ("e", "S", "U", "M", "F", "R", "c", "w"):
        assert fields[name] != ".", f"{name} should always be measured, got '.'"
    for name in ("I", "O", "W"):
        if name in UNMEASURED:
            assert fields[name] == ".", f"{name} is not measured on {sys.platform}, expected '.'"
        else:
            assert fields[name].isdigit(), f"{name} should be a number on {sys.platform}"


def test_maxrss_is_reported_in_kilobytes():
    """M is kilobytes on both platforms, though ru_maxrss is bytes on macOS and kB on linux.

    Without the conversion the macOS figure is off by 1024, so allocating a known amount is the
    only way to tell the two apart.
    """
    megabytes = 200
    result = run_fs_time(
        "-k",
        "@#@FSTIME",
        "--no-load",
        sys.executable,
        "-c",
        f"x = bytearray({megabytes} * 1024 * 1024); x[::4096] = b'1' * (len(x) // 4096)",
    )
    assert result.returncode == 0, result.stderr
    maxrss = int(parse_fields(result.stderr.strip())["M"])
    # generous bounds: the interpreter itself is a few MB, and the point is the unit, not the exact
    # figure. Bytes would read ~1024x larger, kilobytes land just above what we allocated.
    assert megabytes * 1024 < maxrss < megabytes * 1024 * 20, f"M={maxrss} is not plausible kB"


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        (["/bin/sh", "-c", "exit 0"], 0),
        (["/bin/sh", "-c", "exit 1"], 1),
        (["/bin/sh", "-c", "exit 42"], 42),
        (["/bin/sh", "-c", "kill -TERM $$"], 143),  # 128 + SIGTERM
    ],
)
def test_exit_status_is_forwarded(command: list[str], expected: int):
    """recon-surf checks the status of every timed command, so it has to survive the wrapper."""
    result = run_fs_time("--no-load", *command)
    assert result.returncode == expected, result.stderr


def test_missing_command_is_reported_not_raised():
    """A command that is not on PATH gets 127, as a shell would report, without a traceback."""
    result = run_fs_time("--no-load", "no-such-command-anywhere")
    assert result.returncode == 127, result.stderr
    assert "Traceback" not in result.stderr, result.stderr


def test_unwritable_outfile_fails_before_running_the_command(tmp_path: Path):
    """-o is opened up front, as /usr/bin/time does, which exits 125 and runs nothing.

    Opening it only after the command finished would spend the whole runtime and then throw the
    result away with a traceback.
    """
    marker = tmp_path / "the-command-ran"
    result = run_fs_time("--no-load", "-o", str(tmp_path / "no-such-dir" / "out.txt"), "touch", str(marker))
    assert result.returncode == 125, result.stderr
    assert "Traceback" not in result.stderr, result.stderr
    assert not marker.exists(), "the command ran even though its timing could not be written"


def test_interrupt_still_reports(tmp_path: Path):
    """Ctrl-C still writes the timing line, rather than a traceback.

    A terminal sends SIGINT to the whole foreground process group, so the wrapper is interrupted
    while it waits for the command. Exiting there loses the record for the command the user just
    interrupted and prints a python traceback into the log, once per command still in flight.
    """
    outfile = tmp_path / "times.txt"
    process = subprocess.Popen(
        [str(FS_TIME), "-k", "@#@FSTIME", "--no-load", "-o", str(outfile), "sleep", "30"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=dict(os.environ, FASTSURFER_HOME=str(FASTSURFER_HOME)),
        start_new_session=True,  # its own process group, so the signal can target the group
    )
    time.sleep(1)
    os.killpg(process.pid, signal.SIGINT)
    _, stderr = process.communicate(timeout=60)

    assert "Traceback" not in stderr, stderr
    assert process.returncode == 128 + signal.SIGINT, f"expected 130, got {process.returncode}"
    assert outfile.exists(), f"no timing line written; stderr:\n{stderr}"
    fields = parse_fields(outfile.read_text().strip())
    # it was interrupted after about a second, so the elapsed figure has to reflect that, not the 30
    assert 0.5 <= float(fields["e"]) < 10, outfile.read_text()


def test_outfile_is_written_and_appended(tmp_path: Path):
    """-o redirects the line to a file, and -a appends rather than truncating."""
    outfile = tmp_path / "times.txt"
    run_fs_time("-k", "@#@FSTIME", "--no-load", "-o", str(outfile), "echo", "one")
    run_fs_time("-k", "@#@FSTIME", "--no-load", "-o", str(outfile), "-a", "echo", "two")
    assert len(outfile.read_text().splitlines()) == 2, outfile.read_text()
    # without -a the file is truncated
    run_fs_time("-k", "@#@FSTIME", "--no-load", "-o", str(outfile), "echo", "three")
    assert len(outfile.read_text().splitlines()) == 1, outfile.read_text()


@pytest.mark.parametrize(
    ("value", "load_expected"),
    [(None, True), ("", True), ("1", True), ("0", False), ("2", False)],
)
def test_fstime_load_default(value: str | None, load_expected: bool):
    """Unset and empty both mean on, which is what the bash version did and what the help says."""
    env = {} if value is None else {"FSTIME_LOAD": value}
    if value is None:
        os.environ.pop("FSTIME_LOAD", None)
    result = run_fs_time("-k", "@#@FSTIME", "echo", "hello", env=env)
    assert ("@#@FSLOADPRE" in result.stdout) is load_expected, result.stdout


def test_load_lines_carry_the_averages():
    """Both load samples report three averages; the post-run one used to report none at all."""
    result = run_fs_time("-k", "@#@FSTIME", "--load", "echo", "hello")
    for key in ("@#@FSLOADPRE", "@#@FSLOADPOST"):
        line = next(li for li in result.stdout.splitlines() if li.startswith(key))
        assert re.search(r"\bL (\d+\.\d\d) (\d+\.\d\d) (\d+\.\d\d)$", line), f"{key}: {line!r}"


def test_python_commands_are_named_by_their_script():
    """A timed python call is reported as the script, so the key stays a readable command name."""
    result = run_fs_time("-k", "@#@FSTIME", "--no-load", sys.executable, "-c", "pass")
    assert result.stderr.split()[2] != Path(sys.executable).name, result.stderr


def test_time_it_writes_an_entry(tmp_path: Path):
    """functions.sh accepts fs_time, and time_it writes a parsable entry.

    The probe is the part that mattered on macOS: functions.sh runs fs_time once and silently
    disables every per-command timing if it fails, which it always did there.
    """
    log = tmp_path / "exectime.log"
    script = (
        f'source "{FASTSURFER_HOME}/recon_surf/functions.sh" > /dev/null 2>&1\n'
        f'if [[ -z "$timecmd" ]] ; then echo "PROBE-FAILED" ; exit 1 ; fi\n'
        f'time_it "{log}" sleep 1\n'
    )
    result = subprocess.run(
        ["/bin/bash", "-c", script],
        capture_output=True,
        text=True,
        env=dict(os.environ, FASTSURFER_HOME=str(FASTSURFER_HOME)),
        timeout=120,
    )
    assert "PROBE-FAILED" not in result.stdout, "functions.sh could not use fs_time"
    assert log.exists(), result.stdout + result.stderr
    entry = log.read_text()
    assert " e 1." in entry, f"expected ~1 s elapsed, got:\n{entry}"
