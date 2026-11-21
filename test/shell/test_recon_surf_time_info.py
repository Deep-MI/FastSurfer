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

"""Check per-invocation timing reports and their nonfatal shell boundary."""

import shlex
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

FASTSURFER_HOME = Path(__file__).resolve().parents[2]
PARSER = FASTSURFER_HOME / "recon_surf" / "utils" / "extract_recon_surf_time_info.py"


@pytest.mark.parametrize("case", ["fresh", "appended", "stage-less"])
def test_parser_reports_latest_invocation(tmp_path, case):
    """Keep startup timing, metadata and optional stage details within one invocation."""
    stage = "#@# Mask BFS Thu Jan 1 10:00:01 UTC 2026\n" if case != "stage-less" else ""
    latest = (
        "Log file for recon-surf.sh\n"
        "Thu Jan 1 10:00:00 UTC 2026\n"
        "recon-surf.sh --sid current\n"
        "python3 /fastsurfer/FastSurferCNN/quick_qc.py --asegdkt_segfile example.mgz\n"
        "@#@FSTIME 2026:01:01:10:00:00 FastSurferCNN/quick_qc.py N 2 e 1.00\n"
        "================= SURFACE ======================================================\n"
        "recon-all -s current -maskbfs\n"
        f"{stage}"
        "Ended at Thu Jan 1 10:00:04 UTC 2026\n"
        "@#@FSTIME 2026:01:01:10:00:01 recon-all N 5 e 3.00\n"
        "================= DONE =========================================================\n"
    )
    history = latest.replace("current", "previous").replace("Jan 1", "Dec 31")
    text = history + history + latest if case == "appended" else latest
    log = tmp_path / "recon-surf.log"
    log.write_text(text)
    report = tmp_path / "recon-surf_times.yaml"

    result = subprocess.run(
        [sys.executable, str(PARSER), "-i", str(log), "--time_units", "s"],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert log.read_text() == text
    data = yaml.safe_load(report.read_text())
    assert data["date"] == "Thu Jan 1 10:00:00 UTC 2026"
    assert data["subject_id"] == "current"
    startup, surface = data["recon-surf_commands"]
    qc, = startup["Starting up / no stage defined yet"]
    assert qc["cmd"].endswith("quick_qc.py --asegdkt_segfile example.mgz")
    assert qc["duration_s"] == 1.0
    recon, = surface["SURFACE"]
    assert recon["cmd"] == "recon-all -s current -maskbfs"
    assert recon["duration_s"] == 3.0
    assert recon["stages"] == ([] if case == "stage-less" else [{
        "stage_name": "Mask BFS",
        "start": "10:00:01",
        "stop": "10:00:04",
        "duration_s": 3.0,
    }])


@pytest.mark.parametrize("case", ["success", "failure", "partial", "reconstruction-failure"])
@pytest.mark.parametrize(("pipeline", "completion"), [
    ("recon-surf", "reconstruction"),
    ("recon-surfreg", "surface registration"),
])
def test_final_reporting_preserves_reconstruction_status(tmp_path, case, pipeline, completion):
    """Only report failures are nonfatal, and failed refreshes leave no YAML."""
    scripts = tmp_path / "subject" / "scripts"
    scripts.mkdir(parents=True)
    log = scripts / f"{pipeline}.log"
    log.write_text("reconstruction log\n")
    report = scripts / f"{pipeline}_times.yaml"
    report.write_text("date: previous run\n")
    done = scripts / f"{pipeline}.done"
    utils = tmp_path / "utils"
    utils.mkdir()
    (utils / PARSER.name).write_text(
        "import sys\n"
        "from pathlib import Path\n"
        f"case = {case!r}\n"
        "if case != 'failure':\n"
        "    Path(sys.argv[sys.argv.index('-o') + 1]).write_text('date: current run\\n')\n"
        "sys.exit(0 if case == 'success' else 1)\n"
    )
    tail = (FASTSURFER_HOME / "recon_surf" / f"{pipeline}.sh").read_text().split("# Create the Done File\n", 1)[1]
    script = (
        f"FASTSURFER_HOME={shlex.quote(str(FASTSURFER_HOME))}\n"
        f'source "$FASTSURFER_HOME/recon_surf/functions.sh" > /dev/null 2>&1\n'
        'timecmd=""\n'
        f"binpath={shlex.quote(str(tmp_path) + '/')}\n"
        f"python={shlex.quote(sys.executable)}\n"
        f"SUBJECTS_DIR={shlex.quote(str(tmp_path))}\n"
        "subject=subject\n"
        f"LF={shlex.quote(str(log))}\n"
        f"DoneFile={shlex.quote(str(done))}\n"
        "StartTime=start\nEndTime=end\ntRunHours=1\nVERSION=test\ninputargs=()\n"
        + ('RunIt "false" "$LF"\n' if case == "reconstruction-failure" else "")
        + tail
    )
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=30)

    if case == "reconstruction-failure":
        assert result.returncode != 0
        assert not done.exists()
        assert report.read_text() == "date: previous run\n"
        assert "finished without error" not in log.read_text()
    else:
        assert result.returncode == 0, result.stderr
        assert "SUBJECT subject" in done.read_text()
        assert "finished without error" in log.read_text()
        if case == "success":
            assert report.read_text() == "date: current run\n"
            assert "WARNING:" not in log.read_text()
        else:
            assert not report.exists()
            assert "WARNING: Timing report generation failed" in log.read_text()
            assert f"{completion} completed successfully" in result.stdout
