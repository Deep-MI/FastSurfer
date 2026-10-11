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

"""Test the image choice and the path handling of tools/fastsurfer_container.py."""

import importlib.util
import sys
from pathlib import Path

import pytest

# a script rather than a module of FastSurfer, so it is loaded from its path; dataclasses need it
# registered in sys.modules
_SCRIPT = Path(__file__).parent.parent.parent / "tools" / "fastsurfer_container.py"
_spec = importlib.util.spec_from_file_location("fastsurfer_container", _SCRIPT)
launcher = sys.modules.setdefault(_spec.name, importlib.util.module_from_spec(_spec))
_spec.loader.exec_module(launcher)


@pytest.mark.parametrize(
    ("gpu", "prefix"),
    [
        (((8, 6), 580), "cu132"),  # Ampere, current driver
        (((12, 0), 590), "cu132"),  # Blackwell
        (((8, 6), 550), "cu126"),  # driver too old for CUDA 13
        (((6, 1), 580), "cu126"),  # Pascal, dropped by CUDA 13
        (((7, 0), 600), "cu126"),  # Volta
        ((None, 580), "cu132"),  # capability unknown, the driver decides
        ((None, 550), "cu126"),
    ],
)
def test_choose_build_gpu(gpu, prefix):
    build = launcher.choose_build(gpu, "auto")
    assert (build.prefix, build.gpu) == (prefix, True)


def test_choose_build_explains_the_older_image():
    build = launcher.choose_build(((8, 6), 550), "auto")
    assert build.notes and "550" in build.notes[0]
    assert launcher.choose_build(((6, 1), 580), "auto").notes == []


@pytest.mark.parametrize(
    "gpu",
    [
        ((12, 0), 550),  # Blackwell needs CUDA 13, whose driver is missing
        ((3, 7), 580),  # Kepler, no image supports it
        ((8, 6), 470),  # driver too old for every image
    ],
)
def test_choose_build_falls_back_to_cpu(gpu):
    build = launcher.choose_build(gpu, "auto")
    assert (build.prefix, build.gpu) == ("cpu", False)
    assert build.notes
    with pytest.raises(launcher.LauncherError):
        launcher.choose_build(gpu, "cuda")


def test_choose_build_cpu():
    assert launcher.choose_build(((8, 6), 580), "cpu").prefix == "cpu"
    assert launcher.choose_build(None, "auto").prefix == "cpu"
    with pytest.raises(launcher.LauncherError):
        launcher.choose_build(None, "cuda:0")


def test_newest_version():
    tags = ["cu132-v2.6.0", "cu132-v2.10.1", "cu132-v2.9.3", "cu126-v2.11.0", "cu132-v2.11.0-dev0", "latest",
            "cpu-latest", "cu132-v2.10.1_abc123"]
    # numeric, not alphabetic: 2.10 is newer than 2.9; other images and development builds do not count
    assert launcher.newest_version(tags, "cu132") == "2.10.1"
    assert launcher.newest_version(tags, "cu126") == "2.11.0"
    assert launcher.newest_version(tags, "cpu") is None


def test_official_image_version():
    assert launcher.official_image_version("deepmi/fastsurfer:cu118-v2.2.0") == "2.2.0"
    assert launcher.official_image_version("docker.io/deepmi/fastsurfer:cpu-v2.6.0") == "2.6.0"
    assert launcher.official_image_version("/images/fastsurfer-cu118-v2.2.0.sif") == "2.2.0"
    for image in ("deepmi/fastsurfer:latest", "my_fastsurfer:cu132-v2.6.0", "fastsurfer-myimage.sif",
                  "deepmi/fastsurfer:cu132-v2.7.0-dev0"):
        assert launcher.official_image_version(image) is None


def test_age_note():
    releases = [(2, 2, 0), (2, 3, 3), (2, 4, 2), (2, 5, 4), (2, 6, 0)]
    assert launcher.age_note("2.6.0", releases, gpu_unsupported=True) is None
    assert launcher.age_note("2.6.0", [], gpu_unsupported=True) is None  # Docker Hub unreachable
    level, message = launcher.age_note("2.5.4", releases, gpu_unsupported=False)
    assert level == "INFO" and "1 release older" in message
    # an old GPU image instead of the newest release on the CPU: recommend against it
    level, message = launcher.age_note("2.2.0", releases, gpu_unsupported=True, flag="--image")
    assert level == "WARNING" and "4 releases older than the newest, 2.6.0" in message and "CPU" in message


def test_local_tags_apptainer(tmp_path):
    for name in ("fastsurfer-cpu-v2.5.4.sif", "fastsurfer-cpu-v2.6.0.sif", "fastsurfer-myimage.sif", "other.sif"):
        (tmp_path / name).write_bytes(b"")
    tags = launcher.local_tags("apptainer", tmp_path)
    assert sorted(tags) == ["cpu-v2.5.4", "cpu-v2.6.0", "myimage"]
    assert launcher.newest_version(tags, "cpu") == "2.6.0"


def test_plan_paths_run(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "t1.mgz").write_bytes(b"")
    license_file = tmp_path / "license.txt"
    license_file.write_text("")
    args = ["--t1", "data/t1.mgz", "--sid", "data", "--sd", "data/out", "--fs_license", str(license_file),
            "--conformed_name", "elsewhere/orig.mgz", "--3T"]
    out, mounts = launcher.plan_paths(args, tmp_path)
    assert out == ["--t1", str(data / "t1.mgz"), "--sid", "data", "--sd", str(data / "out"),
                   "--fs_license", str(license_file), "--conformed_name", str(tmp_path / "elsewhere/orig.mgz"),
                   "--3T"]
    # the output directories are created, because a mount needs them to exist
    assert (data / "out").is_dir() and (tmp_path / "elsewhere").is_dir()
    # data/out is inside data, and everything is inside tmp_path once the license is shared
    assert launcher.minimal_mounts(mounts) == [tmp_path]
    assert launcher.minimal_mounts([data / "out", data, tmp_path / "elsewhere"]) == [data, tmp_path / "elsewhere"]


def test_plan_paths_missing_input(tmp_path):
    with pytest.raises(launcher.LauncherError):
        launcher.plan_paths(["--t1", "missing.mgz", "--sd", "out"], tmp_path)


def test_plan_paths_brun(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "t1.mgz").write_bytes(b"")
    (b / "t2.nii.gz").write_bytes(b"")
    subjects = tmp_path / "list" / "subjects.txt"
    subjects.parent.mkdir()
    subjects.write_text(f"# comment\n\nsub1={a / 't1.mgz'} --t2 '{b / 't2.nii.gz'}'\n")
    _, mounts = launcher.plan_paths(["--subjects_list", str(subjects), "--sd", str(tmp_path / "out")], tmp_path)
    assert set(launcher.minimal_mounts(mounts)) == {a, b, subjects.parent, tmp_path / "out"}

    out, mounts = launcher.plan_paths(["--subjects", "sub1=a/t1.mgz", "--sd", "out"], tmp_path)
    assert out[:2] == ["--subjects", f"sub1={a / 't1.mgz'}"]

    # relative paths in a list cannot be resolved inside the container
    subjects.write_text("sub1=a/t1.mgz\n")
    with pytest.raises(launcher.LauncherError):
        launcher.plan_paths(["--subjects_list", str(subjects)], tmp_path)


def test_commands(tmp_path):
    args = ["--t1", str(tmp_path / "t1.mgz"), "--sd", str(tmp_path)]
    docker = launcher.docker_command("img:tag", True, [tmp_path], "run_fastsurfer.sh", args, [], "1000:1000")
    assert docker == ["docker", "run", "--rm", "--user", "1000:1000", "--gpus", "all",
                      "-v", f"{tmp_path}:{tmp_path}", "img:tag"] + args
    emulated = launcher.docker_command("img:tag", False, [tmp_path], "run_fastsurfer.sh", args, [], "501:20", True)
    assert emulated[:7] == ["docker", "run", "--rm", "--user", "501:20", "--platform", "linux/amd64"]
    brun = launcher.docker_command("img:tag", False, [tmp_path], "brun_fastsurfer.sh", args, [], "1000:1000")
    assert brun[-len(args) - 4:-len(args)] == ["--entrypoint", "/fastsurfer/tools/Docker/entrypoint.sh", "img:tag",
                                               "/fastsurfer/brun_fastsurfer.sh"]
    sif = tmp_path / "fs.sif"
    apptainer = launcher.apptainer_command("apptainer", sif, False, [tmp_path], "run_fastsurfer.sh", args, [])
    assert apptainer == ["apptainer", "exec", "--no-mount", "home,cwd", "-e", "-B", str(tmp_path), str(sif),
                         "/fastsurfer/run_fastsurfer.sh"] + args
