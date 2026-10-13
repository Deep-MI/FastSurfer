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


def gpu(capability, driver=580, index=0, memory=24.0):
    return launcher.Gpu(index, capability, driver, memory)


@pytest.mark.parametrize(
    ("capability", "driver", "prefix"),
    [
        ((8, 6), 580, "cu132"),  # Ampere, current driver
        ((12, 0), 590, "cu132"),  # Blackwell
        ((8, 6), 550, "cu126"),  # driver too old for CUDA 13
        ((6, 1), 580, "cu126"),  # Pascal, dropped by CUDA 13
        ((7, 0), 600, "cu126"),  # Volta
        (None, 580, "cu132"),  # capability unknown, the driver decides
        (None, 550, "cu126"),
    ],
)
def test_choose_build_gpu(capability, driver, prefix):
    build = launcher.choose_build([gpu(capability, driver)], "auto")
    assert (build.prefix, build.gpu) == (prefix, True)


def test_choose_build_explains_the_older_image():
    build = launcher.choose_build([gpu((8, 6), 550)], "auto")
    assert build.notes and "550" in build.notes[0]
    assert launcher.choose_build([gpu((6, 1))], "auto").notes == []


def test_choose_build_several_gpus():
    # the image has to run on every GPU used: an Ampere and a Pascal need the CUDA 12.6 image
    assert launcher.choose_build([gpu((8, 6)), gpu((6, 1), index=1)], "auto").prefix == "cu126"
    # a Blackwell and a Pascal have no image in common
    build = launcher.choose_build([gpu((12, 0)), gpu((6, 1), index=1)], "auto")
    assert build.prefix == "cpu" and "both GPUs" in build.notes[0]
    with pytest.raises(launcher.LauncherError):
        launcher.choose_build([gpu((12, 0)), gpu((6, 1), index=1)], "cuda")


@pytest.mark.parametrize(
    ("capability", "driver"),
    [
        ((12, 0), 550),  # Blackwell needs CUDA 13, whose driver is missing
        ((3, 7), 580),  # Kepler, no image supports it
        ((8, 6), 470),  # driver too old for every image
    ],
)
def test_choose_build_falls_back_to_cpu(capability, driver):
    build = launcher.choose_build([gpu(capability, driver)], "auto")
    assert (build.prefix, build.gpu) == ("cpu", False)
    assert build.notes
    with pytest.raises(launcher.LauncherError):
        launcher.choose_build([gpu(capability, driver)], "cuda")


def test_choose_build_cpu():
    assert launcher.choose_build([gpu((8, 6))], "cpu").prefix == "cpu"
    assert launcher.choose_build([], "auto").prefix == "cpu"
    with pytest.raises(launcher.LauncherError):
        launcher.choose_build([], "cuda")


def test_select_gpus():
    kepler, pascal, ampere = gpu((3, 7), index=0), gpu((6, 1), index=1), gpu((8, 6), index=2)
    # the newest GPU an image supports, shown alone so FastSurfer uses it as cuda:0
    build = launcher.select_gpus([kepler, pascal, ampere], brun=False)
    assert (build.prefix, build.devices) == ("cu132", [2])
    assert build.notes[0] == "Using GPU 2 (compute capability 8.6, 24 GB)."
    # a GPU with too little memory only when none has enough
    small = gpu((8, 9), index=3, memory=4.0)
    assert launcher.select_gpus([pascal, small], brun=False).devices == [1]
    assert launcher.select_gpus([small, gpu((7, 5), index=4, memory=6.0)], brun=False).devices == [4]
    # brun_fastsurfer.sh gets every GPU the chosen image supports
    assert launcher.select_gpus([kepler, pascal, ampere, gpu((7, 5), index=4)], brun=True).devices == [2, 4]
    assert launcher.select_gpus([kepler], brun=False) is None


def test_cuda_indices():
    assert launcher.cuda_indices("cuda:1") == [1]
    assert launcher.cuda_indices("cuda:0,1") == [0, 1]
    assert launcher.cuda_indices("cuda:0-2,5") == [0, 1, 2, 5]
    assert launcher.cuda_indices("cuda") is None and launcher.cuda_indices("auto") is None
    # an empty range would read as no GPU named, so a downward range is an error
    for value in ("cuda:first", "cuda:3-0"):
        with pytest.raises(launcher.LauncherError):
            launcher.cuda_indices(value)


def test_gpu_request():
    assert launcher.gpu_request([]) == ("auto", None)
    assert launcher.gpu_request(["--device", "cuda"]) == ("cuda", None)
    assert launcher.gpu_request(["--device", "cuda:1"]) == ("cuda", [1])
    # lists and ranges only with brun_fastsurfer.sh: run_fastsurfer.sh takes one GPU
    assert launcher.gpu_request(["--device", "cuda:0-2,5"], brun=True) == ("cuda", [0, 1, 2, 5])
    for value in ("cuda:0,1", "cuda:0-3"):
        for flag in ("--device", "--viewagg_device"):
            with pytest.raises(launcher.LauncherError, match="--brun"):
                launcher.gpu_request([flag, value])
    assert launcher.gpu_request(["--device", "cpu"]) == ("cpu", None)
    # the view aggregation can need a GPU while the segmentation runs on the CPU
    assert launcher.gpu_request(["--device", "cpu", "--viewagg_device", "cuda:2"]) == ("cuda", [2])
    # the last of repeated flags counts, as in the scripts
    assert launcher.gpu_request(["--device", "cpu", "--device", "cuda:1"]) == ("cuda", [1])
    assert launcher.gpu_request(["--device", "cuda:1", "--device", "cpu"]) == ("cpu", None)


def test_plan_build():
    pascal, ampere = gpu((6, 1), index=0), gpu((8, 6), index=1)
    # not named: the best GPU
    assert launcher.plan_build([], [pascal, ampere], brun=False).devices == [1]
    # named: used as named, with every GPU visible so the index keeps its meaning
    build = launcher.plan_build(["--device", "cuda:0"], [pascal, ampere], brun=False)
    assert (build.prefix, build.devices) == ("cu126", None)
    with pytest.raises(launcher.LauncherError):
        launcher.plan_build(["--device", "cuda:2"], [pascal, ampere], brun=False)
    # a named GPU that is too old: stop, with the reason
    with pytest.raises(launcher.LauncherError, match="too old"):
        launcher.plan_build(["--device", "cuda:0"], [gpu((3, 7)), ampere], brun=False)
    assert launcher.plan_build([], [gpu((3, 7))], brun=False).prefix == "cpu"


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

    # subject-specific outputs, e.g. a subject's own --sd: created and shared like the general ones
    own_sd, log = tmp_path / "other" / "sd", tmp_path / "logs" / "sub2.log"
    subjects.write_text(f"sub2={a / 't1.mgz'} --sd {own_sd} --seg_log {log}\n")
    _, mounts = launcher.plan_paths(["--subjects_list", str(subjects)], tmp_path)
    assert {own_sd, log.parent} <= set(launcher.minimal_mounts(mounts))
    assert own_sd.is_dir() and log.parent.is_dir()

    out, mounts = launcher.plan_paths(["--subjects", "sub1=a/t1.mgz", "--sd", "out"], tmp_path)
    assert out[:2] == ["--subjects", f"sub1={a / 't1.mgz'}"]

    # relative paths in a list cannot be resolved inside the container, and brun_fastsurfer.sh does not
    # expand ~ in a list
    for line in ("sub1=a/t1.mgz\n", "sub1=~/t1.mgz\n"):
        subjects.write_text(line)
        with pytest.raises(launcher.LauncherError):
            launcher.plan_paths(["--subjects_list", str(subjects)], tmp_path)


def test_commands(tmp_path):
    args = ["--t1", str(tmp_path / "t1.mgz"), "--sd", str(tmp_path)]
    docker = launcher.docker_command("img:tag", True, [tmp_path], "run_fastsurfer.sh", args, [], "1000:1000")
    assert docker == ["docker", "run", "--rm", "--user", "1000:1000", "--gpus", "all",
                      "--env", "CUDA_DEVICE_ORDER=PCI_BUS_ID", "-v", f"{tmp_path}:{tmp_path}", "img:tag"] + args
    # selected GPUs, quoted so Docker does not split the list
    selected = launcher.docker_command("img:tag", True, [], "run_fastsurfer.sh", [], [], "1:1", devices=[1, 3])
    assert selected[5:7] == ["--gpus", '"device=1,3"']
    sif_gpu = launcher.apptainer_command("apptainer", tmp_path / "fs.sif", True, [], "run_fastsurfer.sh", [], [],
                                         devices=[2])
    assert sif_gpu[2:8] == ["--nv", "--env", "CUDA_DEVICE_ORDER=PCI_BUS_ID", "--env", "CUDA_VISIBLE_DEVICES=2",
                            "--no-mount"]
    emulated = launcher.docker_command("img:tag", False, [tmp_path], "run_fastsurfer.sh", args, [], "501:20", True)
    assert emulated[:7] == ["docker", "run", "--rm", "--user", "501:20", "--platform", "linux/amd64"]
    brun = launcher.docker_command("img:tag", False, [tmp_path], "brun_fastsurfer.sh", args, [], "1000:1000")
    assert brun[-len(args) - 4:-len(args)] == ["--entrypoint", "/fastsurfer/tools/Docker/entrypoint.sh", "img:tag",
                                               "/fastsurfer/brun_fastsurfer.sh"]
    sif = tmp_path / "fs.sif"
    apptainer = launcher.apptainer_command("apptainer", sif, False, [tmp_path], "run_fastsurfer.sh", args, [])
    assert apptainer == ["apptainer", "exec", "--no-mount", "home,cwd", "-e", "-B", str(tmp_path), str(sif),
                         "/fastsurfer/run_fastsurfer.sh"] + args
