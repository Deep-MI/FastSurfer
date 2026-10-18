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

"""Test the image choice and the path handling of the launcher, tools/fastsurfer."""

import importlib.machinery
import importlib.util
import platform
import subprocess
import sys
from pathlib import Path

import pytest

# a command rather than a module of FastSurfer, without the .py suffix, so it is loaded from its path with
# an explicit loader; dataclasses need it registered in sys.modules
_SCRIPT = str(Path(__file__).parent.parent.parent / "tools" / "fastsurfer")
_spec = importlib.util.spec_from_file_location(
    "fastsurfer_launcher", _SCRIPT, loader=importlib.machinery.SourceFileLoader("fastsurfer_launcher", _SCRIPT))
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
            with pytest.raises(launcher.LauncherError, match="brun_fastsurfer.sh"):
                launcher.gpu_request([flag, value])
    assert launcher.gpu_request(["--device", "cpu"]) == ("cpu", None)
    # the view aggregation can need a GPU while the segmentation runs on the CPU
    assert launcher.gpu_request(["--device", "cpu", "--viewagg_device", "cuda:2"]) == ("cuda", [2])
    # the last of repeated flags counts, as in the scripts
    assert launcher.gpu_request(["--device", "cpu", "--device", "cuda:1"]) == ("cuda", [1])
    assert launcher.gpu_request(["--device", "cuda:1", "--device", "cpu"]) == ("cpu", None)
    # a GPU named only for the view aggregation: the segmentation runs on the first GPU, which counts too
    assert launcher.gpu_request(["--viewagg_device", "cuda:2"]) == ("cuda", [0, 2])


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


def test_plan_build_for_type():
    pascal, blackwell = gpu((6, 1), index=0), gpu((12, 0), index=1)
    # a pinned release image: the best GPU among those its type supports, shown alone
    build = launcher.plan_build_for_type([], [pascal, blackwell], False, "cu132")
    assert (build.gpu, build.devices) == (True, [1]) and build.notes[0].startswith("Using GPU 1")
    assert launcher.plan_build_for_type([], [pascal, blackwell], False, "cu126").devices == [0]
    # an older type is judged by the driver; with --brun, every GPU the type supports
    assert launcher.plan_build_for_type([], [pascal, blackwell], True, "cu118").devices == [0, 1]
    # a named GPU is used as named, or stops the launcher if the type does not support it
    assert launcher.plan_build_for_type(["--device", "cuda:1"], [pascal, blackwell], False, "cu132").devices is None
    with pytest.raises(launcher.LauncherError, match="do not support GPU 0"):
        launcher.plan_build_for_type(["--device", "cuda:0"], [pascal, blackwell], False, "cu132")
    # no supported GPU: the CPU, or a stop when a GPU was requested
    assert launcher.plan_build_for_type([], [pascal], False, "cu132").gpu is False
    with pytest.raises(launcher.LauncherError, match="support none"):
        launcher.plan_build_for_type(["--device", "cuda"], [pascal], False, "cu132")
    assert launcher.plan_build_for_type([], [pascal, blackwell], False, "cpu").gpu is False


def test_newest_version():
    tags = ["cu132-v2.6.0", "cu132-v2.10.1", "cu132-v2.9.3", "cu126-v2.11.0", "cu132-v2.11.0-dev0", "latest",
            "cpu-latest", "cu132-v2.10.1_abc123"]
    # numeric, not alphabetic: 2.10 is newer than 2.9; other images and development builds do not count
    assert launcher.newest_version(tags, "cu132") == "2.10.1"
    assert launcher.newest_version(tags, "cu126") == "2.11.0"
    assert launcher.newest_version(tags, "cpu") is None


def test_age_note():
    releases = [(2, 2, 0), (2, 3, 3), (2, 4, 2), (2, 5, 4), (2, 6, 0)]
    assert launcher.age_note("2.6.0", releases, gpu_unsupported=True) is None
    assert launcher.age_note("2.6.0", [], gpu_unsupported=True) is None  # Docker Hub unreachable
    level, message = launcher.age_note("2.5.4", releases, gpu_unsupported=False)
    assert level == "INFO" and "1 release older" in message
    # an old GPU image instead of the newest release on the CPU: recommend against it
    level, message = launcher.age_note("2.2.0", releases, gpu_unsupported=True)
    assert level == "WARNING" and "4 releases older than the newest, 2.6.0" in message and "CPU" in message


@pytest.mark.parametrize(
    ("spec", "parsed"),
    [
        (None, ("auto", "")),
        ("2.6.0", ("version", "2.6.0")),
        ("cu132-v2.6.0", ("tag", "cu132-v2.6.0")),
        ("cu118-v2.4.2", ("tag", "cu118-v2.4.2")),
        ("deepmi/fastsurfer:cu132-v2.6.0", ("release", "cu132-v2.6.0")),
        ("docker.io/deepmi/fastsurfer:cpu-v2.6.0", ("release", "cpu-v2.6.0")),
        ("deepmi/fastsurfer:latest", ("docker", "deepmi/fastsurfer:latest")),
        ("my_fastsurfer:cu132-v2.6.0", ("docker", "my_fastsurfer:cu132-v2.6.0")),
        ("/images/fastsurfer-cu132-v2.6.0.sif", ("sif", "/images/fastsurfer-cu132-v2.6.0.sif")),
    ],
)
def test_parse_image_spec(spec, parsed):
    assert launcher.parse_image_spec(spec) == parsed


def test_local_sif_images(tmp_path):
    for name in ("fastsurfer-cpu-v2.5.4.sif", "fastsurfer-cpu-v2.6.0.sif", "other.sif", ".fastsurfer-x.1.partial.sif"):
        (tmp_path / name).write_bytes(b"")
    (tmp_path / "fastsurfer-cpu-v2.6.0.sif.digest").write_text("sha256:c\n")
    assert launcher.local_sif_images(tmp_path) == {"cpu-v2.5.4": "", "cpu-v2.6.0": "sha256:c"}
    assert launcher.local_sif_images(tmp_path / "missing") == {}


HUB = {"cu132-v2.5.4": {"sha256:b"}, "cu132-v2.6.0": {"sha256:a", "sha256:a64"}, "cu126-v2.6.0": {"sha256:d"},
       "cpu-v2.6.0": {"sha256:c"}}
BOTH = {"apptainer", "docker"}


def decide(tmp_path, kind="auto", value="", runtime="auto", installed=BOTH, docker=None, sif=None, hub=HUB,
           update=False, local_id=None, gpus=None):
    return launcher.decide_image(kind, value, "cu132", runtime, installed, docker, sif or {}, hub, tmp_path,
                                 update, local_id, gpus)


def test_decide_image_nothing_installed(tmp_path):
    # the newest release, built for Apptainer, remembering its digest
    plan = decide(tmp_path)
    assert (plan.runtime, plan.image) == ("apptainer", str(tmp_path / "fastsurfer-cu132-v2.6.0.sif"))
    assert plan.build_from == "docker://deepmi/fastsurfer:cu132-v2.6.0" and plan.record in HUB["cu132-v2.6.0"]
    assert "--image cu132-v2.6.0" in plan.notes[0][1]
    assert decide(tmp_path, installed={"docker"}).image == "deepmi/fastsurfer:cu132-v2.6.0"
    with pytest.raises(launcher.LauncherError, match="cannot be reached"):
        decide(tmp_path, hub=None)


def test_decide_image_one_installed(tmp_path):
    # the installed image, where it is, and a note about the newer release
    plan = decide(tmp_path, docker={"cu132-v2.5.4": "sha256:b"})
    assert (plan.runtime, plan.image, plan.pull) == ("docker", "deepmi/fastsurfer:cu132-v2.5.4", False)
    assert any("2.6.0 is available" in note for _, note in plan.notes)
    # offline, it is used all the same
    assert decide(tmp_path, docker={"cu132-v2.5.4": "sha256:b"}, hub=None).image.endswith("cu132-v2.5.4")
    # the same version in both places counts once, and runs with Apptainer
    plan = decide(tmp_path, docker={"cu132-v2.6.0": "sha256:a"}, sif={"cu132-v2.6.0": "sha256:a"})
    assert plan.runtime == "apptainer" and plan.build_from is None
    # a release pulled into Docker is converted from there for Apptainer, not downloaded again
    plan = decide(tmp_path, runtime="apptainer", docker={"cu132-v2.6.0": "sha256:a"})
    assert plan.build_from == "docker-daemon://deepmi/fastsurfer:cu132-v2.6.0"
    # but not a local build tagged like the release
    assert decide(tmp_path, runtime="apptainer", docker={"cu132-v2.6.0": ""}).build_from.startswith("docker://")
    # images of another type do not count
    assert decide(tmp_path, docker={"cpu-v2.5.4": "sha256:x"}).tag == "cu132-v2.6.0"


def test_image_type_runs():
    turing = [gpu((7, 5), driver=580)]
    assert launcher.image_type_runs("cu132", turing) and launcher.image_type_runs("cu126", turing)
    # older CUDA types are judged by the driver alone
    assert launcher.image_type_runs("cu128", turing) and launcher.image_type_runs("cu118", turing)
    assert not launcher.image_type_runs("cu128", [gpu((7, 5), driver=470)])
    assert not launcher.image_type_runs("cu132", [gpu((6, 1))])  # Pascal, dropped by CUDA 13
    # the CPU image only for a run on the CPU, and no AMD or unknown types on an NVIDIA GPU
    assert not launcher.image_type_runs("cpu", turing) and launcher.image_type_runs("cpu", [])
    assert not launcher.image_type_runs("rocm7.2", turing) and not launcher.image_type_runs("gpu", turing)


def test_decide_image_older_types(tmp_path):
    # images of older types that run on this GPU count too, as on a machine with earlier releases
    turing = [gpu((7, 5), driver=580)]
    installed = {"cu128-v2.5.4": "sha256:1", "cu126-v2.5.4": "sha256:2", "cpu-v2.5.4": "sha256:3"}
    hub = {**HUB, "cu128-v2.5.4": {"sha256:1"}, "cu126-v2.5.4": {"sha256:2"}, "cpu-v2.5.4": {"sha256:3"}}
    # listed newest first; the newest release is not installed, so --update is offered
    with pytest.raises(launcher.LauncherError, match="cu128-v2.5.4.*cu126-v2.5.4.*--update"):
        decide(tmp_path, docker=installed, hub=hub, gpus=turing)
    # one of them alone is used, with a note about the newest release
    plan = decide(tmp_path, docker={"cu128-v2.5.4": "sha256:1"}, hub=hub, gpus=turing)
    assert plan.tag == "cu128-v2.5.4" and any("2.6.0 is available" in note for _, note in plan.notes)
    # a run on the CPU only counts the CPU image
    assert decide(tmp_path, docker=installed, hub=hub, gpus=[]).tag == "cpu-v2.5.4"


def test_decide_image_several_installed(tmp_path):
    # the newest release is among them: suggested first, and no --update, which would download it again
    with pytest.raises(launcher.LauncherError, match="Several") as stop:
        decide(tmp_path, docker={"cu132-v2.5.4": "sha256:b"}, sif={"cu132-v2.6.0": ""})
    message = str(stop.value)
    assert message.index("cu132-v2.6.0") < message.index("cu132-v2.5.4")
    assert "--image cu132-v2.6.0, the newest release" in message and "--update" not in message
    # --update takes the newest, and lists the older ones with how to remove them
    plan = decide(tmp_path, docker={"cu132-v2.5.4": "sha256:b"}, sif={"cu132-v2.6.0": ""}, update=True)
    assert plan.build_from is not None and plan.tag == "cu132-v2.6.0"
    assert any("docker image rm deepmi/fastsurfer:cu132-v2.5.4" in note for _, note in plan.notes)
    # a runtime named with --runtime counts only its own images while it has one
    assert decide(tmp_path, runtime="apptainer", docker={"cu132-v2.5.4": "sha256:b"},
                  sif={"cu132-v2.6.0": ""}).tag == "cu132-v2.6.0"
    assert decide(tmp_path, runtime="docker", docker={"cu132-v2.5.4": "sha256:b"},
                  sif={"cu132-v2.6.0": ""}).tag == "cu132-v2.5.4"
    # a program other than FastSurfer (--exec) takes the newest instead of stopping, without study advice
    plan = launcher.decide_image("auto", "", "cu132", "auto", BOTH, {"cu132-v2.5.4": "sha256:b"}, {"cu132-v2.6.0": ""},
                                 HUB, tmp_path, False, other_program=True)
    assert plan.tag == "cu132-v2.6.0" and any("uses the newest" in note for _, note in plan.notes)
    assert not any("For a study" in note for _, note in plan.notes)


def test_decide_image_digest(tmp_path):
    # a local build tagged like a release: stop, unless named in full or replaced with --update
    local_build = {"cu132-v2.6.0": ""}
    with pytest.raises(launcher.LauncherError, match="differs from the release"):
        decide(tmp_path, docker=local_build)
    with pytest.raises(launcher.LauncherError, match="differs from the release"):
        decide(tmp_path, kind="tag", value="cu132-v2.6.0", docker=local_build)
    assert decide(tmp_path, kind="release", value="cu132-v2.6.0", docker=local_build).runtime == "docker"
    assert decide(tmp_path, docker=local_build, update=True).pull
    # either digest of the release matches, and a .sif without a recorded digest is not judged
    assert decide(tmp_path, docker={"cu132-v2.6.0": "sha256:a64"}).runtime == "docker"
    assert decide(tmp_path, sif={"cu132-v2.6.0": ""}).build_from is None
    with pytest.raises(launcher.LauncherError, match="differs from the release"):
        decide(tmp_path, sif={"cu132-v2.6.0": "sha256:other"})


def test_decide_image_named(tmp_path):
    # a version the type of this machine was not released in: the images of that version are listed
    with pytest.raises(launcher.LauncherError, match="cpu-v2.6.0"):
        decide(tmp_path, kind="version", value="2.6.0", hub={"cpu-v2.6.0": {"sha256:c"}})
    with pytest.raises(launcher.LauncherError, match="no Docker images of FastSurfer 9.9.9"):
        decide(tmp_path, kind="tag", value="cu132-v9.9.9")
    with pytest.raises(launcher.LauncherError, match="Docker"):
        decide(tmp_path, kind="sif", value="x.sif", runtime="docker")
    # an own Docker image runs with Docker, or with Apptainer when asked, converted from the local Docker
    assert decide(tmp_path, kind="docker", value="my_fs:cu132", local_id="sha256:1").runtime == "docker"
    plan = decide(tmp_path, kind="docker", value="my_fs:cu132", runtime="apptainer", local_id="sha256:1")
    assert (plan.build_from, plan.record) == ("docker-daemon://my_fs:cu132", "sha256:1")
    assert plan.image == str(tmp_path / "docker-my_fs-cu132.sif")
    (tmp_path / "docker-my_fs-cu132.sif").write_bytes(b"")
    (tmp_path / "docker-my_fs-cu132.sif.digest").write_text("sha256:1\n")
    own = {"kind": "docker", "value": "my_fs:cu132", "runtime": "apptainer"}
    assert decide(tmp_path, **own, local_id="sha256:1").build_from is None
    # a changed local Docker image is converted again
    assert decide(tmp_path, **own, local_id="sha256:2").build_from
    # one whose name ends like a release is not taken for the installed release image, nor replaces it
    (tmp_path / "fastsurfer-cu132-v2.6.0.sif").write_bytes(b"")
    plan = decide(tmp_path, kind="docker", value="myrepo/fastsurfer:cu132-v2.6.0", runtime="apptainer")
    assert plan.image == str(tmp_path / "docker-myrepo_fastsurfer-cu132-v2.6.0.sif") and plan.build_from


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


def test_plan_paths_long(tmp_path):
    # long_fastsurfer.sh: several T1s (and T2s) after one flag, one per time point
    data = tmp_path / "data"
    data.mkdir()
    for name in ("tp1.mgz", "tp2.mgz", "t2_1.mgz", "t2_2.mgz"):
        (data / name).write_bytes(b"")
    args = ["--tid", "person", "--t1s", "data/tp1.mgz", str(data / "tp2.mgz"), "--tpids", "tp1", "tp2",
            "--t2s", "data/t2_1.mgz", "data/t2_2.mgz", "--sd", "out", "--log", "logs/long.log"]
    out, mounts = launcher.plan_paths(args, tmp_path)
    assert out == ["--tid", "person", "--t1s", str(data / "tp1.mgz"), str(data / "tp2.mgz"), "--tpids", "tp1", "tp2",
                   "--t2s", str(data / "t2_1.mgz"), str(data / "t2_2.mgz"), "--sd", str(tmp_path / "out"),
                   "--log", str(tmp_path / "logs/long.log")]
    assert set(launcher.minimal_mounts(mounts)) == {data, tmp_path / "out", tmp_path / "logs"}
    # every time point has to exist, not only the first
    with pytest.raises(launcher.LauncherError, match="tp3.mgz"):
        launcher.plan_paths(["--t1s", "data/tp1.mgz", "data/tp3.mgz", "--sd", "out"], tmp_path)


def test_exec_mounts(tmp_path):
    work, data, results = tmp_path / "work", tmp_path / "data", tmp_path / "results"
    for directory in (work, data, results):
        directory.mkdir()
    (data / "in.mgz").write_bytes(b"")
    args = ["-v", str(data / "in.mgz"), f"--out={results / 'new.nii.gz'}", "relative.mgz", "2",
            str(tmp_path / "missing" / "x.mgz")]
    mounts = launcher.minimal_mounts(launcher.exec_mounts(args, work))
    # the current directory, the directory of an input, and that of an output in --flag=path form; a path
    # whose directory does not exist is not taken for one, and relative paths stay in the current directory
    assert mounts == sorted([work, data, results])


def test_split_exec():
    argv, program_args = launcher.split_exec(["--runtime", "docker", "--exec", "mri_convert", "--help", "--image", "x"])
    options, rest = launcher.make_parser().parse_known_args(argv)
    # the program's own flags, even the launcher's --help and --image, reach the program
    assert (options.runtime, options.exec, rest) == ("docker", "mri_convert", [])
    assert program_args == ["--help", "--image", "x"]
    # the same with --exec=<program>, also without arguments for it
    argv, program_args = launcher.split_exec(["--exec=nib-convert", "in.mgz", "out.nii.gz"])
    assert (launcher.make_parser().parse_known_args(argv)[0].exec, program_args) == ("nib-convert",
                                                                                   ["in.mgz", "out.nii.gz"])
    assert launcher.split_exec(["--exec=bash"]) == (["--exec=bash"], [])


def test_fastsurfer_script():
    # the FastSurfer scripts by name or by their path in the image; anything else is another program
    assert launcher.fastsurfer_script("brun_fastsurfer.sh") == launcher.BRUN_SCRIPT
    assert launcher.fastsurfer_script("/fastsurfer/long_fastsurfer.sh") == "/fastsurfer/long_fastsurfer.sh"
    assert launcher.fastsurfer_script("run_fastsurfer.sh") == launcher.RUN_SCRIPT
    for other in ("nib-convert", "/somewhere/else/brun_fastsurfer.sh", "recon-surf.sh"):
        assert launcher.fastsurfer_script(other) is None


def test_link_directories(tmp_path):
    # a chain of links, as in a DataLad dataset: the link, a relative link in another directory, the file
    for name in ("data", "annex/key", "objects/abc"):
        (tmp_path / name).mkdir(parents=True)
    target = tmp_path / "objects" / "abc" / "t1.mgz"
    target.write_bytes(b"")
    (tmp_path / "annex" / "key" / "t1.mgz").symlink_to("../../objects/abc/t1.mgz")
    (tmp_path / "data" / "t1.mgz").symlink_to(tmp_path / "annex" / "key" / "t1.mgz")
    directories = launcher._link_directories(tmp_path / "data" / "t1.mgz")
    real = Path(str(tmp_path)).resolve()
    assert {Path(d).resolve() for d in directories} == {real / "data", real / "annex/key", real / "objects/abc"}
    # every directory on the way is shared, not a common parent of them
    _, mounts = launcher.plan_paths(["--t1", str(tmp_path / "data" / "t1.mgz")], tmp_path)
    assert tmp_path not in launcher.minimal_mounts(mounts)


@pytest.mark.parametrize(
    ("text", "too_old"),
    [
        ("apptainer version 1.3.4", False),
        ("singularity-ce version 3.11.4", False),
        ("singularity version 3.7.4", False),
        ("singularity version 3.5.3", True),
        ("SingularityCE ...", False),  # unreadable: no judgement
    ],
)
def test_runtime_too_old(text, too_old):
    assert (launcher.runtime_too_old(text) is not None) == too_old


def test_build_image(tmp_path):
    # a stand-in for `apptainer build <target> <source>` that writes the target, or fails without writing
    builder = tmp_path / "builder.py"
    builder.write_text("import sys\nif sys.argv[2] == 'fail': sys.exit(1)\nopen(sys.argv[1], 'w').write('image')\n")
    sif = tmp_path / "images" / "fastsurfer-cpu-v2.6.0.sif"
    assert launcher.build_image([sys.executable, str(builder), str(sif), "docker://image"])
    assert sif.read_text() == "image"
    # a failed build leaves neither the image nor a partial file behind
    failed = tmp_path / "images" / "fastsurfer-cu132-v2.6.0.sif"
    assert not launcher.build_image([sys.executable, str(builder), str(failed), "fail"])
    assert sorted(p.name for p in sif.parent.iterdir()) == [sif.name]
    # anonymous: the build reads an empty login file, which is removed afterwards
    reader = tmp_path / "reader.py"
    reader.write_text("import sys\nassert sys.argv[1] == '--authfile'\n"
                      "open(sys.argv[3], 'w').write(open(sys.argv[2]).read())\n")
    assert launcher.build_image([sys.executable, str(reader), str(failed), "docker://image"], anonymous=True)
    assert failed.read_text() == "{}\n"
    assert sorted(p.name for p in sif.parent.iterdir()) == sorted([sif.name, failed.name])


def amd_topology(tmp_path, nodes):
    """A stand-in for the kernel driver's /dev/kfd and topology: one node per (gfx_target_version, GiB)."""
    device, topology = tmp_path / "kfd", tmp_path / "nodes"
    device.write_text("")
    for number, (target, gib) in enumerate(nodes):
        node = topology / str(number)
        (node / "mem_banks" / "0").mkdir(parents=True)
        (node / "properties").write_text(f"cpu_cores_count 0\ngfx_target_version {target}\n")
        (node / "mem_banks" / "0" / "properties").write_text(f"heap_type 1\nsize_in_bytes {gib * 2**30}\n")
    return device, topology


def test_query_amd_gpus(tmp_path):
    # node 0 is the CPU, without a GPU architecture; the GPUs are numbered from 0 as ROCm does
    device, topology = amd_topology(tmp_path, [(0, 64), (100302, 8), (110000, 24)])
    gpus = launcher.query_amd_gpus(device, topology)
    assert [(g.index, g.arch, g.memory_gib) for g in gpus] == [(0, "gfx1032", 8), (1, "gfx1100", 24)]
    # without the kernel driver, no AMD GPU
    assert launcher.query_amd_gpus(tmp_path / "missing", topology) == []


@pytest.mark.parametrize(
    ("version", "override"),
    [
        ((11, 0, 0), ""),  # gfx1100, in the image
        ((9, 0, 10), ""),  # gfx90a, the stepping is hexadecimal
        ((10, 3, 2), "10.3.0"),  # gfx1032 (Radeon Pro W6600), runs as gfx1030
        ((9, 0, 12), "9.0.0"),  # gfx90c, an APU, runs as gfx900
        ((8, 0, 3), None),  # gfx803, no image of its family
    ],
)
def test_amd_override(version, override):
    assert launcher.amd_override(launcher.AmdGpu(0, version)) == override


def test_plan_amd_build():
    w6600, rx7900 = launcher.AmdGpu(0, (10, 3, 2), 8), launcher.AmdGpu(1, (11, 0, 0), 24)
    # the GPU with the most memory, shown alone to the container, without an override
    build = launcher.plan_amd_build([], [w6600, rx7900], False)
    assert (build.prefix, build.gpu, build.amd, build.devices, build.env) == ("rocm7.2", True, True, [1], [])
    assert "experimental" in build.notes[0]
    # a GPU of a supported family runs with the override, which says so
    alone = launcher.plan_amd_build([], [w6600], False)
    assert alone.env == ["HSA_OVERRIDE_GFX_VERSION=10.3.0"] and "gfx1030" in alone.notes[1]
    # with --brun, only GPUs that run with the same override, as it applies to the whole container
    assert launcher.plan_amd_build([], [w6600, rx7900], True).devices == [1]
    # named GPUs as named; one that does not exist stops
    assert launcher.plan_amd_build(["--device", "cuda:0"], [w6600, rx7900], False).devices is None
    with pytest.raises(launcher.LauncherError, match="no AMD GPU 2"):
        launcher.plan_amd_build(["--device", "cuda:2"], [w6600, rx7900], False)
    # no supported GPU: the CPU, or an error if a GPU was requested
    old = launcher.AmdGpu(0, (8, 0, 3), 4)
    assert not launcher.plan_amd_build([], [old], False).gpu
    with pytest.raises(launcher.LauncherError):
        launcher.plan_amd_build(["--device", "cuda"], [old], False)


@pytest.mark.parametrize(
    ("error", "system", "wsl", "advice"),
    [
        # the errors Docker prints, and what the launcher tells the user to do instead
        ("permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock: "
         "Get \"http://%2Fvar%2Frun%2Fdocker.sock/v1.45/info\": dial unix /var/run/docker.sock: connect: "
         "permission denied", "Linux", False, "docker group"),
        ("Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?",
         "Linux", False, "administrator starts it"),
        ("Cannot connect to the Docker daemon at unix:///Users/me/.docker/run/docker.sock. Is the docker daemon "
         "running?", "Darwin", False, "start Docker Desktop"),
        ("The command 'docker' could not be found in this WSL 2 distro.\nWe recommend to activate the WSL "
         "integration in Docker Desktop settings.", "Linux", True, "WSL integration"),
        ("error during connect: something else\n", "Linux", False, "does not work: error during connect"),
    ],
)
def test_docker_problem(error, system, wsl, advice):
    assert advice in launcher.docker_problem(error, system, wsl)


def test_plan_image_build_custom():
    nvidia, amd = [gpu((8, 6))], [launcher.AmdGpu(0, (11, 0, 0), 24)]
    # an own image on a machine with only AMD GPUs gets them, whatever its name
    build = launcher.plan_image_build("docker", "my_fastsurfer:dev", [], [], amd, False)
    assert (build.prefix, build.gpu, build.amd) == ("custom", True, True)
    assert launcher.plan_image_build("sif", "/images/own.sif", [], [], amd, False).amd
    # with an NVIDIA GPU, the NVIDIA one, unless the name says ROCm
    build = launcher.plan_image_build("docker", "my_fastsurfer:dev", [], nvidia, amd, False)
    assert (build.gpu, build.amd) == (True, False)
    assert launcher.plan_image_build("docker", "my_fastsurfer:rocm-dev", [], nvidia, amd, False).amd
    # --device cpu gives the container no GPU
    assert not launcher.plan_image_build("docker", "my_fastsurfer:dev", ["--device", "cpu"], [], amd, False).gpu


def test_plan_image_build_rocm_on_cpu():
    # an installed ROCm image also runs on the CPU, without an AMD GPU, with a note unless the CPU was asked for
    for kind, value in (("rocm", ""), ("tag", "rocm7.2-v2.6.0")):
        build = launcher.plan_image_build(kind, value, [], [gpu((8, 6))], [], False)
        assert (build.prefix, build.gpu, build.amd) == ("rocm7.2", False, False) and "CPU" in build.notes[0]
        build = launcher.plan_image_build(kind, value, ["--device", "cpu"], [], [], False)
        assert (build.prefix, build.gpu, build.notes) == ("rocm7.2", False, [])
    # a GPU requested for it without an AMD GPU stops
    with pytest.raises(launcher.LauncherError, match="No AMD GPU"):
        launcher.plan_image_build("rocm", "", ["--device", "cuda"], [], [], False)


def test_decide_image_rocm(tmp_path):
    # for an AMD GPU only the ROCm image counts among the installed ones, else the newest ROCm release
    hub = {**HUB, "rocm7.2-v2.6.0": {"sha256:r"}}
    installed = {"rocm7.2-v2.6.0": "sha256:r", "cu132-v2.6.0": "sha256:a"}
    plan = launcher.decide_image("auto", "", "rocm7.2", "docker", BOTH, installed, {}, hub, tmp_path, False)
    assert plan.image == "deepmi/fastsurfer:rocm7.2-v2.6.0" and not plan.pull
    plan = launcher.decide_image("auto", "", "rocm7.2", "docker", BOTH, {"cu132-v2.6.0": "sha256:a"}, {}, hub,
                                 tmp_path, False)
    assert plan.image == "deepmi/fastsurfer:rocm7.2-v2.6.0"


def test_parse_image_spec_rocm():
    assert launcher.parse_image_spec("rocm") == ("rocm", "")
    assert launcher.parse_image_spec("rocm7.2-v2.6.0") == ("tag", "rocm7.2-v2.6.0")


def test_amd_commands(tmp_path):
    env = ["HSA_OVERRIDE_GFX_VERSION=10.3.0"]
    docker = launcher.docker_command("img:tag", True, [], launcher.RUN_SCRIPT, [], [], "1000:1000", devices=[1],
                                     amd=True, env=env, groups=[44, 109])
    assert "--gpus" not in docker
    start = docker.index("--device")
    assert docker[start:start + 15] == ["--device", "/dev/kfd", "--device", "/dev/dri", "--ipc=host", "--shm-size",
                                        "8G", "--group-add", "44", "--group-add", "109", "--env", env[0], "--env",
                                        "HIP_VISIBLE_DEVICES=1"]
    apptainer = launcher.apptainer_command("apptainer", tmp_path / "fs.sif", True, [], launcher.RUN_SCRIPT, [], [],
                                           devices=[1], amd=True, env=env)
    assert apptainer[2:7] == ["--rocm", "--env", env[0], "--env", "HIP_VISIBLE_DEVICES=1"]


def macos_applications(tmp_path, names):
    """A stand-in for /Applications with FastSurfer packages, each a folder with the setup script."""
    for name in names:
        (tmp_path / name).mkdir()
        (tmp_path / name / launcher.MACOS_SETUP).write_text("")
    return tmp_path


def test_macos_packages(tmp_path):
    # releases by version; the app that opens the console and a development build do not count
    apps = macos_applications(tmp_path, ["FastSurfer2.5.4", "FastSurfer2.6.0", "FastSurfer2.7.0-dev0"])
    (tmp_path / "FastSurfer2.6.0.app").mkdir()
    assert launcher.macos_packages(apps) == {"2.5.4": apps / "FastSurfer2.5.4", "2.6.0": apps / "FastSurfer2.6.0"}
    # --image with the folder of a package names it directly
    dev = apps / "FastSurfer2.7.0-dev0"
    assert launcher.parse_image_spec(str(dev)) == ("macos", str(dev.resolve()))


def test_decide_macos_package(tmp_path):
    packages = {"2.5.4": tmp_path / "FastSurfer2.5.4", "2.6.0": tmp_path / "FastSurfer2.6.0"}
    # the same rules as for images: several installed stop the launcher, newest first
    with pytest.raises(launcher.LauncherError, match="2.6.0, 2.5.4.*--image 2.6.0"):
        launcher.decide_macos_package("auto", "", packages, "2.6.0")
    home, notes = launcher.decide_macos_package("version", "2.5.4", packages, "2.6.0")
    assert home == packages["2.5.4"] and "--image 2.5.4" in notes[0][1]
    # one installed runs, with a note on a newer release
    home, notes = launcher.decide_macos_package("auto", "", {"2.5.4": packages["2.5.4"]}, "2.6.0")
    assert home == packages["2.5.4"] and any("2.6.0 is available" in note for _, note in notes)
    # another program takes the newest, without study advice
    home, notes = launcher.decide_macos_package("auto", "", packages, "2.6.0", other_program=True)
    assert home == packages["2.6.0"] and not any("For a study" in note for _, note in notes)
    # none installed, a version not installed, and a container image: what to do instead
    with pytest.raises(launcher.LauncherError, match="--update"):
        launcher.decide_macos_package("auto", "", {}, "2.6.0")
    with pytest.raises(launcher.LauncherError, match="--update"):
        launcher.decide_macos_package("version", "2.4.2", packages, "2.6.0")
    with pytest.raises(launcher.LauncherError, match="--runtime docker"):
        launcher.decide_macos_package("tag", "cu132-v2.6.0", packages, "2.6.0")


def test_macos_command(tmp_path):
    # the package's setup script, then the program with its arguments unchanged, as bash runs it
    command = launcher.macos_command(tmp_path, "nib-convert", ["a b.mgz", "out.nii.gz"])
    setup = tmp_path / launcher.MACOS_SETUP
    setup.write_text("export STAND_IN=1\n")
    probe = launcher.macos_command(tmp_path, sys.executable, ["-c", "import os,sys; print(os.environ['STAND_IN'], "
                                                                    "sys.argv[1:])", "a b.mgz"])
    assert command[-3:] == ["nib-convert", "a b.mgz", "out.nii.gz"]
    assert subprocess.run(probe, capture_output=True, text=True).stdout == "1 ['a b.mgz']\n"


def test_commands(tmp_path):
    args = ["--t1", str(tmp_path / "t1.mgz"), "--sd", str(tmp_path)]
    run = launcher.RUN_SCRIPT
    docker = launcher.docker_command("img:tag", True, [tmp_path], run, args, [], "1000:1000")
    # --init, so Ctrl-C and a cancelled job reach FastSurfer
    init = ["docker", "run", "--rm", "--init", "--env", "TINI_KILL_PROCESS_GROUP=1"]
    # the host's time zone, on Linux
    zone = (["-v", "/etc/localtime:/etc/localtime:ro"]
            if platform.system() == "Linux" and Path("/etc/localtime").exists() else [])
    assert docker == init + ["--user", "1000:1000", "--gpus", "all", "--env", "CUDA_DEVICE_ORDER=PCI_BUS_ID"] + zone + [
        "-v", f"{tmp_path}:{tmp_path}", "img:tag"] + args
    # selected GPUs, quoted so Docker does not split the list
    selected = launcher.docker_command("img:tag", True, [], run, [], [], "1:1", devices=[1, 3])
    assert selected[8:10] == ["--gpus", '"device=1,3"']
    sif_gpu = launcher.apptainer_command("apptainer", tmp_path / "fs.sif", True, [], run, [], [], devices=[2])
    assert sif_gpu[2:8] == ["--nv", "--env", "CUDA_DEVICE_ORDER=PCI_BUS_ID", "--env", "CUDA_VISIBLE_DEVICES=2",
                            "--no-mount"]
    emulated = launcher.docker_command("img:tag", False, [tmp_path], run, args, [], "501:20", True)
    assert emulated[:10] == init + ["--user", "501:20", "--platform", "linux/amd64"]
    # the other scripts and programs run through the image's entrypoint, which sets up the environment
    for program in (launcher.BRUN_SCRIPT, launcher.SCRIPTS["long_fastsurfer.sh"], "nib-convert"):
        other = launcher.docker_command("img:tag", False, [tmp_path], program, args, [], "1000:1000")
        assert other[-len(args) - 4:-len(args)] == ["--entrypoint", "/fastsurfer/tools/Docker/entrypoint.sh",
                                                    "img:tag", program]
    sif = tmp_path / "fs.sif"
    apptainer = launcher.apptainer_command("apptainer", sif, False, [tmp_path], run, args, [])
    assert apptainer == ["apptainer", "exec", "--no-mount", "home,cwd", "-e", "-B", str(tmp_path), str(sif),
                         "/fastsurfer/run_fastsurfer.sh"] + args
    # --exec: in the current directory, and with Docker attached to the terminal if there is one
    exec_args = ["in.mgz", "out.nii.gz"]
    in_dir = launcher.apptainer_command("apptainer", sif, False, [tmp_path], "nib-convert", exec_args, [],
                                        workdir=tmp_path)
    assert in_dir[-6:] == ["--pwd", str(tmp_path), str(sif), "nib-convert"] + exec_args
    docker_exec = launcher.docker_command("img:tag", False, [tmp_path], "bash", [], [], "1:1", workdir=tmp_path,
                                          tty=True)
    assert docker_exec[8] == "-it"
    assert docker_exec[docker_exec.index("-w") + 1] == str(tmp_path)
    # input from a pipe or file is passed on without a terminal, which Docker refuses for such input
    piped = launcher.docker_command("img:tag", False, [tmp_path], "bash", [], [], "1:1", interactive=True)
    assert piped[8] == "-i"
