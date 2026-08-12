import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from pytest import fixture, mark

FASTSURFER_HOME = Path(__file__).parents[2]
SRUN_FASTSURFER = FASTSURFER_HOME / "srun_fastsurfer.sh"
SEG_SCRIPT = "slurm_cmd_seg.sh"
SURF_SCRIPT = "slurm_cmd_surf.sh"

# the script relies on GNU coreutils behavior (e.g. `head -c -2`, `stat -c`)
pytestmark = mark.skipif(sys.platform != "linux", reason="srun_fastsurfer.sh requires GNU coreutils")


@fixture
def srun_dry(tmp_path_factory) -> Callable[..., str]:
    """Provides a function that performs a dry run of srun_fastsurfer.sh and returns its output."""
    def _run(*extra_args: str) -> str:
        sandbox = tmp_path_factory.mktemp("srun")
        data_dir = sandbox / "data"
        data_dir.mkdir()
        (sandbox / "processed").mkdir()
        (sandbox / "work").mkdir()
        (data_dir / "subject_x.nii.gz").touch()
        fs_license = sandbox / "license.txt"
        fs_license.write_text("fs license\n")
        singularity_image = sandbox / "fastsurfer.sif"
        singularity_image.touch()
        args = [
            "bash", str(SRUN_FASTSURFER),
            "--data", str(data_dir),
            "--sd", str(sandbox / "processed"),
            "--work", str(sandbox / "work"),
            "--pattern", "*.nii.gz",
            "--fs_license", str(fs_license),
            "--singularity_image", str(singularity_image),
            "--dry",
            *extra_args,
        ]
        process = subprocess.run(args, capture_output=True, text=True, cwd=FASTSURFER_HOME)
        assert process.returncode == 0, f"{' '.join(args)} failed:\n{process.stdout}\n{process.stderr}"
        return process.stdout
    return _run


def sbatch_options(output: str, batch_script: str) -> list[str]:
    """Extracts the sbatch options of the job that submits `batch_script` from a dry run output.

    sbatch stops parsing options at the batch script, so only the arguments in front of the batch
    script are options to sbatch, everything after it is passed to the batch script instead.
    """
    for line in output.splitlines():
        tokens = line.split()
        if not tokens or tokens[0] != "sbatch" or not any(t.endswith(batch_script) for t in tokens):
            continue
        script_index = next(i for i, token in enumerate(tokens) if token.endswith(batch_script))
        return tokens[1:script_index]
    raise AssertionError(f"No sbatch command for {batch_script} in the dry run output:\n{output}")


def batch_scripts(output: str) -> list[str]:
    """Extracts the generated slurm batch scripts, which the dry run prints between markers."""
    scripts, script = [], None
    for line in output.splitlines():
        if line.startswith("--- sbatch script"):
            script = []
        elif line.startswith("--- end of script ---") and script is not None:
            scripts.append("\n".join(script))
            script = None
        elif script is not None:
            script.append(line)
    assert scripts, f"No batch scripts in the dry run output:\n{output}"
    return scripts


def test_extra_slurm_options_are_passed_to_both_jobs(srun_dry):
    output = srun_dry("--extra_slurm_options", "--reservation=gpu_nodes")
    assert "--reservation=gpu_nodes" in sbatch_options(output, SEG_SCRIPT)
    assert "--reservation=gpu_nodes" in sbatch_options(output, SURF_SCRIPT)


def test_extra_slurm_options_seg_is_limited_to_the_segmentation_job(srun_dry):
    output = srun_dry("--extra_slurm_options_seg", "--reservation=gpu_nodes")
    assert "--reservation=gpu_nodes" in sbatch_options(output, SEG_SCRIPT)
    assert "--reservation=gpu_nodes" not in sbatch_options(output, SURF_SCRIPT)


def test_extra_slurm_options_surf_is_limited_to_the_surface_job(srun_dry):
    output = srun_dry("--extra_slurm_options_surf", "--qos=long")
    assert "--qos=long" in sbatch_options(output, SURF_SCRIPT)
    assert "--qos=long" not in sbatch_options(output, SEG_SCRIPT)


def test_extra_slurm_options_are_split_into_individual_arguments(srun_dry):
    output = srun_dry("--extra_slurm_options", "--reservation=gpu_nodes --qos=high")
    for batch_script in (SEG_SCRIPT, SURF_SCRIPT):
        options = sbatch_options(output, batch_script)
        assert "--reservation=gpu_nodes" in options
        assert "--qos=high" in options


def test_general_and_pipeline_specific_extra_slurm_options_are_combined(srun_dry):
    output = srun_dry(
        "--extra_slurm_options", "--reservation=gpu_nodes",
        "--extra_slurm_options_seg", "--qos=high",
        "--extra_slurm_options_surf", "--qos=long",
    )
    relevant = ("--reservation", "--qos")
    seg_options = [o for o in sbatch_options(output, SEG_SCRIPT) if o.startswith(relevant)]
    surf_options = [o for o in sbatch_options(output, SURF_SCRIPT) if o.startswith(relevant)]
    assert seg_options == ["--reservation=gpu_nodes", "--qos=high"]
    assert surf_options == ["--reservation=gpu_nodes", "--qos=long"]


def test_extra_slurm_options_are_not_forwarded_to_fastsurfer(srun_dry):
    """Unrecognized options end up in the FastSurfer call, which aborts the run inside the container."""
    output = srun_dry("--extra_slurm_options", "--reservation=gpu_nodes")
    for script in batch_scripts(output):
        assert "--reservation=gpu_nodes" not in script


def test_sbatch_options_are_unaffected_without_extra_slurm_options(srun_dry):
    output = srun_dry()
    assert "" not in sbatch_options(output, SEG_SCRIPT)
    assert "" not in sbatch_options(output, SURF_SCRIPT)
    assert "--reservation" not in output
