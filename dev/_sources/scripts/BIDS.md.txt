BIDS: run_fastsurfer_bids.py
=============================

`run_fastsurfer_bids.py` is a [BIDS-App](https://bids-apps.neuroimaging.io/about/)-style entrypoint for FastSurfer.
It discovers subjects and sessions in a BIDS dataset and hands them to the existing entrypoints: it writes a subject
list and calls [`brun_fastsurfer.sh`](BATCH.md), or [`srun_fastsurfer.sh`](SLURM.md) with `--slurm`. No part of the
pipeline is reimplemented here, and every option it does not define itself is passed through unchanged.

It needs no additional dependencies. What it reads from a dataset (subject, session, T1w/T2w) is spelled out by the
BIDS directory layout itself, so discovery is a glob over `sub-<label>/[ses-<label>/]anat/`.

Usage
-----
```{command-output} ./run_fastsurfer_bids.py --help
:cwd: /../
```

Basic example
--------------
```
./run_fastsurfer_bids.py /data/my_bids_dataset /data/fastsurfer_output participant \
    --participant_label 01 02 --fs_license /data/license.txt -- --threads 4
```

This processes `sub-01` and `sub-02` from the dataset at `/data/my_bids_dataset`. Any options after a literal `--`
are passed through unchanged to `brun_fastsurfer.sh` or `srun_fastsurfer.sh` (see [RUN_FASTSURFER.md](RUN_FASTSURFER.md),
[BATCH.md](BATCH.md) and [SLURM.md](SLURM.md) for the full set).

In Docker
---------
The image's entrypoint is `run_fastsurfer.sh`, so a BIDS run overrides it. Override it with
`tools/Docker/entrypoint.sh` rather than with the script itself: that is what activates the virtual environment the
pipeline runs in, and it takes the script to run as its first argument.

```bash
docker run --gpus all -v $HOME/my_bids_dataset:/data:ro -v $HOME/my_fastsurfer_analysis:/output \
           -v $HOME/my_fs_license.txt:/fs_license/license.txt \
           --entrypoint "/fastsurfer/tools/Docker/entrypoint.sh" \
           --rm --user $(id -u):$(id -g) deepmi/fastsurfer:latest \
           /fastsurfer/run_fastsurfer_bids.py \
           /data /output participant --fs_license /fs_license/license.txt \
           -- --3T --threads 4
```

On a cluster, `--slurm` submits the same cases through `srun_fastsurfer.sh` instead, with its options given after the
`--` as well:

```
./run_fastsurfer_bids.py /data/my_bids_dataset /data/fastsurfer_output participant --slurm \
    --fs_license /data/license.txt -- --partition gpu --work /scratch/fastsurfer
```

```{warning}
`--slurm` is experimental. Check the output of `--dry` before relying on it. Two things differ from the local route:
`--data` is set to `bids_dir`, because `srun_fastsurfer.sh` rewrites every path in the subject list relative to it
before binding it into the container, and the paths are written unquoted, because that rewrite is done with awk and a
quote stops it from matching. A dataset whose path holds a space is therefore refused with `--slurm`, which is a
limitation of `srun_fastsurfer.sh` rather than of BIDS input: neither of its input routes handles a space.
```

Output naming
-------------
`output_dir` is used directly as FastSurfer's `SUBJECTS_DIR`, and every session becomes one directory in it, named
`sub-<label>_ses-<label>`:

```
fastsurfer_output/
├── dataset_description.json
├── bids_subjects.txt
├── sub-01_ses-1/
├── sub-01_ses-2/
└── sub-02_ses-1/
```

Flat, not nested under `sub-<label>/ses-<label>/`. This is the layout FreeSurfer tooling expects of a `SUBJECTS_DIR`,
so every downstream FreeSurfer or FastSurfer command works on the output directory unchanged. A dataset with no
session level keeps the plain `sub-<label>` as the directory name. Note this means `output_dir` is a FreeSurfer
subjects directory that carries a `dataset_description.json` for provenance, not a valid BIDS-derivatives dataset,
which would require the nested layout.

`dataset_description.json` is written if there is not one there already, and records which of the two processing
models produced the directory. The longitudinal pipeline names its timepoints the same way this names its sessions,
so `sub-01_ses-1` from a cross-sectional run and `sub-01_ses-1` from a longitudinal run are the same directory name
holding results of different methods. **One output directory therefore holds one model**, and a run into a directory
recorded as the other one is refused rather than silently mixed.

The generated subject list is kept as `output_dir/bids_subjects.txt`, so a run can be repeated or amended with
`brun_fastsurfer.sh` directly.

`output_dir` must not be the dataset itself or lie inside one of its subjects, since the output would then be found as
input by the next run. `<bids_dir>/derivatives/fastsurfer` is the usual place inside a dataset.

Sessions
--------
Every session is processed on its own, as one cross-sectional case. Longitudinal processing, where the timepoints of a
subject are conditioned on a person-specific template, is run with [`long_fastsurfer.sh`](long_fastsurfer.rst); it is a
different scientific method rather than a different spelling of this one, and this entrypoint does not choose it for
you.

T1w and T2w input
------------------
Only `*_T1w.nii[.gz]` and `*_T2w.nii[.gz]` are considered.

A T2w image is used only with `--use_t2`, which passes it as `--t2` and switches the
[HypVINN](../overview/OUTPUT_FILES.md#hypvinn-module) hypothalamus module to its multimodal mode. That changes what
the module computes, so it is a choice for the whole study rather than something the presence of a file decides:
using a T2 where one happens to exist and not where it does not would put two methods in one set of results. Where
`--use_t2` is given and some sessions have no T2w, those sessions are processed without one and a warning names them.

A session with several T1w images (for example several `run-` or `acq-` entities) is an error rather than a silent
pick of the first, since which image to process is a statement about the data. Restrict the dataset, or process that
session with `run_fastsurfer.sh` directly. An image that is a link to content that is not there, as in a DataLad
dataset before `datalad get`, is an error too.

`--participant_label` and `--session_label` restrict what is processed and accept the labels with or without their
`sub-`/`ses-` prefix. A label that the dataset does not hold is an error rather than an empty run, and where
`--session_label` is given, a subject with no session level is skipped, since none of its data can be the session that
was asked for.

Validation
----------
The dataset is validated with the [bids-validator](https://github.com/bids-standard/bids-validator) command line tool
if it is installed, and skipped with a warning if it is not. `--skip_bids_validator` says that this is intended.

Dry run
-------
`--dry` prints the subject list and the command that would be executed, and writes nothing, which is the cheap way to
check what was discovered before committing to a full run. The flag is spelled as in `srun_fastsurfer.sh`, which also
accepts `--dry_run`.

`test/integration/openneuro_check.sh` does the same against real data. It fetches public OpenNeuro data for a few BIDS
layouts (several sessions of one subject, a T2w in only some sessions, no session level), runs them to a chosen depth
(dry, segmentation or full), and checks the outputs. With `--slurm` it submits the cases instead, and a later call
with `--check_only` checks the output once the jobs are done. It is run by hand, not by CI.
