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

On a cluster, `--slurm` submits the same cases through `srun_fastsurfer.sh` instead, with its options given after the
`--` as well:

```
./run_fastsurfer_bids.py /data/my_bids_dataset /data/fastsurfer_output participant --slurm \
    --fs_license /data/license.txt -- --partition gpu --work /scratch/fastsurfer
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
so every downstream FreeSurfer or FastSurfer command works on the output directory unchanged, and it is the same
naming the longitudinal pipeline uses for its timepoints. A dataset with no session level keeps the plain `sub-<label>`
as the directory name.

A minimal BIDS-derivatives `dataset_description.json` is written into `output_dir` if there is not one there already,
and the generated subject list is kept as `output_dir/bids_subjects.txt`, so a run can be repeated or amended with
`brun_fastsurfer.sh` directly.

Sessions
--------
Every session is processed on its own, as one cross-sectional case. Longitudinal processing, where the timepoints of a
subject are conditioned on a person-specific template, is run with [`long_fastsurfer.sh`](long_fastsurfer.rst); it is a
different scientific method rather than a different spelling of this one, and this entrypoint does not choose it for
you.

T1w and T2w input
------------------
Only `*_T1w.nii[.gz]` and `*_T2w.nii[.gz]` are considered. Where a session has a T2w image, it is passed as `--t2`,
which enables the [HypVINN](../overview/OUTPUT_FILES.md#hypvinn-module) hypothalamus module. A session with several
T1w images (for example several `run-` or `acq-` entities) uses the first in alphabetical order and says so; process
the other one with `run_fastsurfer.sh` directly if that is the wrong choice.

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

`test/integration/openneuro_check.sh` does the same against real data: it fetches two sessions of a randomly drawn
subject from a public OpenNeuro dataset, runs them, and checks the outputs. It is run by hand, not by CI.
