BIDS: run_fastsurfer_bids.py
=============================

`run_fastsurfer_bids.py` is a [BIDS-App](https://bids-apps.neuroimaging.io/about/)-style entrypoint for FastSurfer.
It discovers subjects and sessions directly from a BIDS-valid dataset and delegates processing to the existing
FastSurfer entrypoints ([`brun_fastsurfer.sh`](BATCH.md) for cross-sectional subjects,
[`long_fastsurfer.sh`](long_fastsurfer.rst) for subjects with multiple sessions).

Installation
------------
The BIDS entrypoint requires [pybids](https://bids-standard.github.io/pybids/), which is not installed by default.
Install it with:

```
pip install fastsurfer[bids]
```

Usage
-----
```{command-output} ./run_fastsurfer_bids.py --help
:cwd: /../
```

Basic example
--------------
```
./run_fastsurfer_bids.py /data/my_bids_dataset /data/fastsurfer_output participant \
    --participant_label 01 02 --fs_license /data/license.txt -- --parallel
```

This processes `sub-01` and `sub-02` from the BIDS dataset at `/data/my_bids_dataset`, writing FreeSurfer-style
subject directories into `/data/fastsurfer_output`. Any options after a literal `--` are passed through unchanged to
the underlying `run_fastsurfer.sh`/`brun_fastsurfer.sh`/`long_fastsurfer.sh` calls (see [RUN_FASTSURFER.md](RUN_FASTSURFER.md)
and [BATCH.md](BATCH.md) for the full set of supported options).

Session and longitudinal handling
----------------------------------
- Subjects with a single session (or no `ses-` level in the dataset) are processed cross-sectionally, one call to
  `brun_fastsurfer.sh` per selected subject batch.
- Subjects with **two or more** sessions are, by default, processed as a longitudinal cohort via `long_fastsurfer.sh`
  (person-specific template `sub-X`, timepoints `sub-X_ses-Y`).
- `--cross_sectional` forces every session of every subject to be processed independently, even if multiple sessions
  are present.
- `--longitudinal` forces the longitudinal pipeline; subjects with only one session fall back to cross-sectional
  processing with a warning.
- Passing `--seg_only` or `--surf_only` after the literal `--` also forces cross-sectional processing for multi-session
  subjects, because these modes are not supported by `long_fastsurfer.sh`.

T1w and T2w input
------------------
Only `*_T1w.nii[.gz]` and `*_T2w.nii[.gz]` anatomical images are considered. When a `T2w` image is present for a
session, it is passed as `--t2` to enable the [HypVINN](../overview/OUTPUT_FILES.md#hypvinn-module) hypothalamus
module. This is only supported in the cross-sectional pipeline; `long_fastsurfer.sh` does not accept `--t2`, so T2w
images for longitudinal subjects are ignored with a warning.

Output
------
The `output_dir` is used directly as FastSurfer's `SUBJECTS_DIR`. A minimal BIDS-derivatives
`dataset_description.json` is written into `output_dir`.

Dry run
-------
Pass `--dry_run` to print the commands that would be executed (including the generated subject-list file contents)
without running anything, e.g. to sanity-check subject/session discovery before committing to a full run.
