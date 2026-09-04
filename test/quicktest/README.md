quicktest tests
===============

This suite compares two runs of fastsurfer and is used as a online check for function in the [github quicktest workflow](/.github/workflows/quicktest.yaml).

The `quicktest` suite requires
- A python environment as defined by `fastsurfer[quicktest]`
  ```bash
  python -m pip install $FASTSURFER_HOME[quicktest]
  ```
- A target `subject directory` for an image (processed with a known good version of FastSurfer). This should be placed in the directory defined by the environment variable `REF_DIR`.
- A to-compare `subject directory` for an image. This should be placed in the directory defined by the environment variable `SUBJECTS_DIR`.
- A definition of the test setup in the following environment variables:
  - `REF_DIR`: known-good reference data. One subject per folder, each compared against the folder of the same name in `SUBJECTS_DIR`; `logs` and `slurm` are ignored.
  - `SUBJECTS_DIR`: to-compare/test data
  - `MAX_SUBJECTS`: optional, process at most this many subjects (default: all)

Test 1: Search for errors in to-compare log files
-------------------------------------------------

Contained in test_errors_in_logfiles.py


Test 2: Check existence of expected files in to-compare subject directory
-------------------------------------------------------------------------

Contained in test_file_existence.py

Test 3: Compare output images and segmentation maps
---------------------------------------------------

Contained in test_images.py

Test 4: Check output stats files
--------------------------------

Contained in test_stats.py

Refreshing the reference data after a release
---------------------------------------------

The reference data has to be regenerated after every release, so that it stays the output of the most
recent released version. Outputs drift as the models and the pipeline change, and an old reference
turns that drift into test failures, which then hide any real regression.

The inputs and the references are downloaded from urls kept in repository secrets
(`QUICKTEST_IMAGE_HREF_*` and `QUICKTEST_TARGET_HREF_*`), because the data is too large for the
repository. To refresh one, process the input with the released version and the flags from the
matrix in [quicktest.yaml](/.github/workflows/quicktest.yaml), archive the subject folder, and update
the url. The archive is unpacked into `REF_DIR`, so the subject folder has to sit at its top level
under the name the workflow uses for the case.

Tolerances are separate and do live here, one file per output under [data](data). Widen one only for
a small difference that keeps recurring; a large disagreement means the reference is out of date or
something really changed.
