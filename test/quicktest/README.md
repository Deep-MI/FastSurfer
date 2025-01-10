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
  - `REF_DIR`: known-good reference data
  - `SUBJECTS_DIR`: to-compare/test data
  - `SUBJECTS_LIST`: comma separated list of 

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
