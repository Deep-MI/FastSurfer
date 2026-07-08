Integration tests
==================

Unlike `test/quicktest` (which compares two already-processed subject directories) and
`test/image` (which needs pre-existing reference images), the tests here fetch real data over the
network and drive FastSurfer's entrypoints end-to-end.

BIDS: `test_bids_openneuro.sh`
-------------------------------
Smoke test for `run_fastsurfer_bids.py`. Downloads a small subset of the public OpenNeuro dataset
[ds004937](https://doi.org/10.18112/openneuro.ds004937.v1.0.1) (no login/token required, ~60MB) via
`fetch_openneuro_bids_subset.sh`, then runs `run_fastsurfer_bids.py` once over two subjects to
exercise both routing paths in a single call:

- one subject with only `ses-1` fetched -> cross-sectional path (`brun_fastsurfer.sh`)
- one subject with `ses-1..ses-4` fetched -> longitudinal path (`long_fastsurfer.sh`)

Two subjects with different session counts are required because a single subject cannot exercise
both code paths at once, and this also verifies that a mixed batch is routed correctly (not just
each path in isolation).

```bash
# print the commands that would run, without executing FastSurfer (default)
test/integration/test_bids_openneuro.sh /tmp/bids_smoketest

# actually run FastSurfer (requires fastsurfer[all], a FreeSurfer license, and
# enough time/compute -- the longitudinal subject processes 4 timepoints)
test/integration/test_bids_openneuro.sh /tmp/bids_smoketest --run --fs_license /path/to/license.txt

# actually run, segmentation only (much faster, skips surface reconstruction)
test/integration/test_bids_openneuro.sh /tmp/bids_smoketest --run --fs_license /path/to/license.txt -- --seg_only
```

Requires `fastsurfer[bids]` (pybids) to be installed. Override the default test subjects via the
`CROSS_SUB`/`LONG_SUB`/`LONG_SESSIONS` environment variables if needed.
