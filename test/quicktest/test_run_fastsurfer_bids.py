from run_fastsurfer_bids import _requires_cross_sectional_passthrough


def test_requires_cross_sectional_passthrough_detects_seg_only():
    assert _requires_cross_sectional_passthrough(["--seg_only"])


def test_requires_cross_sectional_passthrough_detects_surf_only():
    assert _requires_cross_sectional_passthrough(["--surf_only", "--parallel", "2"])


def test_requires_cross_sectional_passthrough_allows_full_pipeline_flags():
    assert not _requires_cross_sectional_passthrough(["--parallel", "2", "--3T"])
