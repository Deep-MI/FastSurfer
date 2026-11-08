"""Checks that the partial volume corrected volumes of segstats.py do not depend on the intensity image (norm)."""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from FastSurferCNN import mri_segstats, segstats
from FastSurferCNN.segstats import pv_calc

VOX_VOL = 0.8 ** 3


def make_ventricle_in_white_matter() -> tuple[np.ndarray, np.ndarray]:
    """Lateral ventricle (4) inside white matter (2), noisy partial volume guide, background in the outer two slices."""
    seg = np.full((24, 24, 24), 2, dtype=np.int32)
    seg[6:18, 9:15, 6:18] = 4
    seg[:2], seg[-2:] = 0, 0
    rng = np.random.default_rng(0)
    pv_guide = np.select([seg == 4, seg == 2], [30.0, 110.0], 0.0) + rng.normal(0, 1, seg.shape)
    return seg, pv_guide.astype(np.float32)


def volumes(table) -> dict[int, float]:
    """Volume per SegId of a pv_calc table."""
    return {row["SegId"]: row["Volume_mm3"] for row in table}


@pytest.mark.parametrize("legacy_freesurfer", [False, True])
@pytest.mark.parametrize(("labels", "merged_labels"), [
    ([2, 4], None),
    ([2, 4], {10000: [2, 4]}),
    # segstats.py --measure_only passes no labels, only the merged labels of the measures
    (np.zeros((0,), dtype=int), {10000: [4]}),
])
def test_volume_without_norm_matches_volume_with_norm(labels, merged_labels, legacy_freesurfer):
    seg, pv_guide = make_ventricle_in_white_matter()
    kwargs = {"vox_vol": VOX_VOL, "merged_labels": merged_labels, "legacy_freesurfer": legacy_freesurfer}
    without_norm = volumes(pv_calc(seg, pv_guide, None, labels, **kwargs))
    with_norm = volumes(pv_calc(seg, pv_guide, pv_guide, labels, **kwargs))
    assert without_norm == pytest.approx(with_norm, rel=1e-12)


def test_volume_without_norm_of_uniform_cube():
    seg = np.zeros((12, 12, 12), dtype=np.int32)
    seg[3:8, 3:8, 3:8] = 1
    pv_guide = (seg * 100).astype(np.float32)
    (row,) = pv_calc(seg, pv_guide, None, [1], vox_vol=VOX_VOL)
    assert row["NVoxels"] == 125
    assert row["Volume_mm3"] == pytest.approx(125 * VOX_VOL)


def write_images(directory: Path) -> tuple[Path, Path]:
    """Write the segmentation and the partial volume guide of `make_ventricle_in_white_matter` as mgz files."""
    seg, pv_guide = make_ventricle_in_white_matter()
    affine = np.diag([0.8, 0.8, 0.8, 1.0])
    seg_file, pv_file = directory / "aseg.mgz", directory / "norm.mgz"
    nib.save(nib.MGHImage(seg, affine), seg_file)
    nib.save(nib.MGHImage(pv_guide, affine), pv_file)
    return seg_file, pv_file


def read_measure(stats_file: Path, name: str) -> float:
    """Value of the Measure `name` in the header of a stats file."""
    for line in stats_file.read_text().splitlines():
        if line.startswith(f"# Measure {name},"):
            return float(line.split(",")[3])
    raise AssertionError(f"Measure {name} not found in {stats_file}")


def read_volumes(stats_file: Path) -> dict[int, float]:
    """Volume per SegId of the table in a stats file."""
    lines = stats_file.read_text().splitlines()
    columns = next(line for line in lines if line.startswith("# ColHeaders")).split()[2:]
    rows = [dict(zip(columns, line.split(), strict=True)) for line in lines if line and not line.startswith("#")]
    return {int(row["SegId"]): float(row["Volume_mm3"]) for row in rows}


def test_segstats_measure_only_without_normfile(tmp_path):
    # recon-surf.sh computes brainvol.stats like this, without --normfile
    seg_file, pv_file = write_images(tmp_path)
    measure = {}
    for normfile in ([], ["--normfile", str(pv_file)]):
        stats_file = tmp_path / f"brainvol{len(normfile)}.stats"
        args = segstats.make_arguments().parse_args([
            "--segfile", str(seg_file), "--pvfile", str(pv_file), *normfile, "--measure_only",
            "--segstatsfile", str(stats_file), "--sd", str(tmp_path), "--sid", "subject",
            "measures", "--compute", "VentricleChoroidVol",
        ])
        assert segstats.main(args) == 0
        measure[bool(normfile)] = read_measure(stats_file, "VentricleChoroidVol")
    assert measure[False] == pytest.approx(measure[True], rel=1e-6)
    # the ventricle has 12 * 6 * 12 voxels, not only its border voxels
    assert measure[False] == pytest.approx(12 * 6 * 12 * VOX_VOL, rel=0.05)


def test_mri_segstats_pv_without_i(tmp_path):
    seg_file, pv_file = write_images(tmp_path)
    volume = {}
    for intensity in ([], ["--i", str(pv_file)]):
        stats_file = tmp_path / f"aseg{len(intensity)}.stats"
        args = mri_segstats.parse_arguments([
            "--seg", str(seg_file), "--pv", str(pv_file), *intensity, "--sum", str(stats_file), "--id", "2", "4",
        ])
        assert mri_segstats.main(args) == 0
        volume[bool(intensity)] = read_volumes(stats_file)
    assert volume[False] == pytest.approx(volume[True], rel=1e-6)
