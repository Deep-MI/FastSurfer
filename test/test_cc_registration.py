from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest
from neuroreg.segreg import segreg
from neuroreg.segreg.atlas import load_atlas_data

from CorpusCallosum.registration.midsagittal_plane_alignment import (
    find_midplane_transform,
    register_centroids_to_fsavg,
)


_LABEL_COORDS = {
    2: (10, 20, 30),
    41: (30, 20, 30),
    4: (12, 40, 20),
    43: (32, 40, 20),
    10: (15, 25, 50),
    49: (35, 25, 50),
}


def _make_affine(zooms: tuple[float, float, float] = (0.8, 1.0, 1.2)) -> np.ndarray:
    affine = np.eye(4, dtype=np.float64)
    affine[0, 0], affine[1, 1], affine[2, 2] = zooms
    return affine


def _make_aseg(
    *,
    shape: tuple[int, int, int] = (64, 64, 64),
    zooms: tuple[float, float, float] = (0.8, 1.0, 1.2),
) -> nib.Nifti1Image:
    data = np.zeros(shape, dtype=np.int16)
    for label, coord in _LABEL_COORDS.items():
        data[coord] = label
    return nib.Nifti1Image(data, affine=_make_affine(zooms))


def _make_orig(
    *,
    shape: tuple[int, int, int] = (64, 64, 64),
    zooms: tuple[float, float, float] = (0.8, 1.0, 1.2),
) -> nib.Nifti1Image:
    data = np.zeros(shape, dtype=np.float32)
    return nib.Nifti1Image(data, affine=_make_affine(zooms))


def test_register_centroids_to_fsavg_matches_neuroreg_and_adjusts_resolution():
    aseg_img = _make_aseg()
    neuroreg_result = segreg(
        mov=aseg_img,
        atlas="fsaverage",
        dof=6,
        label_set="fsaverage_centroids",
    )
    fsaverage_vox2ras, fsaverage_header = load_atlas_data("fsaverage")

    aseg2fsavg_vox2vox, aseg2fsavg_ras2ras, fsavg_hires_vox2ras, fsavg_header = register_centroids_to_fsavg(aseg_img)

    expected_delta = np.array([0.8, 1.2, 1.0], dtype=np.float64)
    expected_dims = np.ceil(np.asarray(fsaverage_header["dims"], dtype=np.float64) / expected_delta).astype(int)
    resolution_trans = np.diagflat(np.append(expected_delta, [1.0])).astype(np.float64)
    expected_hires_vox2ras = np.concatenate(
        [(resolution_trans @ fsaverage_vox2ras)[:, :3], fsaverage_vox2ras[:, 3:4]],
        axis=1,
    )
    expected_vox2vox = np.linalg.inv(expected_hires_vox2ras) @ neuroreg_result.r2r @ aseg_img.affine

    assert aseg2fsavg_vox2vox.shape == (4, 4)
    assert aseg2fsavg_ras2ras.shape == (4, 4)
    assert fsavg_hires_vox2ras.shape == (4, 4)
    assert aseg2fsavg_ras2ras == pytest.approx(neuroreg_result.r2r, abs=1e-6)
    assert fsavg_hires_vox2ras == pytest.approx(expected_hires_vox2ras, abs=1e-6)
    assert aseg2fsavg_vox2vox == pytest.approx(expected_vox2vox, abs=1e-5)
    assert sorted(fsavg_header) == ["Mdc", "Pxyz_c", "delta", "dims"]
    assert np.asarray(fsavg_header["delta"], dtype=np.float64) == pytest.approx(expected_delta, abs=1e-6)
    assert fsavg_header["dims"] == expected_dims.tolist()


def test_find_midplane_transform_fsaverage_preserves_fsaverage_geometry():
    orig = _make_orig()
    aseg_img = _make_aseg()
    _, fsaverage_header = load_atlas_data("fsaverage")
    expected_delta = np.array([0.8, 1.2, 1.0], dtype=np.float64)
    expected_dims = np.ceil(np.asarray(fsaverage_header["dims"], dtype=np.float64) / expected_delta).astype(int)

    result = find_midplane_transform(orig=orig, aseg_img=aseg_img, midplane_method="fsaverage")

    assert result.orig2fsavg_vox2vox.shape == (4, 4)
    assert result.fsavg_vox2ras.shape == (4, 4)
    assert result.fsavg_shape == tuple(expected_dims.tolist())
    assert result.fsavg_header_dict["dims"] == expected_dims.tolist()
    assert result.base_middle_vox == pytest.approx(128.0 / 0.8)
    assert result.midline_shift_vox == pytest.approx(0.0)
    assert result.midline_shift_diagnostics == {}


def test_find_midplane_transform_fsaverage_symmetry_uses_neuroreg_base_registration():
    orig = _make_orig()
    aseg_img = _make_aseg()

    result = find_midplane_transform(orig=orig, aseg_img=aseg_img, midplane_method="fsaverage_symmetry")

    assert result.orig2fsavg_vox2vox.shape == (4, 4)
    assert result.fsavg_vox2ras.shape == (4, 4)
    assert result.base_middle_vox == pytest.approx(128.0 / 0.8)
    assert np.isfinite(result.midline_shift_vox)
    assert "candidate_scores" in result.midline_shift_diagnostics
    assert "selected_shift_support" in result.midline_shift_diagnostics
