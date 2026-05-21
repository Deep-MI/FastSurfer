#!/usr/bin/env python3
"""Run mris_volmask on a cropped subject volume and embed the result."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.freesurfer.io import read_geometry

from cropped_volume import crop_slices, freesurfer_env, load_volume, save_volume


def _surface_voxel_bounds(subject_dir: Path, hemi: str, img: nib.spatialimages.SpatialImage) -> tuple[np.ndarray, np.ndarray]:
    inv = np.linalg.inv(img.affine)
    coords = []
    for surface in ("white", "pial"):
        ras, _ = read_geometry(str(subject_dir / "surf" / f"{hemi}.{surface}"))
        voxel = (inv @ np.c_[ras, np.ones(len(ras))].T).T[:, :3]
        coords.append(voxel)
    points = np.vstack(coords)
    return np.floor(points.min(axis=0)).astype(int), (np.ceil(points.max(axis=0)) + 1).astype(int)


def _bounds(shape: tuple[int, ...], surface_start: np.ndarray, surface_stop: np.ndarray, margin: int) -> tuple[np.ndarray, np.ndarray]:
    start = surface_start
    stop = surface_stop
    start = np.maximum(0, start - margin)
    stop = np.minimum(shape, stop + margin)
    return start.astype(int), stop.astype(int)


def _copy_surface_files(source: Path, target: Path, hemi: str) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for surface in ("white", "pial"):
        shutil.copy2(source / "surf" / f"{hemi}.{surface}", target / f"{hemi}.{surface}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sd", required=True, type=Path)
    parser.add_argument("--sid", required=True)
    parser.add_argument("--hemi", required=True, choices=("lh", "rh"))
    parser.add_argument("--aseg-name", default="aseg")
    parser.add_argument("--out-root", default="ribbon")
    parser.add_argument("--cap-distance", default="2")
    parser.add_argument("--margin", default=32, type=int)
    parser.add_argument("--label-left-white", default="20")
    parser.add_argument("--label-left-ribbon", default="10")
    parser.add_argument("--label-right-white", default="120")
    parser.add_argument("--label-right-ribbon", default="110")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    subject_dir = args.sd / args.sid
    mri_dir = subject_dir / "mri"
    source_volume = mri_dir / f"{args.aseg_name}.mgz"
    img, data = load_volume(source_volume)
    surface_start, surface_stop = _surface_voxel_bounds(subject_dir, args.hemi, img)
    start, stop = _bounds(data.shape, surface_start, surface_stop, args.margin)
    crop = crop_slices(start, stop)

    tmp_root = subject_dir / "tmp"
    tmp_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"cropped-volmask-{args.hemi}-", dir=tmp_root) as tmpdir:
        tmp_subject_dir = Path(tmpdir) / args.sid
        tmp_mri_dir = tmp_subject_dir / "mri"
        tmp_surf_dir = tmp_subject_dir / "surf"
        tmp_mri_dir.mkdir(parents=True, exist_ok=True)
        _copy_surface_files(subject_dir, tmp_surf_dir, args.hemi)
        save_volume(data[crop], img, tmp_mri_dir / f"{args.aseg_name}.mgz", start)

        hemi_flag = "--lh-only" if args.hemi == "lh" else "--rh-only"
        cmd = [
            "mris_volmask",
            "--sd",
            str(Path(tmpdir)),
            "--aseg_name",
            args.aseg_name,
            "--label_left_white",
            args.label_left_white,
            "--label_left_ribbon",
            args.label_left_ribbon,
            "--label_right_white",
            args.label_right_white,
            "--label_right_ribbon",
            args.label_right_ribbon,
            "--cap_distance",
            args.cap_distance,
            "--out_root",
            args.out_root,
            hemi_flag,
            args.sid,
        ]
        subprocess.run(cmd, check=True, env=freesurfer_env())
        cropped_img, cropped_mask = load_volume(tmp_mri_dir / f"{args.out_root}.mgz")

    out = np.zeros_like(data, dtype=np.asarray(cropped_mask).dtype)
    out[crop] = cropped_mask.astype(out.dtype, copy=False)
    save_volume(out, img, mri_dir / f"{args.out_root}.mgz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
