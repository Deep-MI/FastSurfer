#!/usr/bin/env python3
"""Merge left/right mris_volmask one-hemi outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lh", required=True, type=Path, help="Left-hemi full ribbon/WM mask.")
    parser.add_argument("--rh", required=True, type=Path, help="Right-hemi full ribbon/WM mask.")
    parser.add_argument("--out", required=True, type=Path, help="Merged ribbon/WM mask output.")
    parser.add_argument("--lh-ribbon", required=True, type=Path, help="Binary left ribbon output.")
    parser.add_argument("--rh-ribbon", required=True, type=Path, help="Binary right ribbon output.")
    parser.add_argument("--left-ribbon-label", default=3, type=int)
    parser.add_argument("--right-ribbon-label", default=42, type=int)
    return parser.parse_args()


def save_like(reference: nib.spatialimages.SpatialImage, data: np.ndarray, path: Path) -> None:
    image = nib.MGHImage(data.astype(np.uint8, copy=False), reference.affine, reference.header.copy())
    image.set_data_dtype(np.uint8)
    nib.save(image, str(path))


def main() -> int:
    args = parse_args()
    lh_img = nib.load(args.lh)
    rh_img = nib.load(args.rh)
    lh_data = np.asanyarray(lh_img.dataobj, dtype=np.uint8)
    rh_data = np.asanyarray(rh_img.dataobj, dtype=np.uint8)

    merged = np.where(lh_data != 0, lh_data, rh_data)
    save_like(lh_img, merged, args.out)
    save_like(lh_img, (lh_data == args.left_ribbon_label).astype(np.uint8), args.lh_ribbon)
    save_like(rh_img, (rh_data == args.right_ribbon_label).astype(np.uint8), args.rh_ribbon)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
