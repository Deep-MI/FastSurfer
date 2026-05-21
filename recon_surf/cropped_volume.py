#!/usr/bin/env python3
"""Small helpers for cropped FreeSurfer volume wrappers."""

from __future__ import annotations

import os
from pathlib import Path

import nibabel as nib
import numpy as np


def load_volume(path: Path) -> tuple[nib.spatialimages.SpatialImage, np.ndarray]:
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    if data.ndim == 4:
        data = data[..., 0]
    return img, np.asarray(data)


def crop_affine(affine: np.ndarray, start: np.ndarray) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, 3] = start
    return affine @ transform


def save_volume(
    data: np.ndarray,
    source: nib.spatialimages.SpatialImage,
    path: Path,
    start: np.ndarray | None = None,
) -> None:
    affine = source.affine if start is None else crop_affine(source.affine, start)
    image = nib.MGHImage(data, affine, source.header.copy())
    image.set_data_dtype(data.dtype)
    nib.save(image, str(path))


def bounds_from_mask(mask: np.ndarray, margin: int) -> tuple[np.ndarray, np.ndarray]:
    coords = np.array(np.nonzero(mask))
    if coords.size == 0:
        start = np.zeros(mask.ndim, dtype=int)
        stop = np.array(mask.shape, dtype=int)
    else:
        start = np.maximum(0, coords.min(axis=1) - margin)
        stop = np.minimum(mask.shape, coords.max(axis=1) + 1 + margin)
    return start.astype(int), stop.astype(int)


def crop_slices(start: np.ndarray, stop: np.ndarray) -> tuple[slice, ...]:
    return tuple(slice(int(s), int(e)) for s, e in zip(start, stop))


def freesurfer_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("USER", "fastsurfer")
    env.setdefault("LOGNAME", env["USER"])
    return env
