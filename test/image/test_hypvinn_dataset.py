"""The HypVINN dataset scales each view by the voxel sizes of its in-plane axes."""

import numpy as np
import pytest

from FastSurferCNN.utils.checkpoint import get_config_file, load_checkpoint_config_defaults
from FastSurferCNN.utils.parser_defaults import FASTSURFER_ROOT
from HypVINN.data_loader.dataset import HypVINNDataset
from HypVINN.utils.load_config import load_config

# the config files the pipeline uses by default
CONFIGS = load_checkpoint_config_defaults("config", get_config_file("HypVINN"))
SHAPE = (10, 20, 30)


@pytest.mark.parametrize("plane", ["axial", "coronal", "sagittal"])
@pytest.mark.parametrize("zooms", [(1.0, 2.0, 3.0), (0.8, 0.8, 1.2), (1.2, 0.9, 0.9), (1.0, 1.0, 1.0)])
def test_scale_factor_uses_in_plane_voxel_sizes(plane, zooms):
    cfg = load_config(str(FASTSURFER_ROOT / CONFIGS[plane]))
    dataset = HypVINNDataset("test", {"t1": np.zeros(SHAPE, dtype=np.uint8)}, zooms, cfg, mode="t1")
    # the dataset stores slices as (N, H, W, C), and all axes of SHAPE have different lengths,
    # so the in-plane axes of the image (and their voxel sizes) can be identified by their length
    height, width = dataset.images.shape[1:3]
    in_plane_zooms = np.asarray([zooms[SHAPE.index(height)], zooms[SHAPE.index(width)]])
    expected = cfg.MODEL.BASE_RES / in_plane_zooms
    assert np.allclose(dataset[0]["scale_factor"], expected)
