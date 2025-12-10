# Copyright 2025 AI in Medical Imaging, German Center for Neurodegenerative Diseases(DZNE), Bonn
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from collections.abc import Iterator
from pathlib import Path
from typing import cast, overload

import nibabel as nib
import numpy as np
import torch
from monai import transforms
from numpy import typing as npt

from CorpusCallosum.data import constants
from CorpusCallosum.transforms.segmentation import CropAroundACPC
from CorpusCallosum.utils.checkpoint import YAML_DEFAULT as CC_YAML
from FastSurferCNN.download_checkpoints import load_checkpoint_config_defaults
from FastSurferCNN.download_checkpoints import main as download_checkpoints
from FastSurferCNN.models.networks import FastSurferVINN
from FastSurferCNN.utils import Image3d, Image4d, Shape2d, Shape3d, Shape4d, Vector2d, nibabelImage
from FastSurferCNN.utils.parallel import thread_executor


def load_model(device: torch.device | None = None) -> FastSurferVINN:
    """Load trained model from checkpoint.

    Parameters
    ----------
    device : torch.device or None, optional
        Device to load model to, by default None.
        If None, uses CUDA if available, else CPU.

    Returns
    -------
    FastSurferVINN
        Loaded and initialized model in evaluation mode.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    params = {
        "num_classes": 3,
        "num_filters": 71,
        "num_filters_interpol": 32,
        "num_channels": 9,
        "kernel_h": 3,
        "kernel_w": 3,
        "kernel_c": 1,
        "stride_conv": 1,
        "stride_pool": 2,
        "pool": 2,
        "height": 128,
        "width": 128,
        "base_res": 1.0,
        "interpolation_mode": "bilinear",
        "crop_position": "top_left",
        "out_tensor_width": 320,
        "out_tensor_height": 320,
    }
    model = FastSurferVINN(params)
    
    download_checkpoints(cc=True)
    cc_config: dict[str, Path] = load_checkpoint_config_defaults(
        "checkpoint",
        filename=CC_YAML,
    )
    checkpoint_path = constants.FASTSURFER_ROOT / cc_config['segmentation']
    
    weights = torch.load(checkpoint_path, weights_only=True, map_location=device)
    model.load_state_dict(weights)
    model.eval()
    model.to(device)
    return model


def run_inference(
    model: "torch.nn.Module",
    image_slice: Image3d,
    ac_center: Vector2d,
    pc_center: Vector2d,
    voxel_size: tuple[float, float],
    device: torch.device | None = None,
    transform: transforms.Transform | None = None
) -> tuple[np.ndarray[Shape4d, np.dtype[int]], Image4d, Image4d]:
    """Run inference on a single image slice.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model.
    image_slice : np.ndarray
        LIA-oriented input image as numpy array of shape (L, I, A).
    ac_center : np.ndarray
        Anterior commissure coordinates.
    pc_center : np.ndarray
        Posterior commissure coordinates.
    voxel_size : a pair of floats
        Voxel size of inferior/superior and anterior/posterior direction in mm.
    device : torch.device, optional
        Device to run inference on. If None, uses the device of the model.
    transform : transforms.Transform, optional
        Custom transform pipeline.

    Returns
    -------
    seg_labels : npt.NDArray[int]
        The segmentation result.
    inputs : npt.NDArray[float]
        The inputs to the model.
    soft_labels : npt.NDArray[float]
        The softlabel output.
    """
    if device is None:
        device = next(model.parameters()).device
        
    crop_around_acpc = CropAroundACPC(keys=['image'], padding_mm=35, random_translate=0)
    to_discrete = transforms.AsDiscrete(argmax=True, to_onehot=3)

    # Preprocess slice
    _inputs = torch.from_numpy(image_slice[:,None]) #,:256,:256]) # artifact from training script
    sample = {'image': _inputs, 'AC_center': ac_center, 'PC_center': pc_center, 'res': np.asarray(voxel_size)}
    sample_cropped = crop_around_acpc(sample)
    _inputs, to_pad = sample_cropped['image'], sample_cropped['to_pad']
    _inputs = transforms.utils.rescale_array(_inputs, 0, 1, dtype=np.float32).to(device)

    # split into slices with 9 channels each
    # Generate views with sliding window of 9 slices
    batch_size, channels, height, width = _inputs.shape
    _inputs = _inputs.unfold(0, 9, 1).swapdims(-1, 1).reshape(-1, 9*channels, height, width)

    # Post-process outputs
    with torch.no_grad():
        scale_factors = torch.ones((_inputs.shape[0], 2), device=device) / torch.asarray([voxel_size], device=device)
        
        _logits = model(_inputs, scale_factor=scale_factors)
        _softlabels = transforms.Activations(softmax=True, dim=1)(_logits)
        
        softlabels = _softlabels.cpu().numpy()
        _labels = torch.stack([to_discrete(i) for i in _softlabels])

        # Pad back to original size, to_pad is a tuple[int, int, int, int]
        pad_tuples = ((0, 0),) * 2 + (to_pad[:2], to_pad[2:])
        labels = np.pad(_labels.cpu().numpy(), pad_tuples, mode='constant', constant_values=0)
        softlabels = np.pad(softlabels, pad_tuples, mode='constant', constant_values=0)

    return tuple(x.transpose(0, 2, 3, 1) for x in (labels, _inputs.cpu().numpy(), softlabels))


def load_validation_data(
    path: str | Path,
) -> tuple[npt.NDArray[str], npt.NDArray[float], npt.NDArray[float], Iterator[int], npt.NDArray[str], list[str]]:
    """Load validation data from CSV file and compute label widths.

    Reads a CSV file containing image paths, label paths, and AC/PC coordinates,
    then computes the width (number of slices with non-zero labels) for each label file.

    Parameters
    ----------
    path : str or Path
        Path to the CSV file containing validation data. The CSV should have columns:
        image, label, AC_center_x, AC_center_y, AC_center_z,
        PC_center_x, PC_center_y, PC_center_z.

    Returns
    -------
    images : npt.NDArray[str]
        Array of image file paths.
    ac_centers : npt.NDArray[float]
        Array of anterior commissure coordinates (x, y, z).
    pc_centers : npt.NDArray[float]
        Array of posterior commissure coordinates (x, y, z).
    label_widths : Iterator[int]
        Iterator yielding the number of slices with non-zero labels for each label file.
    labels : npt.NDArray[str]
        Array of label file paths.
    subj_ids : list[str]
        List of subject IDs (from CSV index).
    """
    import pandas as pd

    data = pd.read_csv(path, index_col=0, header=None)
    data.columns = ["image", "label", "AC_center_x", "AC_center_y", "AC_center_z",
                    "PC_center_x", "PC_center_y", "PC_center_z"]

    ac_centers = data[["AC_center_x", "AC_center_y", "AC_center_z"]].values
    pc_centers = data[["PC_center_x", "PC_center_y", "PC_center_z"]].values
    images = data["image"].values
    labels = data["label"].values
    subj_ids = data.index.values.tolist()

    def _load(label_path: str | Path) -> int:
        """Compute the width of non-zero slices in a label image.

        Parameters
        ----------
        label_path : str or Path
            Path to the label image file

        Returns
        -------
        int
            Number of slices containing non-zero labels, or total slices if <= 100
        """
        label_img = cast(nibabelImage, nib.load(label_path))

        if label_img.shape[0] > 100:
            # check which slices have non-zero values
            label_data = np.asarray(label_img.dataobj)
            non_zero_slices = np.any(label_data > 0, axis=(1,2))
            first_nonzero = np.argmax(non_zero_slices)
            last_nonzero = len(non_zero_slices) - np.argmax(non_zero_slices[::-1])
            return last_nonzero - first_nonzero
        else:
            return label_img.shape[0]

    label_widths = thread_executor().map(_load, data["label"])

    return images, ac_centers, pc_centers, label_widths, labels, subj_ids

@overload
def one_hot_to_label(one_hot: Image4d, label_ids: list[int] | None = None) -> np.ndarray[Shape3d, np.dtype[int]]: ...

@overload
def one_hot_to_label(one_hot: Image3d, label_ids: list[int] | None = None) -> np.ndarray[Shape2d, np.dtype[int]]: ...

def one_hot_to_label(
    one_hot: np.ndarray[tuple[int, ...], np.dtype[bool]],
    label_ids: list[int] | None = None,
) -> np.ndarray[tuple[int, ...], np.dtype[int]]:
    """Convert one-hot encoded segmentation to label map.

    Converts a one-hot encoded segmentation array to discrete labels by taking
    the argmax along the last axis and optionally mapping to specific label values.

    Parameters
    ----------
    one_hot : np.ndarray of floats
        One-hot encoded segmentation array of shape (..., num_classes).
    label_ids : array_like of ints, optional
        List of label IDs to map classes to. If None, defaults to [0, FORNIX_LABEL, CC_LABEL].
        The index in this list corresponds to the class index from argmax.

    Returns
    -------
    npt.NDArray[int]
        Label map with discrete integer labels.
    """
    if label_ids is None:
        from CorpusCallosum.data.constants import CC_LABEL, FORNIX_LABEL
        label_ids = [0, FORNIX_LABEL, CC_LABEL]

    label = np.argmax(one_hot, axis=3)
    if label_ids is not None:
        label = np.asarray(label_ids)[label]

    return label


def run_inference_on_slice(
        model: "torch.nn.Module",
        test_slab: Image3d,
        ac_center: Vector2d,
        pc_center: Vector2d,
        voxel_size: tuple[float, float],
) -> tuple[np.ndarray[Shape3d, np.dtype[int]], Image4d, Image4d]:
    """Run inference on a single slice.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model for inference.
    test_slab : np.ndarray
        Input image slice.
    ac_center : npt.NDArray[float]
        Anterior commissure coordinates (Inferior and Anterior values).
    pc_center : npt.NDArray[float]
        Posterior commissure coordinates (Inferior and Posterior values).
    voxel_size : a pair of floats
        Voxel sizes in superior/inferior and anterior/posterior direction in mm.

    Returns
    -------
    results: np.ndarray
        Label map after one-hot conversion.
    inputs: np.ndarray
        Preprocessed input image.
    outputs_soft: npt.NDArray[float]
        Softlabel outputs (non-discrete).
    
    """
    # add zero in front of AC_center and PC_center
    ac_center = np.concatenate([np.zeros(1), ac_center])
    pc_center = np.concatenate([np.zeros(1), pc_center])

    _results, inputs, outputs_soft = run_inference(model, test_slab, ac_center, pc_center, voxel_size)
    results = one_hot_to_label(_results)

    return results, inputs, outputs_soft
