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

from pathlib import Path

import numpy as np
import torch
from monai import transforms
from monai.networks.nets import DenseNet
from numpy import typing as npt

from CorpusCallosum.transforms.localization_transforms import CropAroundACPCFixedSize
from CorpusCallosum.utils.checkpoint import YAML_DEFAULT as CC_YAML
from FastSurferCNN.download_checkpoints import load_checkpoint_config_defaults
from FastSurferCNN.download_checkpoints import main as download_checkpoints
from FastSurferCNN.utils.parser_defaults import FASTSURFER_ROOT

PATCH_SIZE = (64, 64)


def load_model(device: torch.device) -> DenseNet:
    """Load trained numerical localization model from checkpoint.

    Parameters
    ----------
    device : torch.device
        Device to load model to.

    Returns
    -------
    DenseNet
        Loaded and initialized model in evaluation mode.
    """

    # Initialize model architecture (must match training)
    model = DenseNet( # densenet201
        spatial_dims=2,
        in_channels=3,
        out_channels=4,
        init_features=64,
        growth_rate=32,
        block_config=(6, 12, 48, 32),
        bn_size=4,
        act=("relu", {"inplace": True}),
        norm=("batch", {"affine": True}),
        dropout_prob=0.2
    )

    download_checkpoints(cc=True)
    cc_config = load_checkpoint_config_defaults(
        "checkpoint",
        filename=CC_YAML,
    )
    checkpoint_path = FASTSURFER_ROOT / cc_config['localization']

    # Load state dict
    if isinstance(checkpoint_path, str) or isinstance(checkpoint_path, Path):
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
    else:
        state_dict = checkpoint_path
        
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def get_transforms() -> transforms.Compose:
    """Get preprocessing transforms for inference.

    Returns
    -------
    transforms.Compose
        Composed transform pipeline including:
        - Intensity scaling to [0,1]
        - Fixed size cropping around AC-PC points
    """
    tr = [
        transforms.ScaleIntensityd(keys=['image'], minv=0, maxv=1),
        CropAroundACPCFixedSize(keys=['image'], fixed_size=PATCH_SIZE, random_translate=0),
    ]
    return transforms.Compose(tr)


def preprocess_volume(
    image_volume: np.ndarray,
    center_pt: npt.NDArray[float],
    transform: transforms.Transform | None = None
) -> dict[str, torch.Tensor]:
    """Preprocess a volume for inference.

    Parameters
    ----------
    image_volume : np.ndarray
        Input image volume of shape (W, W, D) in RAS.
    center_pt : np.ndarray
        Center point coordinates for cropping on the slice with shape (3,).
    transform : transforms.Transform or None, optional
        Custom transform pipeline, by default None.
        If None, uses default transforms from get_transforms().

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary containing preprocessed image tensor.
    """
    if transform is None:
        transform = get_transforms()

    # During training we used AC/PC coordinates, but during inference
    # we approximate this by the center of the third ventricle.
    # Therefore we put in the third ventricle center as dummy AC/PC coordinates
    # for cropping the image.
    sample = {"image": image_volume[None], "AC_center": center_pt[1:][None], "PC_center": center_pt[1:][None]}

    # Apply transforms
    transformed = transform(sample)
    
    # Add batch dimension if needed
    if torch.is_tensor(transformed["image"]):
        if transformed["image"].ndim == 3:
            transformed["image"] = transformed["image"].unsqueeze(0)
            
    return transformed

def run_inference(
        model: torch.nn.Module,
        image_volume: np.ndarray,
        third_ventricle_center: np.ndarray,
        device: torch.device | None = None,
        transform: transforms.Transform | None = None
    ) -> tuple[npt.NDArray[float], npt.NDArray[float], np.ndarray, tuple[int, int]]:
    """
    Run inference on an image volume
    
    Parameters
    ----------
    model : DenseNet
        Trained model for inference.
    image_volume : np.ndarray
        Input volume as numpy array.
    third_ventricle_center : np.ndarray
        Initial center point estimate for cropping.
    device : torch.device, optional
        Device to run inference on, by default None.
    transform : transforms.Transform, optional
        Custom transform pipeline, defaults to preconfigured transforms of `get_transforms`.

    Returns
    -------
    pc_ccord : np.ndarray
        Predicted PC coordinates.
    ac_coord : np.ndarray
        Predicted AC coordinates.
    image : np.ndarray
        Processed input images.
    crop_offsets : tuple[int, int]
        Crop offsets (left, top).
    """
    if device is None:
        device = next(model.parameters()).device

    # prepend zero to third_ventricle_center
    third_ventricle_center = np.concatenate([np.zeros(1), third_ventricle_center])
    
    # Preprocess
    t_dict = preprocess_volume(image_volume, third_ventricle_center, transform)

    transformed_original = t_dict['image']
    inputs = transformed_original.to(device)

    inputs = inputs.transpose(0, 1)
    batch_size, channels, height, width = inputs.shape
    inputs = inputs.unfold(0, 3, 1).swapdims(0, 1).reshape(-1, 3*channels, height, width)

    # Run inference
    with torch.no_grad():
        outputs = model(inputs) * torch.as_tensor([PATCH_SIZE + PATCH_SIZE], device=device)

    t_crops = [(t_dict['crop_left'] + t_dict['crop_top']) * 2]
    outs: npt.NDArray[float] = outputs.cpu().numpy() + np.asarray(t_crops, dtype=float)
    return outs[:, :2], outs[:, 2:], inputs.cpu().numpy(), (t_dict["crop_left"][0], t_dict["crop_top"][0])


def run_inference_on_slice(
        model: DenseNet,
        image_slice: np.ndarray,
        center_pt: np.ndarray,
        debug_output: str | None = None,
) -> tuple[npt.NDArray[float], npt.NDArray[float]]:
    """Run inference on a single slice to detect AC and PC points.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model for AC-PC detection.
    image_slice : np.ndarray
        3D image mid-slices to run inference on in RAS.
    center_pt : np.ndarray
        Initial center point estimate for cropping.
    debug_output : str, optional
        Path to save debug visualization, by default None.

    Returns
    -------
    ac_coords : np.ndarray
        Detected AC coordinates with shape (2,) containing its [y,x] positions.
    pc_coords : np.ndarray
        Detected PC coordinates with shape (2,) containing its [y,x] positions.
    """

    # Run inference
    pc_coords, ac_coords, *_ = run_inference(model, image_slice, center_pt)
    center_pt = np.mean(np.concatenate([ac_coords, pc_coords], axis=0), axis=0)
    pc_coords, ac_coords, _, (crop_left, crop_top) = run_inference(model, image_slice, center_pt)
    pc_coords = np.mean(pc_coords, axis=0)
    ac_coords = np.mean(ac_coords, axis=0)

    if debug_output is not None:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.imshow(image_slice[image_slice.shape[0]//2, :, :], cmap='gray')
        # Plot points on all views
        ax.scatter(pc_coords[1], pc_coords[0], c='r', marker='x', label='PC')
        ax.scatter(ac_coords[1], ac_coords[0], c='b', marker='x', label='AC')
        # make a box where the crop is
        ax.add_patch(Rectangle((crop_top, crop_left), 64, 64, fill=False, color='r', linewidth=2))        
        plt.savefig(debug_output, bbox_inches='tight')
        plt.close()


    return ac_coords, pc_coords
