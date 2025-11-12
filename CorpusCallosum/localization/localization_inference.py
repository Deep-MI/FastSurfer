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

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from monai import transforms
from monai.networks.nets import DenseNet

from CorpusCallosum.data import constants
from CorpusCallosum.transforms.localization_transforms import CropAroundACPCFixedSize
from CorpusCallosum.utils.checkpoint import YAML_DEFAULT as CC_YAML
from FastSurferCNN.download_checkpoints import load_checkpoint_config_defaults
from FastSurferCNN.download_checkpoints import main as download_checkpoints


def load_model(device: torch.device | None = None) -> DenseNet:
    """Load trained numerical localization model from checkpoint.

    Parameters
    ----------
    checkpoint_path : str or Path or None, optional
        Path to model checkpoint, by default None.
        If None, downloads and uses default checkpoint.
    device : torch.device or None, optional
        Device to load model to, by default None.
        If None, uses CUDA if available, else CPU.

    Returns
    -------
    DenseNet
        Loaded and initialized model in evaluation mode
    """
    if device is None or device == "auto":
        from FastSurferCNN.utils.common import find_device
        device = find_device(device)
    
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
    checkpoint_path = constants.FASTSURFER_ROOT / cc_config['localization']

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
        CropAroundACPCFixedSize(
            keys=['image'], 
            fixed_size=(64, 64), 
            random_translate=0,
        ),
    ]
    return transforms.Compose(tr)


def preprocess_volume(
    image_volume: np.ndarray,
    center_pt: np.ndarray,
    transform: transforms.Transform | None = None
) -> dict[str, torch.Tensor]:
    """Preprocess a volume for inference.

    Parameters
    ----------
    image_volume : np.ndarray
        Input image volume
    center_pt : np.ndarray
        Center point coordinates for cropping
    transform : transforms.Transform or None, optional
        Custom transform pipeline, by default None.
        If None, uses default transforms from get_transforms().

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary containing preprocessed image tensor
    """
    if transform is None:
        transform = get_transforms()

    sample = {"image": image_volume, "AC_center": center_pt, "PC_center": center_pt}

    # Apply transforms
    transformed = transform(sample)
    
    # Add batch dimension if needed
    if torch.is_tensor(transformed["image"]):
        if len(transformed["image"].shape) == 3:
            transformed["image"] = transformed["image"].unsqueeze(0)
            
    return transformed

def run_inference(model: DenseNet,
                 image_volume: np.ndarray,
                 third_ventricle_center: np.ndarray,
                 device: torch.device | None = None,
                 transform: transforms.Transform | None = None
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    """
    Run inference on an image volume
    
    Parameters
    ----------
    model : torch.nn.Module
        Trained model for inference
    image_volume : np.ndarray
        Input volume as numpy array
    third_ventricle_center : np.ndarray
        Initial center point estimate for cropping
    device : torch.device, optional
        Device to run inference on, by default None
    transform : transforms.Transform, optional
        Custom transform pipeline, by default None

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
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # prepend zero to third_ventricle_center
    third_ventricle_center = np.concatenate([np.zeros(1), third_ventricle_center])
    
    # Preprocess
    t_dict = preprocess_volume(image_volume[None], third_ventricle_center, transform)


    transformed_original = t_dict['image']
    inputs = transformed_original.to(device)


    inputs = inputs.transpose(0, 1)
    batch_size, channels, height, width = inputs.shape
    inputs = inputs.unfold(0, 3, 1).swapdims(0, 1).reshape(-1, 3*channels, height, width)
    

    # Run inference
    with torch.no_grad():
        outputs = model(inputs)
        
        # Scale outputs to image size
        # img_size = torch.tensor([inputs.shape[2], inputs.shape[3], 
        #                        inputs.shape[2], inputs.shape[3]], 
        #                       dtype=torch.float32,
        #                       device=device)
        outputs = outputs * 64

    t_crops = [[t_dict['crop_left'], t_dict['crop_top'], t_dict['crop_left'], t_dict['crop_top']]]
    outs: np.ndarray = (outputs + torch.tensor(t_crops, dtype=outputs.dtype, device=outputs.device)).numpy()
    return outs[:, :2], outs[:, 2:], inputs.numpy(), tuple(int(t_dict[k].item()) for k in ['crop_left', 'crop_top'])


def run_inference_on_slice(model: DenseNet,
                           image_slice: np.ndarray, 
                           center_pt: np.ndarray, 
                           debug_output: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Run inference on a single slice to detect AC and PC points.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model for AC-PC detection
    image_slice : np.ndarray
        3D image slice to run inference on
    center_pt : np.ndarray
        Initial center point estimate for cropping
    debug_output : str, optional
        Path to save debug visualization, by default None

    Returns
    -------
    ac_coords : np.ndarray
        Detected AC coordinates with shape (2,) containing its [y,x] positions.
    pc_coords : np.ndarray
        Detected PC coordinates with shape (2,) containing its [y,x] positions.
    """

    # Run inference
    pc_coords, ac_coords, _, (crop_left, crop_top) = run_inference(model, image_slice, center_pt)
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
