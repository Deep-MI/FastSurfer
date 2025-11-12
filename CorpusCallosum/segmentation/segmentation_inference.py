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

import nibabel as nib
import numpy as np
import torch
from monai import transforms

from CorpusCallosum.data import constants
from CorpusCallosum.transforms.segmentation_transforms import CropAroundACPC
from CorpusCallosum.utils.checkpoint import YAML_DEFAULT as CC_YAML
from FastSurferCNN.download_checkpoints import load_checkpoint_config_defaults
from FastSurferCNN.download_checkpoints import main as download_checkpoints
from FastSurferCNN.models.networks import FastSurferVINN


def load_model(device: torch.device | None = None) -> FastSurferVINN:
    """Load trained model from checkpoint.

    Parameters
    ----------
    checkpoint_path : str or None, optional
        Path to model checkpoint, by default None.
        If None, downloads and uses default checkpoint.
    device : torch.device or None, optional
        Device to load model to, by default None.
        If None, uses CUDA if available, else CPU.

    Returns
    -------
    FastSurferVINN
        Loaded and initialized model in evaluation mode
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
    cc_config = load_checkpoint_config_defaults(
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
    model: FastSurferVINN,
    image_slice: np.ndarray,
    AC_center: np.ndarray,
    PC_center: np.ndarray,
    voxel_size: float,
    device: torch.device | None = None,
    transform: transforms.Transform | None = None
) -> dict[str, np.ndarray]:
    """Run inference on a single image slice.

    Parameters
    ----------
    model : FastSurferVINN
        Trained model
    image_slice : np.ndarray
        Input image as numpy array
    AC_center : np.ndarray
        Anterior commissure coordinates
    PC_center : np.ndarray
        Posterior commissure coordinates
    voxel_size : float
        Voxel size in mm
    device : torch.device or None, optional
        Device to run inference on, by default None.
        If None, uses the device of the model.
    transform : transforms.Transform or None, optional
        Custom transform pipeline, by default None

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary containing:
        - segmentation : Binary segmentation map
        - landmarks : Predicted landmark coordinates
    """
    if device is None:
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = next(model.parameters()).device
        
    def crop_around_acpc(img: np.ndarray, 
                    ac: np.ndarray, 
                    pc: np.ndarray, 
                    vox_size: float) -> dict[str, np.ndarray]:
        """Crop image around AC-PC points.

        Parameters
        ----------
        img : np.ndarray
            Input image
        ac : np.ndarray
            Anterior commissure coordinates
        pc : np.ndarray
            Posterior commissure coordinates
        vox_size : float
            Voxel size in mm

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary containing cropped image and metadata
        """
        return CropAroundACPC(keys=['image'], padding_mm=35, random_translate=0)(
            {'image': img, 'AC_center': ac, 'PC_center': pc, 'res': vox_size}
        )

    # Preprocess slice
    inputs = torch.from_numpy(image_slice[:,None,:256,:256]) # artifact from training script
    crop_dict = crop_around_acpc(inputs, AC_center, PC_center, voxel_size)
    inputs, to_pad = crop_dict['image'], crop_dict['to_pad']
    inputs = transforms.utils.rescale_array(inputs, 0, 1, dtype=np.float32)
    inputs = inputs.to(device)

    post_trans = transforms.Compose(
        [transforms.Activations(softmax=True), transforms.AsDiscrete(argmax=True, to_onehot=3)]
    )

    # split into slices with 9 channels each
    # Generate views with sliding window of 9 slices
    batch_size, channels, height, width = inputs.shape
    inputs = inputs.unfold(0, 9, 1).swapdims(0, 1).reshape(-1, 9*channels, height, width)

    # Post-process outputs
    with torch.no_grad():
        scale_factors = torch.ones((inputs.shape[0], 2), device=device) / voxel_size
        
        outputs = model(inputs, scale_factor=scale_factors)
        
        # average the outputs along the batch dimension
        outputs_avg = torch.mean(outputs, dim=0, keepdim=True)
    
        outputs_soft = outputs.cpu().numpy() #transforms.Activations(softmax=True)(outputs) # non_discrete outputs
        outputs = torch.stack([post_trans(i) for i in outputs])
        outputs_avg = torch.stack([post_trans(i) for i in outputs_avg])
        
        # Pad back to original size, to_pad is a tuple[int, int, int, int]
        pad_tuples = ((0, 0),) * 2 + (to_pad[:2], to_pad[2:])
        outputs = np.pad(outputs, pad_tuples, mode='constant', constant_values=0)
        outputs_avg = np.pad(outputs_avg, pad_tuples, mode='constant', constant_values=0)
        outputs_soft = np.pad(outputs_soft, pad_tuples, mode='constant', constant_values=0)    

    return (
        outputs.transpose(0,2,3,1),
        inputs.cpu().numpy().transpose(0,2,3,1),
        outputs_avg.transpose(0,2,3,1),
        outputs_soft.transpose(0,2,3,1),
    )


def load_validation_data(path):
    from concurrent.futures import ThreadPoolExecutor

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
        label_img = nib.load(label_path)
        
        if label_img.shape[0] > 100:
            # check which slices have non-zero values
            label_data = np.asarray(label_img.dataobj)
            non_zero_slices = np.any(label_data > 0, axis=(1,2))
            first_nonzero = np.argmax(non_zero_slices)
            last_nonzero = len(non_zero_slices) - np.argmax(non_zero_slices[::-1])
            return last_nonzero - first_nonzero
        else:
            return label_img.shape[0]
    label_widths = ThreadPoolExecutor().map(_load, data["label"])    
    
    return images, ac_centers, pc_centers, label_widths, labels, subj_ids


def one_hot_to_label(one_hot, label_ids=None):
    if label_ids is None:
        label_ids = [0, 192, 250]
    label = np.argmax(one_hot, axis=3)
    if label_ids is not None:
        label = np.asarray(label_ids)[label]
    return label



def run_inference_on_slice(model: FastSurferVINN, 
                          test_slice: np.ndarray,
                          AC_center: np.ndarray, 
                          PC_center: np.ndarray,
                          voxel_size: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run inference on a single slice.

    Parameters
    ----------
    model : FastSurferVINN
        Trained model for inference
    test_slice : np.ndarray
        Input image slice
    AC_center : np.ndarray
        Anterior commissure coordinates
    PC_center : np.ndarray
        Posterior commissure coordinates 
    voxel_size : float
        Voxel size in mm

    Returns
    -------
    results: np.ndarray
        Label map after one-hot conversion
    inputs: np.ndarray
        Preprocessed input image
    outputs_avg: np.ndarray
        Averaged model outputs
    outputs_soft: np.ndarray
        Softlabel outputs (non-discrete)
    
    """
    # add zero in front of AC_center and PC_center
    AC_center = np.concatenate([np.zeros(1), AC_center])
    PC_center = np.concatenate([np.zeros(1), PC_center])

    results, inputs, outputs_avg, outputs_soft = run_inference(model, test_slice, AC_center, PC_center, voxel_size)
    results = one_hot_to_label(results)

    return results, inputs, outputs_avg, outputs_soft
