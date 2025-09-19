import torch
import numpy as np
from monai import transforms
from monai.networks.nets import DenseNet as DenseNet_monai

from transforms.localization_transforms import CropAroundACPCFixedSize


def load_model(checkpoint_path, device=None):
    """
    Load the trained numerical localization model from checkpoint
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: torch device to load model to (defaults to CUDA if available)
    
    Returns:
        model: Loaded model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model architecture (must match training)
    model = DenseNet_monai( # densenet201
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
    
    # Load state dict
    if isinstance(checkpoint_path, str):
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
    else:
        state_dict = checkpoint_path

        
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model

def get_transforms():
    """Get preprocessing transforms for inference"""
    tr = [
        # transforms.LoadImaged(
        #     keys=['image'], 
        #     reader="NibabelReader", 
        #     image_only=True, 
        #     dtype=torch.float32, 
        #     ensure_channel_first=True
        # ),
        transforms.ScaleIntensityd(keys=['image'], minv=0, maxv=1),
        CropAroundACPCFixedSize(
            keys=['image'], 
            fixed_size=(64, 64), 
            random_translate=0
        ),
    ]
    return transforms.Compose(tr)

def preprocess_volume(image_volume, center_pt, transform=None):
    """
    Preprocess a volume for inference
    
    Args:
        image_volume: Input volume as numpy array or path to nifti file
        transform: Optional custom transform pipeline
    
    Returns:
        preprocessed: Preprocessed image tensor ready for model input
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

def run_inference(model, image_volume, third_ventricle_center, device=None, transform=None):
    """
    Run inference on an image volume
    
    Args:
        model: Trained model
        image_volume: Input volume as numpy array or path to nifti file
        device: torch device to run inference on
        transform: Optional custom transform pipeline
    
    Returns:
        dict containing predicted AC and PC coordinates in original image space
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
    views = []
    for i in range(batch_size - 2):  # -2 to ensure we have 3 slices per view
        view = inputs[i:i+3]  # Take 3 consecutive slices
        view = view.reshape(1, 3*channels, height, width)  # Reshape to combine slices into channels
        views.append(view)

    inputs = torch.cat(views, dim=0)  # Stack all views into batch dimension
    

    # Run inference
    with torch.no_grad():
        outputs = model(inputs)
        
        # Scale outputs to image size
        # img_size = torch.tensor([inputs.shape[2], inputs.shape[3], 
        #                        inputs.shape[2], inputs.shape[3]], 
        #                       dtype=torch.float32,
        #                       device=device)
        outputs = outputs * 64
        
    outputs[:, 0] += t_dict['crop_left']
    outputs[:, 1] += t_dict['crop_top']
    outputs[:, 2] += t_dict['crop_left']
    outputs[:, 3] += t_dict['crop_top']


    return (outputs[:,:2].cpu().numpy(), 
            outputs[:,2:].cpu().numpy(), 
            inputs.cpu().numpy(), 
            (t_dict['crop_left'], t_dict['crop_top']))


def run_inference_on_slice(model, image_slice, center_pt, debug_output=None):

    # Run inference
    pc_coords, ac_coords, inputs, (crop_left, crop_top) = run_inference(model, image_slice, center_pt)
    center_pt = np.mean(np.concatenate([ac_coords, pc_coords], axis=0), axis=0)
    pc_coords, ac_coords, inputs, (crop_left, crop_top) = run_inference(model, image_slice, center_pt)
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
