import nibabel as nib
import numpy as np
import torch
from monai import transforms

from CorpusCallosum.transforms.segmentation_transforms import CropAroundACPC
from CorpusCallosum.utils.checkpoint import YAML_DEFAULT as CC_YAML
from FastSurferCNN.download_checkpoints import load_checkpoint_config_defaults
from FastSurferCNN.download_checkpoints import main as download_checkpoints
from FastSurferCNN.models.networks import FastSurferVINN


def load_model(checkpoint_path, device=None):
    """
    Load the trained model from checkpoint
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: torch device to load model to (defaults to CUDA if available)
    
    Returns:
        model: Loaded model
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
    checkpoint_path = cc_config['segmentation']
    
    #model = torch.load(checkpoint_path, map_location=device, weights_only=False)
    weights = torch.load(checkpoint_path, weights_only=True, map_location=device)
    model.load_state_dict(weights)
    model.eval()
    model.to(device)
    return model

def run_inference(model, image_slice, AC_center, PC_center, voxel_size, device=None, transform=None):
    """
    Run inference on a single image slice
    
    Args:
        model: Trained model
        image_slice: Input image as numpy array
        device: torch device to run inference on
        transform: Optional custom transform pipeline
    
    Returns:
        dict containing:
            segmentation: Segmentation map if model produces segmentation
            landmarks: Predicted landmarks if model produces localization
    """
    orig_shape = image_slice.shape
    
    if device is None:
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = next(model.parameters()).device
        
    def crop_around_acpc(img, ac, pc, vox_size):
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
    views = []
    for i in range(batch_size - 8):  # -8 to ensure we have 9 slices per view
        view = inputs[i:i+9]  # Take 9 consecutive slices
        view = view.reshape(1, 9*channels, height, width)  # Reshape to combine slices into channels
        views.append(view)
        
    inputs = torch.cat(views, dim=0)  # Stack all views into batch dimension

    # Post-process outputs
    with torch.no_grad():
        scale_factors = torch.ones((inputs.shape[0], 2), device=device) * (1 / voxel_size)
        
        outputs = model(inputs, scale_factor=scale_factors)
        
        # average the outputs along the batch dimension
        outputs_avg = torch.mean(outputs, dim=0).unsqueeze(0)
    
        outputs_soft = outputs.cpu().numpy() #transforms.Activations(softmax=True)(outputs) # non_discrete outputs
        outputs = torch.stack([post_trans(i) for i in outputs])
        outputs_avg = torch.stack([post_trans(i) for i in outputs_avg])
        
        pad_left, pad_right, pad_top, pad_bottom = to_pad
        # Pad back to original size
        outputs = np.pad(outputs, ((0,0), (0,0), (pad_left.item(), pad_right.item()), 
                                   (pad_top.item(), pad_bottom.item())), mode='constant', constant_values=0)
        outputs_avg = np.pad(outputs_avg, ((0,0), (0,0), (pad_left.item(), pad_right.item()), 
                                           (pad_top.item(), pad_bottom.item())), mode='constant', constant_values=0)
        outputs_soft = np.pad(outputs_soft, ((0,0), (0,0), (pad_left.item(), pad_right.item()), 
                                             (pad_top.item(), pad_bottom.item())), mode='constant', constant_values=0)
    
    # restore original shape
    if orig_shape[-2:] != outputs.shape[-2:]:
        new_outputs = np.zeros((outputs.shape[0], outputs.shape[1], orig_shape[-2], orig_shape[-1]))
        new_outputs[:,:,:256,:256] = outputs
        outputs = new_outputs
        
        new_outputs_avg = np.zeros((outputs_avg.shape[0], outputs_avg.shape[1], orig_shape[-2], orig_shape[-1]))
        new_outputs_avg[:,:,:256,:256] = outputs_avg
        outputs_avg = new_outputs_avg

        new_outputs_soft = np.zeros((outputs_soft.shape[0], outputs_soft.shape[1], 
                                     orig_shape[-2], orig_shape[-1]), dtype=np.float32)
        new_outputs_soft[:,:,:256,:256] = outputs_soft
        outputs_soft = new_outputs_soft

    return (
        outputs.transpose(0,2,3,1),
        inputs.cpu().numpy().transpose(0,2,3,1),
        outputs_avg.transpose(0,2,3,1),
        outputs_soft.transpose(0,2,3,1),
    )

# TODO: load validation data and run inference on it to confirm correct processing


def load_validation_data(path):
    import pandas as pd
    data = pd.read_csv(path, index_col=0, header=None)
    data.columns = ["image", "label", "AC_center_x", "AC_center_y", "AC_center_z", 
                    "PC_center_x", "PC_center_y", "PC_center_z"]
    
    ac_centers = data[["AC_center_x", "AC_center_y", "AC_center_z"]].values
    pc_centers = data[["PC_center_x", "PC_center_y", "PC_center_z"]].values
    images = data["image"].values
    labels = data["label"].values
    subj_ids = data.index.values.tolist()

    label_widths = []
    for label_path in data['label']:
        label_img =nib.load(label_path)
        
        if label_img.shape[0] > 100:
            # check which slices have non-zero values
            label = label_img.get_fdata()
            non_zero_slices = np.any(label > 0, axis=(1,2))
            first_nonzero = np.argmax(non_zero_slices)
            last_nonzero = len(non_zero_slices) - np.argmax(non_zero_slices[::-1])
            label_widths.append(last_nonzero - first_nonzero)
        else:
            label_widths.append(label_img.shape[0])
        
    
    
    return images, ac_centers, pc_centers, label_widths, labels, subj_ids


def one_hot_to_label(one_hot, label_ids=None):
    if label_ids is None:
        label_ids = [0, 192, 250]
    label = np.argmax(one_hot, axis=3)
    if label_ids is not None:
        label = np.where(label == 0, label_ids[0], label)
        label = np.where(label == 1, label_ids[1], label)
        label = np.where(label == 2, label_ids[2], label)

    return label


# TODO: add heuristic that removes islands that are far away



def run_inference_on_slice(model, test_slice, AC_center, PC_center, voxel_size):

    # add zero in front of AC_center and PC_center
    AC_center = np.concatenate([np.zeros(1), AC_center])
    PC_center = np.concatenate([np.zeros(1), PC_center])

    results, inputs, outputs_avg, outputs_soft = run_inference(model, test_slice, AC_center, PC_center, voxel_size)
    results = one_hot_to_label(results)

    return results, inputs, outputs_avg, outputs_soft



def remove_small_clusters(label_data, min_cluster_size=100):
    """
    Removes small clusters of connected components from a label image.
    
    Args:
        label_data: numpy array containing the label data
        min_cluster_size: minimum size of clusters to keep (default: 100)
        
    Returns:
        cleaned_label: numpy array with small clusters removed
    """
    from scipy.ndimage import label as ndlabel
    

    list_of_cleaned_labels = []

    for label_id in range(label_data.shape[1]-1):

        # Create a binary mask of the label
        binary_mask = label_data[:,label_id+1] > 0

        
        # Label the connected components
        labeled_array, num_features = ndlabel(binary_mask)
        
        # Create a mask for small clusters
        small_clusters_mask = np.zeros_like(binary_mask, dtype=bool)
        for i in range(1, num_features + 1):
            small_cluster = (labeled_array == i)
            if np.sum(small_cluster) < min_cluster_size:
                small_clusters_mask |= small_cluster
        
        # Remove small clusters from the original label
        cleaned_label = label_data[:,label_id+1].copy()
        cleaned_label[small_clusters_mask] = 0
        list_of_cleaned_labels.append(cleaned_label)


        # plot binary mask
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(2,len(binary_mask))
        # for i in range(len(binary_mask)):
        #     ax[0,i].imshow(binary_mask[i])
        #     ax[1,i].imshow(cleaned_label[i])
        # plt.show()
        
    return np.stack([label_data[:,0]]+list_of_cleaned_labels, axis=1)

