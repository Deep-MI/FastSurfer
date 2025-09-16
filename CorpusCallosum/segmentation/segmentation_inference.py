import time
import torch
import numpy as np
import nibabel as nib

from monai import transforms
from monai.metrics import DiceMetric, HausdorffDistanceMetric

from FastSurferCNN.models.networks import FastSurferVINN
from transforms.segmentation_transforms import CropAroundACPC, UncropAroundACPC


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
        
    crop_around_acpc = lambda img, ac, pc, vox_size: CropAroundACPC(keys=['image'], padding_mm=35, random_translate=0)({'image': img, 'AC_center': ac, 'PC_center': pc, 'res': vox_size})

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
        outputs = np.pad(outputs, ((0,0), (0,0), (pad_left.item(), pad_right.item()), (pad_top.item(), pad_bottom.item())), mode='constant', constant_values=0)
        outputs_avg = np.pad(outputs_avg, ((0,0), (0,0), (pad_left.item(), pad_right.item()), (pad_top.item(), pad_bottom.item())), mode='constant', constant_values=0)
        outputs_soft = np.pad(outputs_soft, ((0,0), (0,0), (pad_left.item(), pad_right.item()), (pad_top.item(), pad_bottom.item())), mode='constant', constant_values=0)
    
    # restore original shape
    if orig_shape[-2:] != outputs.shape[-2:]:
        new_outputs = np.zeros((outputs.shape[0], outputs.shape[1], orig_shape[-2], orig_shape[-1]))
        new_outputs[:,:,:256,:256] = outputs
        outputs = new_outputs
        
        new_outputs_avg = np.zeros((outputs_avg.shape[0], outputs_avg.shape[1], orig_shape[-2], orig_shape[-1]))
        new_outputs_avg[:,:,:256,:256] = outputs_avg
        outputs_avg = new_outputs_avg

        new_outputs_soft = np.zeros((outputs_soft.shape[0], outputs_soft.shape[1], orig_shape[-2], orig_shape[-1]), dtype=np.float32)
        new_outputs_soft[:,:,:256,:256] = outputs_soft
        outputs_soft = new_outputs_soft

    return outputs.transpose(0,2,3,1), inputs.cpu().numpy().transpose(0,2,3,1), outputs_avg.transpose(0,2,3,1), outputs_soft.transpose(0,2,3,1)

# TODO: load validation data and run inference on it to confirm correct processing


def load_validation_data(path):
    import pandas as pd
    data = pd.read_csv(path, index_col=0, header=None)
    data.columns = ["image", "label", "AC_center_x", "AC_center_y", "AC_center_z", "PC_center_x", "PC_center_y", "PC_center_z"]
    
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


def one_hot_to_label(one_hot, label_ids=[0,192,250]):
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



def run_validation():
    
    # Load model

    model_path = "/groups/ag-reuter/projects/corpus_callosum_fornix/pollakc/cc_pipeline/weights/segmentation_weights_cc_fn.pth"
    # /groups/ag-reuter/projects/corpus_callosum_fornix/pollakc/network/experiments/CCFN_softmax01/checkpoints/best_metric_model.pth

    model = load_model(model_path)
    
    # Load a test image slice
    #test_img = nib.load("/groups/ag-reuter/projects/corpus_callosum_fornix/label_QC/added_images/48e2d11f/orig_up.mgz")
    
    val_images, val_ac, val_pc, val_label_widths, val_labels, val_subj_ids = load_validation_data("/groups/ag-reuter/projects/corpus_callosum_fornix/pollakc/network/data/difficult_joined_labels.csv")
    
    dice_out = []
    dice_out_single_slice = []
    dice_out_dict = {}
    dice_out_single_slice_dict = {}
    
    # Initialize Hausdorff distance metric
    hd_out = []
    hd_out_single_slice = []
    hd_out_dict = {}
    hd_out_single_slice_dict = {}
    
    for img_path, AC_center, PC_center, label_width, label_path, subj_id in zip(val_images, val_ac, val_pc, val_label_widths, val_labels, val_subj_ids):

        # if subj_id != "abf05659":
        #     continue


        label_width = 5

        test_img = nib.load(img_path)
        test_slice = test_img.get_fdata()
            
        # crop to middle 9+5-1 (13) slices 
        test_slice = test_slice[256//2-label_width//2-4:256//2+label_width//2+5]
    

        # Run inference
        start_time = time.time()
        results, inputs, outputs_avg, outputs_soft = run_inference(model, test_slice, AC_center, PC_center, voxel_size=test_img.header.get_zooms()[0])
        inference_time = time.time() - start_time
        print(f"Inference took {inference_time:.3f} seconds")
        
        label_img = nib.load(label_path)
        label = label_img.get_fdata()
        
        # calculate dice score
        dice_metric = DiceMetric(include_background=False, reduction="mean")
        hd_metric = HausdorffDistanceMetric(include_background=False, percentile=95.0, reduction="mean")
        
        # Convert label to one-hot format
        label_tensor = torch.from_numpy(label)
        
        if label_tensor.shape[0] > 100:
            # select non-zero slices
            label_tensor = label_tensor[label_tensor.any(axis=(1,2))]

        # crop to label width
        label_tensor = label_tensor[label_tensor.shape[0]//2-label_width//2:label_tensor.shape[0]//2+label_width//2+1]
        
        # map to 0,1,2
        ids = np.unique(label)
        label_tensor = torch.where(label_tensor == ids[0], 0, label_tensor)
        label_tensor = torch.where(label_tensor == ids[1], 1, label_tensor)
        label_tensor = torch.where(label_tensor == ids[2], 2, label_tensor)
        
        label_onehot = torch.nn.functional.one_hot(label_tensor.long(), num_classes=3)  # Convert to one-hot with 3 classes
        label_onehot = label_onehot.permute(0, 3, 1, 2)  # Move class dimension to second position (B,C, H, W)
        #label_onehot = label_onehot[:,:,:256,:256]
        
        # Reshape results to (B, C, H, W)
        results_tensor = torch.from_numpy(results)
        results_tensor = results_tensor.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)

        # Remove small clusters
        results_tensor = remove_small_clusters(results_tensor.numpy(), min_cluster_size=100)
        results_tensor = torch.from_numpy(results_tensor)

        
        
        # Calculate Dice score
        dice_score = dice_metric(results_tensor, label_onehot)
        midslice = results_tensor.shape[0]//2
        dice_single_slice = dice_metric(results_tensor[None,midslice], label_onehot[None,midslice])
        
        # Calculate Hausdorff distance
        # Get physical spacing from the image header for accurate distance calculation
        spacing = test_img.header.get_zooms()[:3]  # Get voxel dimensions in mm
        if len(spacing) == 3:
            # Use only in-plane spacing for 2D slices
            spacing_tensor = torch.tensor([spacing[1], spacing[2]], dtype=torch.float32)
        else:
            spacing_tensor = torch.tensor(spacing, dtype=torch.float32)
            
        hd_score = hd_metric(results_tensor, label_onehot, spacing=spacing_tensor.numpy().tolist())
        hd_single_slice = hd_metric(results_tensor[None,midslice], label_onehot[None,midslice], spacing=spacing_tensor.numpy().tolist())
        
        # Store results
        dice_out.append(dice_score.mean(axis=0).numpy().tolist())
        dice_out_single_slice.append(dice_single_slice.numpy().tolist())
        dice_out_dict[subj_id] = dice_score.mean(axis=0).numpy().tolist()
        dice_out_single_slice_dict[subj_id] = dice_single_slice.numpy()[0].tolist()
        
        hd_out.append(hd_score.mean(axis=0).numpy().tolist())
        hd_out_single_slice.append(hd_single_slice.numpy().tolist())
        hd_out_dict[subj_id] = hd_score.mean(axis=0).numpy().tolist()
        hd_out_single_slice_dict[subj_id] = hd_single_slice.numpy()[0].tolist()
        
        print(f"Subject: {subj_id}")
        print(f"Dice mean: {[f'{x:.3f}' for x in dice_score.mean(axis=0).numpy().tolist()]}")
        print(f"HD95 mean: {[f'{x:.3f}' for x in hd_score.mean(axis=0).numpy().tolist()]} mm")


        
        
        # Convert numpy array to NIfTI image before saving
        nifti_img_out = nib.Nifti1Image(results, affine=test_img.affine, header=test_img.header)
        nifti_img_in = nib.Nifti1Image(inputs, affine=test_img.affine, header=test_img.header)
        nifti_orig_slice = nib.Nifti1Image(test_slice[4:-4], affine=test_img.affine, header=test_img.header)
        nifti_avg_slice = nib.Nifti1Image(outputs_avg, affine=test_img.affine, header=test_img.header)
        nifti_label = nib.Nifti1Image(label, affine=test_img.affine, header=test_img.header)
        nifti_final_out = nib.Nifti1Image(one_hot_to_label(results), affine=test_img.affine, header=test_img.header)
        nib.save(nifti_img_in, "/workspace/outputs/segmentation_input.nii.gz")
        nib.save(nifti_img_out, "/workspace/outputs/segmentation.nii.gz")
        nib.save(nifti_orig_slice, "/workspace/outputs/segmentation_orig.nii.gz")
        nib.save(nifti_avg_slice, "/workspace/outputs/segmentation_avg.nii.gz")
        nib.save(nifti_label, "/workspace/outputs/segmentation_label.nii.gz")
        nib.save(nifti_final_out, "/workspace/outputs/segmentation_final.nii.gz")
        import shutil
        shutil.copy(img_path, "/workspace/outputs/segmentation_orig.mgz")
        shutil.copy(label_path, "/workspace/outputs/segmentation_label.mgz")

        




    print(f'Overall Validation Dice: {[f"{x:.3f}" for x in np.mean(dice_out, axis=0).tolist()]}')
    print(f'Overall Validation HD95: {[f"{x:.3f}" for x in np.mean(hd_out, axis=0).tolist()]} mm')

    import pandas as pd
    # Save Dice scores
    dice_out_df = pd.DataFrame.from_dict(dice_out_dict, orient='index', columns=["CC", "FN"])
    dice_single_slice_df = pd.DataFrame.from_dict(dice_out_single_slice_dict, orient='index', columns=["CC", "FN"])
    dice_out_df.to_csv("/workspace/outputs/dice_out.csv")
    dice_single_slice_df.to_csv("/workspace/outputs/dice_single_slice.csv")
    
    # Save Hausdorff distances
    hd_out_df = pd.DataFrame.from_dict(hd_out_dict, orient='index', columns=["CC", "FN"])
    hd_single_slice_df = pd.DataFrame.from_dict(hd_out_single_slice_dict, orient='index', columns=["CC", "FN"])
    hd_out_df.to_csv("/workspace/outputs/hd_out.csv")
    hd_single_slice_df.to_csv("/workspace/outputs/hd_single_slice.csv")
    
    # Create a combined metrics dataframe
    combined_metrics = pd.DataFrame()
    combined_metrics['Dice_CC'] = dice_out_df['CC']
    combined_metrics['Dice_FN'] = dice_out_df['FN']
    combined_metrics['HD95_CC'] = hd_out_df['CC']
    combined_metrics['HD95_FN'] = hd_out_df['FN']
    combined_metrics.to_csv("/workspace/outputs/combined_metrics.csv")

    # Testset: Overall Dice: ['0.957', '0.829']   HD95: ['1.018', '2.799']
    # Testset only 5 slices: Overall Validation Dice: ['0.957', '0.831']  HD95: ['1.025', '2.318']
    # Difficultset: Overall Validation Dice: ['0.944', '0.785']   HD95: ['1.189', '4.080']
    # Difficultset only 5 slices: Overall Validation Dice: ['0.946', '0.784']  HD95: ['1.155', '4.101']


if __name__ == "__main__":
    run_validation()