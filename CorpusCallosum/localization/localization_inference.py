import time
import torch
import numpy as np
import nibabel as nib
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


    # model = torch.nn.DataParallel(model)
    # model.load_state_dict(state_dict)
    # model = model.module
    # torch.save(model.state_dict(), '/workspace/weights/localization_weights1.pth')
        
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


    return outputs[:,:2].cpu().numpy(), outputs[:,2:].cpu().numpy(), inputs.cpu().numpy(), (t_dict['crop_left'], t_dict['crop_top'])

def load_validation_data(path):
    import pandas as pd
    data = pd.read_csv(path, index_col=0, header=None)
    data.columns = ["image", "label", "AC_center_x", "AC_center_y", "AC_center_z", "PC_center_x", "PC_center_y", "PC_center_z"]

    data = data.drop(['15656','5bd8d9b2-e0d3-4a40-b00c-03dfffc5b206'], errors='ignore')
    
    ac_centers = data[["AC_center_x", "AC_center_y", "AC_center_z"]].values
    pc_centers = data[["PC_center_x", "PC_center_y", "PC_center_z"]].values
    images = data["image"].values

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


    extended_data = pd.read_csv("/groups/ag-reuter/projects/corpus_callosum_fornix/pollakc/network/data/found_labels_with_meta_data_difficult_final.csv", index_col=0)
    extended_data = extended_data.loc[data.index]

    third_ventricle_centers = []
    vox_sizes = []
    for aseg_up in extended_data['aseg_up_nocc']:
        aseg_up_img = nib.load(aseg_up)
        aseg_up_data = aseg_up_img.get_fdata()

        aseg_up_mid = aseg_up_data.shape[0] // 2

        tv_center = np.mean(np.argwhere(aseg_up_data == 14), axis=0)[1:]

        if np.isnan(tv_center).any():
            import pdb; pdb.set_trace()

        third_ventricle_centers.append(tv_center)
        vox_sizes.append(np.prod(aseg_up_img.header.get_zooms()[1]))


    subj_ids = data.index.values
    
    return images, ac_centers, pc_centers, label_widths, third_ventricle_centers, vox_sizes, subj_ids



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




# TODO: add check if the prediction of first and second round diverges too much
    
def run_validation():
    from matplotlib import pyplot as plt
    from matplotlib.patches import Rectangle
    
    # Load model
    #model_path = "/groups/ag-reuter/projects/corpus_callosum_fornix/pollakc/network/experiments_localization_2/finetune_03_fixweights/checkpoints/best_metric_model.pth"
    model_path = '/workspace/weights/localization_weights_acpc.pth'

    model = load_model(model_path)
    
    # Load a test image slice
    #test_img = nib.load("/groups/ag-reuter/projects/corpus_callosum_fornix/label_QC/added_images/48e2d11f/orig_up.mgz")
    
    val_images, val_ac, val_pc, val_label_widths, val_third_ventricle_centers, val_vox_sizes, val_subj_ids = load_validation_data("/groups/ag-reuter/projects/corpus_callosum_fornix/pollakc/network/data/test_joined_labels.csv")
    
    dist_out = []
    dist_out_dict = {}
    uncertainty_out_dict = {}
    for img_path, AC_center, PC_center, label_width, third_ventricle_center, vox_size, subj_id in zip(val_images, val_ac, val_pc, val_label_widths, val_third_ventricle_centers, val_vox_sizes, val_subj_ids):

        # if subj_id != '1ca3a723-d981-4bbd-ae97-3f1f03ce5f0e':
        #     continue

        test_img = nib.load(img_path)
        test_slice = test_img.get_fdata()

        #label_width = 13

        
            
        # crop to middle 3+-1 (13) slices 
        test_slice = test_slice[256//2-label_width//2-1:256//2+label_width//2+2]

        # Run inference
        start_time = time.time()
        ac_coords, pc_coords, inputs, (crop_left, crop_top) = run_inference(model, test_slice, third_ventricle_center)
        center_pt = np.mean(np.concatenate([ac_coords, pc_coords], axis=0), axis=0)
        ac_coords, pc_coords, inputs, (crop_left, crop_top) = run_inference(model, test_slice, center_pt)


        inference_time = time.time() - start_time
        print(f"Inference took {inference_time:.3f} seconds")

        ac_dist = np.linalg.norm(AC_center[1:] - np.mean(ac_coords, axis=0)) / vox_size
        pc_dist = np.linalg.norm(PC_center[1:] - np.mean(pc_coords, axis=0)) / vox_size
        # ac_dist = np.linalg.norm(AC_center[1:] - ac_coords[ac_coords.shape[0]//2]) / vox_size
        # pc_dist = np.linalg.norm(PC_center[1:] - pc_coords[pc_coords.shape[0]//2]) / vox_size
        dist_out.append([ac_dist, pc_dist])
        dist_out_dict[subj_id] = [ac_dist, pc_dist]

        print(f"Distance AC: {ac_dist:.4f}, PC: {pc_dist:.4f}")


        # fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        # # Original image views
        # #ax.imshow(inputs[inputs.shape[0]//2, 1], cmap='gray')
        # ax.imshow(test_slice[test_slice.shape[0]//2, :, :], cmap='gray')
        # # Plot points on all views
        # pc_coords_plot = np.mean(pc_coords, axis=0)
        # ac_coords_plot = np.mean(ac_coords, axis=0)
        # ax.scatter(PC_center[2], PC_center[1], c='g', marker='o', label='Pred PC', s=2, alpha=0.5)
        # ax.scatter(AC_center[2], AC_center[1], c='y', marker='o', label='Pred AC', s=2, alpha=0.5)
        # ax.scatter(pc_coords_plot[1], pc_coords_plot[0], c='r', marker='x', label='PC', s=2, alpha=0.5)
        # ax.scatter(ac_coords_plot[1], ac_coords_plot[0], c='b', marker='x', label='AC', s=2, alpha=0.5)

        # for i in range(len(pc_coords)):
        #     ax.scatter(pc_coords[i][1], pc_coords[i][0], c='orange', marker='x', label='PC', s=2, alpha=0.5)
        #     ax.scatter(ac_coords[i][1], ac_coords[i][0], c='purple', marker='x', label='AC', s=2, alpha=0.5)

        # # make a box where the crop is
        # ax.add_patch(Rectangle((crop_top, crop_left), 64, 64, fill=False, color='r', linewidth=2))        
        # plt.savefig(f"/workspace/outputs/slice.png", bbox_inches='tight', dpi=500)
        # plt.close()

        # print(np.linalg.norm(PC_center[1:] - pc_coords, axis=1))
        # print(np.linalg.norm(AC_center[1:] - ac_coords, axis=1))

        # fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        # plt.plot(np.linalg.norm(PC_center[1:] - pc_coords, axis=1), color='r')
        # plt.plot(np.linalg.norm(AC_center[1:] - ac_coords, axis=1), color='b')
        # plt.hlines([np.linalg.norm(PC_center[1:] - pc_coords[pc_coords.shape[0]//2])], 0, len(np.linalg.norm(PC_center[1:] - pc_coords, axis=1)), color='r', linestyle='--')
        # plt.hlines([np.linalg.norm(AC_center[1:] - ac_coords[ac_coords.shape[0]//2])], 0, len(np.linalg.norm(AC_center[1:] - ac_coords, axis=1)), color='b', linestyle='--')
        # plt.savefig(f"/workspace/outputs/slice_pred_dist.png", bbox_inches='tight')
        # plt.close()


        # print('Uncertainty PC: ', np.linalg.norm(pc_coords - pc_coords[pc_coords.shape[0]//2]))
        # print('Uncertainty AC: ', np.linalg.norm(ac_coords - ac_coords[ac_coords.shape[0]//2]))
        # uncertainty_out_dict[subj_id] = [np.linalg.norm(pc_coords - pc_coords[pc_coords.shape[0]//2]), np.linalg.norm(ac_coords - ac_coords[ac_coords.shape[0]//2])]


        
        #import pdb; pdb.set_trace()


        # if len(dist_out_dict) == 3:
        #     break



    import pandas as pd
    dist_out_df = pd.DataFrame.from_dict(dist_out_dict, orient='index', columns=['ac_dist', 'pc_dist'])
    dist_out_df.to_csv("/workspace/outputs/dist_out_dict.csv")

    uncertainty_out_df = pd.DataFrame.from_dict(uncertainty_out_dict, orient='index', columns=['pc_uncertainty', 'ac_uncertainty'])
    uncertainty_out_df.to_csv("/workspace/outputs/uncertainty_localization_out_dict.csv")
        
    
    # Convert numpy array to NIfTI image before saving
    #nifti_img_in = nib.Nifti1Image(inputs, affine=test_img.affine, header=test_img.header)
    #nifti_orig_slice = nib.Nifti1Image(test_slice[4:-4], affine=test_img.affine, header=test_img.header)
    #nib.save(nifti_img_in, "/workspace/outputs/segmentation_input.nii.gz")
    #nib.save(nifti_orig_slice, "/workspace/outputs/segmentation_orig.nii.gz")



    print(f'Overall error - AC: {np.mean(dist_out, axis=0)[0]:.4f} mm, PC: {np.mean(dist_out, axis=0)[1]:.4f} mm')


    # validation set, middle 2x AC: 0.7648 mm, PC: 0.8181 mm
    # validation set, mean 2x   AC: 0.7638 mm, PC: 0.8404 mm  --- chose mean
    
    # test set (mean 2x)      AC: 0.9004 mm, PC: 0.9482 mm

    # diificult set (mean 2x):       AC: 0.9179 mm, PC: 1.3477 mm


# Example usage:
if __name__ == "__main__":
    run_validation()