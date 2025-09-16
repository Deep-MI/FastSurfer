import nibabel as nib
import numpy as np
import skimage.measure
import scipy.ndimage
import pandas as pd
from shape.resample_poly import iterative_resample_polygon

def get_endpoints(cc_mask, AC_2d, PC_2d, resolution, return_coordinates=True, contour_smoothing=1.0):
    """
    Determines endpoints of CC by finding the point in the contour closest to the anterior and posterior commisure (with some offsets)

    NOTE: Expects LIA orientation
    """
    image_size = cc_mask.shape

    # Calculate angle between AC-PC line and horizontal using numpy
    ac_pc_vector = PC_2d - AC_2d
    horizontal_vector = np.array([0, -20])
    # Calculate angle using dot product formula: cos(theta) = (a·b)/(|a||b|)
    dot_product = np.dot(ac_pc_vector, horizontal_vector)
    norms = np.linalg.norm(ac_pc_vector) * np.linalg.norm(horizontal_vector)
    theta = np.arccos(dot_product / norms)
    

    # Convert symbolic theta to float and convert from radians to degrees
    theta_degrees = theta * 180 / np.pi
    rotated_cc_mask = scipy.ndimage.rotate(cc_mask, -theta_degrees, order=0, reshape=False)
    

    # rotate points around center
    origin_point = np.array([image_size[0]//2, image_size[1]//2])
    
    # Create rotation matrix for -theta
    rot_matrix = np.array([[np.cos(-theta), -np.sin(-theta)],
                            [np.sin(-theta), np.cos(-theta)]])
    
    # Translate points to origin, rotate, then translate back
    pc_centered = PC_2d - origin_point
    ac_centered = AC_2d - origin_point
    
    rotated_PC_2d = (rot_matrix @ pc_centered) + origin_point
    rotated_AC_2d = (rot_matrix @ ac_centered) + origin_point

    # get contour of CC
    gaussian_cc_mask = scipy.ndimage.gaussian_filter(rotated_cc_mask.astype(float), sigma=contour_smoothing)
    #gaussian_cc_mask = scipy.ndimage.gaussian_filter(gaussian_cc_mask, sigma=1.0)
    contour = skimage.measure.find_contours(gaussian_cc_mask, level=0.5)[0].T

    contour = iterative_resample_polygon(contour.T, 701).T
    contour = contour[:,:-1]

    rotated_AC_2d = np.array(rotated_AC_2d).astype(float)
    rotated_PC_2d = np.array(rotated_PC_2d).astype(float)

    # move posterior commisure 5 mm posterior
    rotated_PC_2d = rotated_PC_2d + np.array([10 * resolution, -5 * resolution])

    # move anterior commisure 1.5 mm anterior
    rotated_AC_2d = rotated_AC_2d + np.array([0, 5 * resolution])

    # find point in contour closest to AC
    AC_startpoint_idx = np.argmin(np.linalg.norm(contour - rotated_AC_2d[:,None], axis=0))
    
    # find point in contour closest to PC
    PC_startpoint_idx = np.argmin(np.linalg.norm(contour - rotated_PC_2d[:,None], axis=0))

    # rotate startpoints to original orientation
    # Create rotation matrix
    rot_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]])

    # rotate contour to original orientation
    contour_rotated = np.zeros_like(contour)

    origin_point = np.array(origin_point).astype(float)
    # Create rotation matrix
    rot_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]])
    
    # Translate points to origin, rotate, then translate back
    contour_centered = contour - origin_point[:,None]
    contour_rotated = (rot_matrix @ contour_centered) + origin_point[:,None]


    if return_coordinates:
        AC_contour_point = contour[:,AC_startpoint_idx]
        PC_contour_point = contour[:,PC_startpoint_idx]

        # Translate points to origin, rotate, then translate back
        ac_centered = AC_contour_point - origin_point
        pc_centered = PC_contour_point - origin_point

        start_point_A = (rot_matrix @ ac_centered) + origin_point
        start_point_P = (rot_matrix @ pc_centered) + origin_point

        return contour_rotated, start_point_A, start_point_P
    else:
        return contour_rotated, AC_startpoint_idx, PC_startpoint_idx


def get_endpoints_from_nib(cc_label_nib, paths_csv, subj_id, return_coordinates=True):
    cc_mask = cc_label_nib.get_fdata() == 192
    cc_mask = cc_mask[cc_mask.shape[0]//2]


    posterior_commisure_center = paths_csv.loc[subj_id, 'PC_center_r':'PC_center_s'].to_numpy().astype(float)
    anterior_commisure_center = paths_csv.loc[subj_id, 'AC_center_r':'AC_center_s'].to_numpy().astype(float)

    # adjust LR from label coordinates to orig_up coordinates
    posterior_commisure_center[0] = 128
    anterior_commisure_center[0] = 128

    # orientation I, A
    # rotate image so anterior and posterior commisure are horizontal
    AC_2d = anterior_commisure_center[1:]
    PC_2d = posterior_commisure_center[1:]

    return get_endpoints(cc_mask, AC_2d, PC_2d, resolution=cc_label_nib.header.get_zooms()[1], return_coordinates=return_coordinates)


if __name__ == "__main__":
    from tqdm import tqdm
    OUTPUT_TO_RAS = True
    PLOT = False

    paths_csv = pd.read_csv('/groups/ag-reuter-2/users/pollakc/corpus_callosum_fornix/pollakc/network/data/found_labels_with_meta_data_difficult_final.csv', index_col=0)

    for subj_id in tqdm(paths_csv.index):
        try:
            cc_label_nib = nib.load(paths_csv.loc[subj_id, 'label_merged'])
        except Exception as e:
            import pdb; pdb.set_trace()
            print(subj_id, 'error', e)
            continue
        
        
        
        # if np.sum(cc_mask) < 20:
        #     print(subj_id, 'skipping')
        #     continue
        
        contour, start_point_A, start_point_P = get_endpoints_from_nib(cc_label_nib, paths_csv, subj_id)

        
        
        # if PLOT:
        #     # Add visualization
        #     import matplotlib.pyplot as plt
        #     fig, ax = plt.subplots(figsize=(10, 8))
        #     ax.imshow(cc_mask, cmap='gray')
        #     ax.plot(contour[1], contour[0], 'b-', label='Contour')
        #     # Plot initial endpoint estimates
        #     ax.plot(start_point_A[1], start_point_A[0], 'rx', 
        #             markersize=8)
        #     ax.plot(start_point_P[1], start_point_P[0], 'rx', 
        #             markersize=8, label='Ours')
        #     ax.legend()
        #     ax.set_title(f'Subject: {subj_id}')
        #     # Save plot if desired
        #     #plt.savefig(f'./endpoint_plots/{subj_id}.png', dpi=300, bbox_inches='tight')
        #     plt.show()
        #     plt.close()


        if OUTPUT_TO_RAS:
            # use vox2ras matrix to convert to mm
            vox2ras_matrix = cc_label_nib.affine
            
            # Add a third dimension (z) with 0 and a fourth dimension (homogeneous coordinate) with 1
            contour_homogeneous = np.vstack([contour, np.zeros(contour.shape[1]), np.ones(contour.shape[1])])
            start_point_A_homogeneous = np.hstack([start_point_A, [0, 1]])
            start_point_P_homogeneous = np.hstack([start_point_P, [0, 1]])
            
            # Apply the transformation
            contour = (vox2ras_matrix @ contour_homogeneous)[:3, :]
            start_point_A = (vox2ras_matrix @ start_point_A_homogeneous)[:3]
            start_point_P = (vox2ras_matrix @ start_point_P_homogeneous)[:3]


        np.save(f'./contour_data/endpoints_{subj_id}.npy', np.array([start_point_A, start_point_P]), allow_pickle=False)
        np.save(f'./contour_data/contours_{subj_id}.npy', np.array(contour), allow_pickle=False)
        