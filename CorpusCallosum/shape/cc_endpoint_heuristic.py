import lapy
import numpy as np
import scipy.ndimage
import skimage.measure


def get_endpoints(cc_mask, AC_2d, PC_2d, resolution, return_coordinates=True, contour_smoothing=1.0):
    """
    Determines endpoints of CC by finding the point in the contour closest to
    the anterior and posterior commisure (with some offsets)

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
    origin_point = np.array([image_size[0] // 2, image_size[1] // 2])

    # Create rotation matrix for -theta
    rot_matrix = np.array([[np.cos(-theta), -np.sin(-theta)], [np.sin(-theta), np.cos(-theta)]])

    # Translate points to origin, rotate, then translate back
    pc_centered = PC_2d - origin_point
    ac_centered = AC_2d - origin_point

    rotated_PC_2d = (rot_matrix @ pc_centered) + origin_point
    rotated_AC_2d = (rot_matrix @ ac_centered) + origin_point

    # get contour of CC
    gaussian_cc_mask = scipy.ndimage.gaussian_filter(rotated_cc_mask.astype(float), sigma=contour_smoothing)
    # gaussian_cc_mask = scipy.ndimage.gaussian_filter(gaussian_cc_mask, sigma=1.0)
    contour = skimage.measure.find_contours(gaussian_cc_mask, level=0.5)[0].T



    # Add z=0 coordinate to make 3D, then remove it after resampling
    contour_3d = np.vstack([contour, np.zeros(contour.shape[1])])
    contour_3d = lapy.tria_mesh.TriaMesh._TriaMesh__resample_polygon(contour_3d.T, 701).T
    contour = contour_3d[:2]

    
    contour = contour[:, :-1]

    rotated_AC_2d = np.array(rotated_AC_2d).astype(float)
    rotated_PC_2d = np.array(rotated_PC_2d).astype(float)

    # move posterior commisure 5 mm posterior
    rotated_PC_2d = rotated_PC_2d + np.array([10 * resolution, -5 * resolution])

    # move anterior commisure 1.5 mm anterior
    rotated_AC_2d = rotated_AC_2d + np.array([0, 5 * resolution])

    # find point in contour closest to AC
    AC_startpoint_idx = np.argmin(np.linalg.norm(contour - rotated_AC_2d[:, None], axis=0))

    # find point in contour closest to PC
    PC_startpoint_idx = np.argmin(np.linalg.norm(contour - rotated_PC_2d[:, None], axis=0))

    # rotate startpoints to original orientation
    # Create rotation matrix
    rot_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    # rotate contour to original orientation
    contour_rotated = np.zeros_like(contour)

    origin_point = np.array(origin_point).astype(float)
    # Create rotation matrix
    rot_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    # Translate points to origin, rotate, then translate back
    contour_centered = contour - origin_point[:, None]
    contour_rotated = (rot_matrix @ contour_centered) + origin_point[:, None]

    if return_coordinates:
        AC_contour_point = contour[:, AC_startpoint_idx]
        PC_contour_point = contour[:, PC_startpoint_idx]

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
    cc_mask = cc_mask[cc_mask.shape[0] // 2]

    posterior_commisure_center = paths_csv.loc[subj_id, "PC_center_r":"PC_center_s"].to_numpy().astype(float)
    anterior_commisure_center = paths_csv.loc[subj_id, "AC_center_r":"AC_center_s"].to_numpy().astype(float)

    # adjust LR from label coordinates to orig_up coordinates
    posterior_commisure_center[0] = 128
    anterior_commisure_center[0] = 128

    # orientation I, A
    # rotate image so anterior and posterior commisure are horizontal
    AC_2d = anterior_commisure_center[1:]
    PC_2d = posterior_commisure_center[1:]

    return get_endpoints(
        cc_mask, AC_2d, PC_2d, resolution=cc_label_nib.header.get_zooms()[1], return_coordinates=return_coordinates
    )

