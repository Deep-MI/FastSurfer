import numpy as np

def calculate_cc_index(cc_contour):
    """
    Calculate CC index based on three perpendicular measurements.
    
    Args:
        cc_contour: 2xN array of contour points in ACPC space
        
    Returns:
        float: Sum of thicknesses at three measurement points
    """
    # Get anterior and posterior points
    anterior_idx = np.argmin(cc_contour[0])  # Leftmost point
    posterior_idx = np.argmax(cc_contour[0])  # Rightmost point
    
    # Get the longest line (anterior to posterior)
    ap_line = cc_contour[:,posterior_idx] - cc_contour[:,anterior_idx]
    ap_length = np.linalg.norm(ap_line)
    ap_unit = np.array([-ap_line[1], ap_line[0]]) / ap_length
    
    # Get midpoint of AP line
    midpoint = cc_contour[:,anterior_idx] + (ap_line/2)
    
    # Get perpendicular direction
    
    
    # Get intersection points with contour for each measurement line
    def get_intersections(start_point, direction):
        # Get all points above and below the line
        points = cc_contour.T - start_point[None,:]
        dots = np.dot(points, direction)
        signs = np.sign(dots)
        sign_changes = np.where(np.diff(signs))[0]
        
        intersections = []
        for idx in sign_changes:
            # Linear interpolation between points
            t = -dots[idx] / (dots[idx+1] - dots[idx])
            intersection = cc_contour[:,idx] + t * (cc_contour[:,idx+1] - cc_contour[:,idx])
            intersections.append(intersection)
            
        return np.array(intersections)
    
    # Get three measurements
    most_anterior_pt = cc_contour[:,anterior_idx]
    perpendicular_unit = np.array([-ap_unit[1], ap_unit[0]])
    

    anterior_intersections = get_intersections(most_anterior_pt - 10*perpendicular_unit, ap_unit)

    # sort by x
    anterior_intersections = anterior_intersections[np.argsort(anterior_intersections[:,0])]

    middle_ints = get_intersections(midpoint, perpendicular_unit) 

    if len(middle_ints) != 2:
        print(f"WARNING: The perpendicular line should intersect the contour twice, but it intersects {len(middle_ints)} times")

    # plt.close()
    
    

    # calculate index
    ap_distance = np.linalg.norm(anterior_intersections[0] - anterior_intersections[-1])
    anterior_distance = np.linalg.norm(anterior_intersections[0] - anterior_intersections[1])
    posterior_distance = np.linalg.norm(anterior_intersections[-1] - anterior_intersections[-2])
    top_distance = np.linalg.norm(middle_ints[0] - middle_ints[1])

    index = (anterior_distance + posterior_distance + top_distance) / ap_distance




    # fig, ax = plt.subplots(figsize=(8, 6))
    
    # # Plot the CC contour
    # ax.plot(cc_contour[0], cc_contour[1], 'k-', linewidth=1)
    # # add line from last to first
    # ax.plot([cc_contour[0,-1], cc_contour[0,0]], [cc_contour[1,-1], cc_contour[1,0]], 
    #         'k-', linewidth=1)
    
    # # Plot AP line
    # ax.plot([cc_contour[0,anterior_idx], cc_contour[0,posterior_idx]], 
    #         [cc_contour[1,anterior_idx], cc_contour[1,posterior_idx]], 
    #         'r--', linewidth=1)#, label='Anterior-posterior line')
    
    
    # # Plot the three measurement lines
    # for i, ints in enumerate(zip(anterior_intersections[:-1], anterior_intersections[1:])):

    #     if i != 1:
    #         ax.plot([ints[0][0], ints[1][0]], [ints[0][1], ints[1][1]], 
    #                 'b-', linewidth=1, label='Measurement line horizontal' if i==0 else None)
        
    # ax.plot([middle_ints[0,0], middle_ints[1,0]], [middle_ints[0,1], middle_ints[1,1]], 
    #         'g-', linewidth=1, label='Measurement lines vertical')


    # print(middle_ints[0,], middle_ints[1,1])
    # print(midpoint[1], midpoint[0])
    # ax.plot([middle_ints[0,0], midpoint[0]], [middle_ints[0,1], midpoint[1]], 
    #         'r--', linewidth=1)#, label='Superior-inferior line')

    # #plt.scatter(midpoint[0], midpoint[1], color='green', s=20)
        
    # ax.set_aspect('equal')
    # ax.legend()
    # # add gray background to CC contour
    # # Fill the inside of the contour with a gray shade
    # from matplotlib.path import Path
    # from matplotlib.patches import PathPatch
    
    # # Create a path from the contour points
    # contour_path = Path(np.array([cc_contour[0], cc_contour[1]]).T)
    
    # # Create a patch from the path and add it to the axes
    # patch = PathPatch(contour_path, facecolor='gray', alpha=0.2, edgecolor=None)
    # ax.add_patch(patch)

    # # invert x
    # ax.invert_xaxis()
    # #ax.set_title('CC Index Measurement Lines')
    # plt.axis('off')
    # plt.show()

    return index

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from cc_thickness import convert_to_ras
    from shape.cc_endpoint_heuristic import get_endpoints
    import pandas as pd
    import nibabel as nib
    from tqdm import tqdm
    # Create visualization of CC index measurements


    paths_csv = pd.read_csv('/groups/ag-reuter/projects/corpus_callosum_fornix/pollakc/network/data/found_labels_with_meta_data_difficult_final.csv', index_col=0)


    for subj_num, subj_id in enumerate(tqdm(paths_csv.index)):
        #subj_id = '099f7f5a'
        
        label_path = paths_csv.loc[subj_id, 'label_merged']
        
        try:
            cc_label_nib = nib.load(label_path)
        except Exception as e:
            import pdb; pdb.set_trace()
            print(subj_id, 'error', e)
            continue

        PC_2d = paths_csv.loc[subj_id, 'PC_center_r':'PC_center_s'].to_numpy().astype(float)[1:]
        AC_2d = paths_csv.loc[subj_id, 'AC_center_r':'AC_center_s'].to_numpy().astype(float)[1:]
        

        cc_mask = cc_label_nib.get_fdata() == 192
        cc_mask = cc_mask[cc_mask.shape[0]//2]


        contour, anterior_endpoint_idx, posterior_endpoint_idx = get_endpoints(cc_mask, AC_2d, PC_2d, cc_label_nib.header.get_zooms()[1], return_coordinates=False)


        
        contour = convert_to_ras(contour, cc_label_nib.affine)

        contour_2d=contour#[[2,0]].T[1:]
        #contour = contour[[2,0,1]]

        index = calculate_cc_index(contour_2d)

        print(subj_id, index)





