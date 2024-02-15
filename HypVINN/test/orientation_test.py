import nibabel as nib
import numpy as np

def calculate_flip_orientation(iornt,base_ornt):
    # ornt[P, 1] is flip of axis N, where 1 means no flip and -1 means flip.
    new_iornt=iornt.copy()

    for axno, direction in np.asarray(base_ornt):
        idx=np.where(iornt[:,0] == axno)
        idirection=iornt[int(idx[0][0]),1]
        if direction == idirection:
            new_iornt[int(idx[0][0]), 1] = 1.0
        else:
            new_iornt[int(idx[0][0]), 1] = -1.0

    return new_iornt

def reorient_img(img,ref_img):
    '''
    orientation transform. ornt[N,1]` is flip of axis N of the array implied by `shape`, where 1 means no flip and -1 means flip.
    For example, if ``N==0 and ornt[0,1] == -1, and there’s an array arr of shape shape, the flip would correspond to the effect of
    np.flipud(arr). ornt[:,0] is the transpose that needs to be done to the implied array, as in arr.transpose(ornt[:,0])
    Parameters
    ----------
    img
    base_img

    Returns
    -------

    '''

    ref_ornt =nib.io_orientation(ref_img.affine)
    iornt=nib.io_orientation(img.affine)

    if not np.array_equal(iornt,ref_ornt):
        #flip orientation
        fornt = calculate_flip_orientation(iornt,ref_ornt)
        img = img.as_reoriented(fornt)
        #transpose axis
        tornt = np.ones_like(ref_ornt)
        tornt[:,0] = ref_ornt[:,0]
        img = img.as_reoriented(tornt)
    return img

#labels ='/groups/ag-reuter/projects/hypothalamus/vinn_version/hypvinn_testsuite/RS/output_hypvinn_multi/0e82dc50-c21b-42a2-8c6a-75b26841419e/mri/hyposeg_multi.nii.gz'
move = '/groups/ag-reuter/projects/hypothalamus/vinn_version/hypvinn_testsuite/RS/output_hypvinn/0e82dc50-c21b-42a2-8c6a-75b26841419e/temp/T1_BC.nii.gz'

ref = '/groups/ag-reuter/projects/hypothalamus/vinn_version/hypvinn_testsuite/RS/output/0e82dc50-c21b-42a2-8c6a-75b26841419e/mri/orig.mgz'

img_ref = nib.load(ref)
print('print ref orientation {}'.format(nib.aff2axcodes(img_ref.affine)))

img_move = nib.load(move)
print('print move orientation {}'.format(nib.aff2axcodes(img_move.affine)))

new_img = reorient_img(img_move,ref_img=img_ref)
print('print new orientation {}'.format(nib.aff2axcodes(new_img.affine)))

