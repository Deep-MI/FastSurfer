#!/usr/bin/env python3


# Copyright 2019 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
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


# IMPORTS
import numpy as np
import sys
import SimpleITK as sitk
import nibabel as nib
from nibabel.freesurfer.mghformat import MGHHeader


def mgh_from_sitk(sitk_img, orig_mgh_header=None):
    if orig_mgh_header:
        h1 = MGHHeader.from_header(orig_mgh_header)
    else:
        h1 = MGHHeader()
    # get voxels sizes and set zooms (=delta in h1 header)
    spacing = sitk_img.GetSpacing()
    h1.set_zooms(np.asarray(spacing))
    # Get direction cosines from sitk image, reshape to 3x3 Matrix
    direction = np.asarray(sitk_img.GetDirection()).reshape(3, 3, order="F") * [-1, -1, 1]
    h1["Mdc"] = direction
    # compute affine
    origin = np.asarray(sitk_img.GetOrigin()).reshape(3, 1) * [[-1], [-1], [1]]
    affine = np.vstack([np.hstack([h1['Mdc'].T * h1["delta"], origin]), [0, 0, 0, 1]])
    # get dims and calculate and set new image center in world coords
    dims = np.array(sitk_img.GetSize())
    if dims.size == 3: 
        dims = np.hstack((dims, [1]))
    h1['dims'] = dims
    h1['Pxyz_c'] = affine.dot(np.hstack((dims[:3] / 2.0, [1])))[:3]
    # swap axes as data is stored differently between sITK and Nibabel
    data = np.swapaxes(sitk.GetArrayFromImage(sitk_img),0,2)
    # assemble MGHImage from header, image data and affine
    mgh_img = nib.MGHImage(data, affine, h1)
    return mgh_img
    
    
def sitk_from_mgh(img):
    # reorder data as structure differs between nibabel and sITK:
    data = np.swapaxes(np.asanyarray(img.dataobj),0,2)
    # sitk can only create image with system native endianness 
    if not data.dtype.isnative:
        data = data.byteswap().newbyteorder()
    # create image from array
    img_sitk = sitk.GetImageFromArray(data)
    # Get direction from MDC, need to change sign of dim 0 and 1
    direction = img.header["Mdc"] * [-1, -1, 1]
    img_sitk.SetDirection(direction.ravel(order="F"))
    # set voxel sizes
    img_sitk.SetSpacing(np.array(img.header.get_zooms()).tolist())    
    # Get origin from affine, needs to change sign of dim 0 and 1
    origin = img.affine[:3, 3:] * [[-1], [-1], [1]]
    img_sitk.SetOrigin(origin.ravel())
    return img_sitk
    

def readITKimage(filename, vox_type = None, with_header=False):
    # If image is nifti image
    header = None
    if filename[-7:] == ".nii.gz" or filename[-4:] == ".nii":
        print("read Nifti image via sITK ...")
        if vox_type:
            itkimage = sitk.ReadImage(filename, vox_type)
        else:
            itkimage = sitk.ReadImage(filename)
    # if image is mgz
    elif filename[-4:] == ".mgz":
        print("read MGZ (FreeSurfer) image via nibabel...")
        image = nib.load(filename)
        header = image.header
        itkimage = sitk_from_mgh(image)
        if vox_type:
            itkimage = sitk.Cast(itkimage,vox_type)
    else:
        sys.exit("read ERROR: {} image type not supported (only: .mgz, .nii, .nii.gz).\n".format(filename))
    if with_header:
        return itkimage, header
    else:
        return itkimage


def writeITKimage(img, filename, header=None):
    # If image is nifti image
    if filename[-7:] == ".nii.gz" or filename[-4:] == ".nii":
        print("write Nifti image via sITK...")
        sitk.WriteImage(img, filename)
    # if image is mgz
    elif filename[-4:] == ".mgz":
        print("write MGZ (FreeSurfer) image via nibabel...")
        mgh_image = mgh_from_sitk(img, header)
        nib.save(mgh_image, filename)
    else:
        sys.exit("write ERROR: {} image type not supported (only: .mgz, .nii, .nii.gz).\n".format(filename))



if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("\nError: pass input and output file names!\n")
        sys.exit(1)
    img, hdr = readITKimage(sys.argv[1], with_header=True)
    writeITKimage(img, sys.argv[2], hdr)
    sys.exit(0)

