#!/usr/bin/env python3


# Copyright 2024 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
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
import optparse
import sys
import numpy as np
from numpy import typing as npt
import nibabel.freesurfer.io as fs
import nibabel as nib
from scipy import sparse
from lapy import TriaMesh
from smooth_aparc import smooth_aparc


HELPTEXT = """
Script to sample labels from image to surface and clean up. 

USAGE:
sample_parc  --inseg <segimg> --insurf <surf> --incort <cortex.label>
              --seglut <seglut> --surflut <surflut> --outaparc <out_aparc>
              --projmm <float> --radius <float>


Dependencies:
    Python 3.8

    Numpy
    http://www.numpy.org

    Nibabel to read and write FreeSurfer surface meshes
    http://nipy.org/nibabel/


Original Author: Martin Reuter
Date: Dec-18-2023

"""

h_inseg = "path to input segmentation image"
h_incort = "path to input cortex label mask"
h_insurf = "path to input surface"
h_outaparc = "path to output aparc"
h_surflut = "FreeSurfer look-up-table for values on surface"
h_seglut = "Look-up-table for values in segmentation image (rows need to correspond to surflut)"
h_projmm = "Sample along normal at projmm distance (in mm), default 0"
h_radius = "Search around sample location at radius (in mm) for label if 'unknown', default None"


def options_parse():
    """Command line option parser.

    Returns
    -------
    options
        object holding options

    """
    parser = optparse.OptionParser(
        version="$Id: smooth_aparc,v 1.0 2018/06/24 11:34:08 mreuter Exp $",
        usage=HELPTEXT,
    )
    parser.add_option("--inseg", dest="inseg", help=h_inseg)
    parser.add_option("--insurf", dest="insurf", help=h_insurf)
    parser.add_option("--incort", dest="incort", help=h_incort)
    parser.add_option("--surflut", dest="surflut", help=h_surflut)
    parser.add_option("--seglut", dest="seglut", help=h_seglut)
    parser.add_option("--outaparc", dest="outaparc", help=h_outaparc)
    parser.add_option("--projmm", dest="projmm", help=h_projmm, default=0.0, type="float")
    parser.add_option("--radius", dest="radius", help=h_radius, default=None, type="float")
    (options, args) = parser.parse_args()

    if options.insurf is None or options.inseg is None or options.outaparc is None:
        sys.exit("ERROR: Please specify input surface, input image and output aparc!")

    if options.surflut is None or options.seglut is None:
        sys.exit("ERROR: Please specify surface and segmentatin image LUT!")

    # maybe later add functionality, to not have a cortex label, e.g. 
    # like FreeSurfer find largest connected component and fill only
    # the other unknown regions
    if options.incort is None:
        sys.exit("ERROR: Please specify surface cortex label!")

    return options



def sample_nearest_nonzero(img, vox_coords, radius=3.0):
    """Sample closest non-zero value in a ball of radius around vox_coords.

    Parameters
    ----------
    img : nibabel.image
        Image to sample. Voxels need to be isotropic.
    vox_coords : ndarray float shape(n,3)
        Coordinates in voxel space around which to search.
    radius : float default 3.0
        Consider all voxels inside this radius to find a non-zero value.

    Returns
    -------
    samples : np.ndarray (n,)
        Sampled values. Retruns zero for vertices where values are zero in ball. 
    """
    # check for isotropic voxels 
    voxsize = img.header.get_zooms()
    print("Check isotropic vox sizes: {}".format(voxsize))
    assert (np.max(np.abs(voxsize - voxsize[0])) < 0.001), 'Voxels not isotropic!'
    data = np.asarray(img.dataobj)
    
    # radius in voxels:
    rvox = radius * voxsize[0]
    
    # sample window around nearest voxel
    x_nn = np.rint(vox_coords).astype(int)
    # Reason: to always have the same number of voxels that we check
    # and to be consistent with FreeSurfer, we center the window at
    # the nearest neighbor voxel, instead of at the float vox coordinates

    # create box with 2*rvox+1 side length to fully contain ball
    # and get coordiante offsets with zero at center
    ri = np.floor(rvox).astype(int)
    l = np.arange(-ri,ri+1)
    xv, yv, zv = np.meshgrid(l, l, l)
    dd = np.sqrt(xv*xv + yv*yv + zv*zv).flatten()

    # flatten and keep only sphere with radius
    xv = xv.flatten()[dd<=rvox]
    yv = yv.flatten()[dd<=rvox]
    zv = zv.flatten()[dd<=rvox]
    dd = dd[dd<=rvox]

    # stack to get offset vectors
    offsets = np.column_stack((xv, yv, zv))

    # sort offsets according to distance
    # Note: we keep the first zero voxel so we can later
    # determine if all voxels are zero with the argmax trick
    sortidx = np.argsort(dd)
    dd = dd[sortidx]
    offsets = offsets[sortidx,:]

    # reshape and tile to add to list of coords
    n = x_nn.shape[0]
    toffsets = np.tile(offsets.transpose().reshape(1,3,offsets.shape[0]),(n,1,1))
    s_coords = x_nn[:, :, np.newaxis] + toffsets

    # get image data at the s_coords locations
    s_data = data[s_coords[:,0], s_coords[:,1], s_coords[:,2]]

    # get first non-zero if possible
    nzidx = (s_data!=0).argmax(axis=1)
    # the above return index zero if all elements are zero which is OK for us
    # as we can then sample there and get a value of zero
    samples = s_data[np.arange(s_data.shape[0]),nzidx]
    return samples


def sample_img(surf, img, cortex=None, projmm=0.0, radius=None):
    """Sample volume at a distance from the surface.

    Parameters
    ----------
    surf : tuple | str
        Surface as returned by nibabel fs.read_geometry, where:
        surf[0] is the np.array of (n, 3) vertex coordinates and
        surf[1] is the np.array of (m, 3) triangle indices.
        If type is str, read surface from file.
    img : nibabel.image | str
        Image to sample.
        If type is str, read image from file.
    cortex : np.ndarray | str
        Filename of cortex label or np.array with cortex indices
    projmm : float
        Sample projmm mm along the surface vertex normals (default=0).
    radius : float [optional] | None
        If given and if the sample is equal to zero, then consider
        all voxels inside this radius to find a non-zero value.

    Returns
    -------
    samples : np.ndarray (n,)
        Sampled values.
    """
    if isinstance(surf, str):
        surf = fs.read_geometry(surf, read_metadata=True)
    if isinstance(img, str):
        img = nib.load(img)
    if isinstance(cortex, str):
        cortex = fs.read_label(cortex)
    nvert = surf[0].shape[0]
    # Compute Cortex Mask
    if cortex is not None:
        mask = np.zeros(nvert, dtype=bool)
        mask[cortex] = True
    else:
        mask = np.ones(nvert, dtype=bool)

    data = np.asarray(img.dataobj)
    # Use LaPy TriaMesh for vertex normal computation
    T = TriaMesh(surf[0], surf[1])
    # compute sample coordinates projmm mm along the surface normal
    # in surface RAS coordiante system:
    x = T.v + projmm * T.vertex_normals()
    # mask cortex
    xx = x[mask]

    # compute Transformation from surface RAS to voxel space:
    Torig = img.header.get_vox2ras_tkr()
    Tinv = np.linalg.inv(Torig)
    x_vox = np.dot(xx, Tinv[:3, :3].T) + Tinv[:3, 3]
    # sample at nearest voxel
    x_nn = np.rint(x_vox).astype(int)
    samples_nn = data[x_nn[:,0], x_nn[:,1], x_nn[:,2]]
    # no search for zeros, done:
    if not radius:
        samplesfull = np.zeros(nvert, dtype="int")
        samplesfull[mask] = samples_nn
        return samplesfull
    # search for zeros, but no zeros exist, done:
    zeros = np.asarray(samples_nn==0).nonzero()[0]
    if zeros.size == 0:
        samplesfull = np.zeros(nvert, dtype="int")
        samplesfull[mask] = samples_nn
        return samplesfull
    # here we need to do the hard work of searching in a windows
    # for non-zero samples
    print("sample_img: found {} zero samples, searching radius ...".format(zeros.size))
    z_nn = x_nn[zeros]
    z_samples = sample_nearest_nonzero(img, z_nn, radius=radius)
    samples_nn[zeros] = z_samples
    samplesfull = np.zeros(nvert, dtype="int")
    samplesfull[mask] = samples_nn
    return samplesfull


def replace_labels(img_labels, img_lut, surf_lut):
    """Replace image labels with corresponding surface labels or unknown.

    Parameters
    ----------
    img_labels : np.ndarray(n,)
        Array with imgage label ids.
    img_lut : str
        Filename for image label look up table.
    surf_lut : str
        Filename for surface label look up table.

    Returns
    -------
    surf_labels : np.ndarray (n,)
        Array with surface label ids.
    surf_ctab : np.ndarray shape(m,4)
        Surface color table (RGBA).
    surf_names : np.ndarray[str] shape(m,)
        Names of label regions.
    """
    surflut = np.loadtxt(surf_lut, usecols=(0,2,3,4,5), dtype="int")
    surf_ids = surflut[:,0]
    surf_ctab =  surflut[:,1:5]
    surf_names = np.loadtxt(surf_lut, usecols=(1), dtype="str")
    imglut = np.loadtxt(img_lut, usecols=(0,2,3,4,5), dtype="int")
    img_ids = imglut[:,0]
    img_names = np.loadtxt(img_lut, usecols=(1), dtype="str")
    assert (np.all(img_names == surf_names)), "Label names in the LUTs do not agree!"
    lut = np.zeros((img_labels.max()+1,), dtype = img_labels.dtype)
    lut[img_ids] = surf_ids
    surf_labels = lut[img_labels]
    return surf_labels, surf_ctab, surf_names


def sample_parc (surf, seg, imglut, surflut, outaparc, cortex=None, projmm=0.0, radius=None):
    """Replace image labels with corresponding surface labels or unknown.

    Parameters
    ----------
    surf : tuple | str
        Surface as returned by nibabel fs.read_geometry, where:
        surf[0] is the np.array of (n, 3) vertex coordinates and
        surf[1] is the np.array of (m, 3) triangle indices.
        If type is str, read surface from file.
    seg : nibabel.image | str
        Image to sample.
        If type is str, read image from file.
    imglut : str
        Filename for image label look up table.
    surflut : str
        Filename for surface label look up table.
    outaparc : str
        Filename for output surface parcellation.
    cortex : np.ndarray | str
        Filename of cortex label or np.ndarray with cortex indices
    projmm : float
        Sample projmm mm along the surface vertex normals (default=0).
    radius : float [optional] | None
        If given and if the sample is equal to zero, then consider
        all voxels inside this radius to find a non-zero value.
    """
    if isinstance(cortex, str):
        cortex = fs.read_label(cortex)
    if isinstance(surf, str):
        surf = fs.read_geometry(surf, read_metadata=True)
    if isinstance(seg, str):
        seg = nib.load(seg)
    # get rid of unknown labels first and translate the rest (avoids too much filling)
    segdata, surfctab, surfnames = replace_labels(np.asarray(seg.dataobj), imglut, surflut)
    # create img with new data (needed by sample img)
    seg2 = nib.MGHImage(segdata, seg.affine, seg.header)

    surfsamples = sample_img(surf, seg2, cortex, projmm, radius)
    smooths = smooth_aparc(surf, surfsamples, cortex)
    fs.write_annot(outaparc, smooths, ctab=surfctab, names=surfnames)


if __name__ == "__main__":
    # Command Line options are error checking done here
    options = options_parse()

    sample_parc(options.insurf, options.inseg, options.seglut, options.surflut, options.outaparc, options.incort, options.projmm, options.radius)

    sys.exit(0)

