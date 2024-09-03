#!/usr/bin/env python3


# Copyright 2022 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
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

import align_points as align
import image_io as iio
import lta as lta
import numpy as np
import SimpleITK as sitk
from numpy import typing as npt

HELPTEXT = """

Script to align two images based on the centroids of their segmentations

USAGE:

align_seg.py --srcseg <img> --trgseg <img> [--affine] --outlta <out.lta> 

align_seg.py --srcseg <img> --flipped      [--affine] --outlta <out.lta>


Dependencies:
    Python 3.8+
    numpy
    SimpleITK https://simpleitk.org/ (v2.1.1)


Description:
For each common segmentation ID in the two inputs, the centroid coordinate is 
computed. The point pairs are then aligned by finding the optimal translation
and rotation (rigid) or affine. The output is a FreeSurfer LTA registration file.

Original Author: Martin Reuter
Date: Aug-24-2022
"""


h_srcseg = "path to source segmentation (e.g. aparc+aseg.mgz)"
h_trgseg = "path to target segmentation "
h_affine = (
    "register affine, instead of rigid (default), cannot be combined with --flipped"
)
h_outlta = "path to output transform lta file"
h_flipped = "register to left-right flipped as target aparc+aseg (cortical needed)"
h_midslice = "Optional, only for flipped. Slice where the midplane should be." \
             " Defaults to middle of image (width-1)/2."


def options_parse():
    """
    Command line option parser.

    Returns
    -------
    options
        Object holding options.
    """
    parser = optparse.OptionParser(
        version="$Id:align_seg.py,v 1.0 2022/08/24 21:22:08 mreuter Exp $",
        usage=HELPTEXT,
    )
    parser.add_option("--srcseg", dest="srcseg", help=h_srcseg)
    parser.add_option("--trgseg", dest="trgseg", help=h_trgseg)
    parser.add_option(
        "--affine", dest="affine", help=h_affine, default=False, action="store_true"
    )
    parser.add_option(
        "--flipped", dest="flipped", help=h_flipped, default=False, action="store_true"
    )
    parser.add_option("--midslice", dest="midslice", help=h_midslice, default=None, type="float")
    parser.add_option("--outlta", dest="outlta", help=h_outlta)
    (options, args) = parser.parse_args()
    if (
        options.srcseg is None
        or (options.trgseg is None and not options.flipped)
        or options.outlta is None
    ):
        sys.exit(
            "\nERROR: Please specify srcseg and trgseg (or flipped)" \
            " as well as output lta file\n   Use --help to see all options.\n"
        )
    return options


def get_seg_centroids(
        seg_mov: sitk.Image,
        seg_dst: sitk.Image,
        label_ids: npt.NDArray[int] | None = None
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Extract the centroids of the segmentation labels for mov and dst in RAS coords.

    Parameters
    ----------
    seg_mov : sitk.Image
        Source segmentation image.
    seg_dst : sitk.Image
        Target segmentation image.
    label_ids : Optional[npt.NDArray[int]]
        List of label ids to extract (Default value = []).

    Returns
    -------
    centroids_mov
        List of centroids of source segmentation.
    centroids_dst
        List of centroids of target segmentation.
    """
    if label_ids is not None:
        # use all joint labels except -1 and 0:
        nda1 = sitk.GetArrayFromImage(seg_mov)
        nda2 = sitk.GetArrayFromImage(seg_dst)
        lids = np.intersect1d(nda1, nda2)
        lids = lids[(lids > 0)]
    else:
        lids = label_ids
    # extract centroids from segmentation labels
    label_stats_mov = sitk.LabelShapeStatisticsImageFilter()
    label_stats_mov.Execute(seg_mov)
    label_stats_dst = sitk.LabelShapeStatisticsImageFilter()
    label_stats_dst.Execute(seg_dst)
    counter = 0
    centroids_mov = np.empty([lids.size, 3])
    centroids_dst = np.empty([lids.size, 3])
    # print(lids)
    for label in lids:
        label = int(label)
        centroids_mov[counter] = label_stats_mov.GetCentroid(label)
        centroids_dst[counter] = label_stats_dst.GetCentroid(label)
        # print("centroid physical: {}".format(centroid_world))
        # centroidf = itkmask.TransformPhysicalPointToContinuousIndex(centroid_world)
        # print("centroid voxel: {}".format(centroidf))
        # centroid  = itkmask.TransformPhysicalPointToIndex(centroid_world)
        counter = counter + 1
    # FreeSurfer seems to have different RAS than sITK physical space
    # so we flip the axis accordingly (will ensure the return matrix
    # of a registration is RAS2RAS)
    centroids_mov = centroids_mov * np.array([-1, -1, 1])
    centroids_dst = centroids_dst * np.array([-1, -1, 1])
    return centroids_mov, centroids_dst


def align_seg_centroids(
        seg_mov: sitk.Image,
        seg_dst: sitk.Image,
        label_ids: npt.NDArray[int] | None = None,
        affine: bool = False
) -> npt.NDArray:
    """
    Align the segmentations based on label centroids (rigid is default).

    Parameters
    ----------
    seg_mov : sitk.Image
        Path to src segmentation (e.g. aparc+aseg.mgz).
    seg_dst : sitk.Image
        Path to trg segmentation.
    label_ids : Optional[npt.NDArray[int]]
        List of label ids to align. Defaults to None.
    affine : bool
        True if affine should be returned.
        False if rigid should be returned. Defaults to False.

    Returns
    -------
    T
        Aligned centroids RAS2RAS transform.
    """
    # get centroids of each label in image
    centroids_mov, centroids_dst = get_seg_centroids(seg_mov, seg_dst, label_ids)
    # register
    if affine:
        T = align.find_affine(centroids_mov, centroids_dst)
    else:
        T = align.find_rigid(centroids_mov, centroids_dst)
    # print(T)
    return T


def get_vox2ras(img:sitk.Image) -> npt.NDArray:
    """
    Extract voxel to RAS (affine) from sitk image.
    
    Parameters
    ----------
    seg : sitk.Image
        Sitk Image.

    Returns
    -------
    vox2ras
        VOX2RAS (affine) transformation matrix.
    """
    vox2ras = np.zeros((4,4))
    vox2ras[3,3] = 1.0
    # Rotation
    cosines = img.GetDirection()
    vox2ras[0,0:3] = cosines[0:3] * np.array([-1, -1, -1])
    vox2ras[1,0:3] = cosines[3:6] * np.array([-1, -1, -1])
    vox2ras[2,0:3] = cosines[6:9] 
    # Scaling (multiply rows, hope this is correct)
    # not tested for anisotropic images
    spacing = img.GetSpacing()
    vox2ras[0:3,0:3] = spacing * vox2ras[0:3,0:3]
    # Translation
    vox2ras[0:3,3] = img.GetOrigin() * np.array([-1, -1, 1])
    return vox2ras

def align_flipped(seg: sitk.Image, mid_slice: float | None = None) -> npt.NDArray:
    """
    Registrate Left - right (make upright).

    Register cortial lables

    Parameters
    ----------
    seg : sitk.Image
        Segmentation Image. Should be aparc+aseg (DKT or not).
    mid_slice : Optional[float]
        Where the mid slice will be in upright space. Defaults to (width-1)/2.


    Returns
    -------
    Tsqrt
        RAS2RAS transformation matrix for registration.
    """
    lhids = np.array(
        [
            1002,
            1003,
            1005,
            1006,
            1007,
            1008,
            1009,
            1010,
            1011,
            1012,
            1013,
            1014,
            1015,
            1016,
            1017,
            1018,
            1019,
            1020,
            1021,
            1022,
            1023,
            1024,
            1025,
            1026,
            1027,
            1028,
            1029,
            1030,
            1031,
            1034,
            1035,
        ]
    )
    rhids = np.array(
        [
            2002,
            2003,
            2005,
            2006,
            2007,
            2008,
            2009,
            2010,
            2011,
            2012,
            2013,
            2014,
            2015,
            2016,
            2017,
            2018,
            2019,
            2020,
            2021,
            2022,
            2023,
            2024,
            2025,
            2026,
            2027,
            2028,
            2029,
            2030,
            2031,
            2034,
            2035,
        ]
    )
    ls = lhids.size
    label_stats = sitk.LabelShapeStatisticsImageFilter()
    label_stats.Execute(seg)
    centroids = np.empty([2 * lhids.size, 3])
    centroidsras = np.empty([2 * lhids.size, 3])
    counter = 0
    for label in np.concatenate((lhids, rhids)):
        label = int(label)
        centroidsras[counter] = label_stats.GetCentroid(label)
        # convert to vox coord:
        centroids[counter] = seg.TransformPhysicalPointToContinuousIndex(centroidsras[counter])
        counter = counter + 1

    # compute vox2ras matrix from image information
    vox2ras = get_vox2ras(seg)
    print(f"vox2ras:\n {vox2ras}")
    ras2vox = np.linalg.inv(vox2ras)
    print(f"ras2vox:\n {ras2vox}")

    # find mid slice of image (usually 127.5 for 256 width)
    # instead we could also fix this to be 128 independent of width
    # like mri_cc does. 
    if not mid_slice:
        middle = 0.5*(seg.GetWidth()-1.0)
    else:
        middle = mid_slice
    print(f"Mid slice will be at: {middle}")

    # negate right-left by flipping across middle of image (as make_upright would do it)
    centroids_flipped = centroids.copy()
    centroids_flipped[:,0] = -1 * (centroids[:,0] - middle) + middle

    # now right is left and left is right (re-order)
    centroids_flipped = np.concatenate(
        (centroids_flipped[ls::, :], centroids_flipped[0:ls, :])
    )
    # register centroids to LR-flipped versions
    T = align.find_rigid(centroids, centroids_flipped)
    # get half-way transform
    from scipy.linalg import sqrtm

    Tsqrt = np.real(sqrtm(T))
    print(f"Matrix sqrt diff: {np.linalg.norm(T - (Tsqrt @ Tsqrt))}")

    # convert vox2vox to ras2ras:
    Tsqrt = vox2ras @ Tsqrt @ ras2vox 
    #print(Tsqrt)

    return Tsqrt


if __name__ == "__main__":
    # Command Line options are error checking done here
    options = options_parse()

    print()
    print("Align Segmentations Parameters:")
    print()
    print(f"- src seg {options.srcseg}")
    if options.trgseg is not None:
        print(f"- trg seg {options.trgseg}")
    if options.flipped:
        print("- registering with left-right flipped image")
    if options.affine:
        print("- affine registration")
    else:
        print("- rigid registration")
    print(f"- out lta {options.outlta}")

    print(f"\nreading src {options.srcseg}")
    srcseg, srcheader = iio.readITKimage(
        options.srcseg, sitk.sitkInt16, with_header=True
    )
    if options.trgseg is not None:
        print(f"reading trg {options.trgseg} ...")
        trgseg, trgheader = iio.readITKimage(
            options.trgseg, sitk.sitkInt16, with_header=True
        )
        # register segmentations:
        T = align_seg_centroids(srcseg, trgseg, affine=options.affine)
    else:
        # flipped:
        T = align_flipped(srcseg, options.midslice)
        trgheader = srcheader

    # write transform lta
    print(f"writing: {options.outlta}")
    lta.writeLTA(
        options.outlta, T, options.srcseg, srcheader, options.trgseg, trgheader
    )

    print("...done\n")

    sys.exit(0)
