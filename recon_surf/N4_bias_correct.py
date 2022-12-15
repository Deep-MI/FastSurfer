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
import optparse
import sys

import SimpleITK as sitk
import numpy as np

import image_io as iio

HELPTEXT = """

Script to call SITK N4 Bias Correction

USAGE:
N4_bias_correct  --invol <img.nii> --outvol <corrected.nii> OPTIONS


Dependencies:
    Python 3.8

    SimpleITK https://simpleitk.org/ (v2.1.1)


N4 Bias Correction:

The input image will be N4 bias corrected. Optimally you would pass a brainmask via
--mask. If no mask is given, all 0 values will be masked to speed-up correction and
avoid influence of flat zero regions on the bias field. 


WM Normalization and UCHAR:

Unless --skipwm is passed the output will be converted to UCHAR.
For this a ball of radius 50 voxels at the center of the image is used to find the
1st and 90th percentile of intensities in the brain. This region is then mapped to
3 .. 110, so that (WM) will be around 110.

There are different options how the center is found:
- If a mask (e.g. brainmask) is passed, the centroid of the mask will be used and 
  the ball will be masked to crop non-brain tissue.
- If no mask is passed, the OTSU method is used to find a brain mask and centroid. 
  Note, that the ball will not be cropped and may contain non-brain tissue.
- If a talairach.xfm transform is passed, we use the origin of the Talairach space
  as center for the ball. If additionally a mask was passed, the ball will be 
  masked to crop non-brain tissue. 



Original Author: Martin Reuter
Date: Mar-18-2022
"""

h_invol = 'path to input.nii.gz'
h_outvol = 'path to corrected.nii.gz'
h_mask = 'optional: path to mask.nii.gz'
h_shrink = '<int> shrink factor, default: 4'
h_levels = '<int> number of fitting levels, default: 4'
h_numiter = '<int> max number of iterations per level, default: 50'
h_thres = '<float> convergence threshold, default: 0.0'
h_skipwm = 'skip normalize WM to 110 (for UCHAR scale)'
h_tal = '<sting> file name of talairach.xfm if using this for finding origin'
h_threads = '<int> number of threads, default: 1'


def options_parse():
    """
    Command line option parser
    """
    parser = optparse.OptionParser(version='$Id: N4_bias_correct.py,v 1.0 2022/03/18 21:22:08 mreuter Exp $',
                                   usage=HELPTEXT)
    parser.add_option('--in', dest='invol', help=h_invol)
    parser.add_option('--out', dest='outvol', help=h_outvol)
    parser.add_option('--mask', dest='mask', help=h_mask, default=None)
    parser.add_option('--shrink', dest='shrink', help=h_shrink, default=4, type="int")
    parser.add_option('--levels', dest='levels', help=h_levels, default=4, type="int")
    parser.add_option('--numiter', dest='numiter', help=h_numiter, default=50, type="int")
    parser.add_option('--thres', dest='thres', help=h_thres, default=0.0, type="float")
    parser.add_option('--skipwm', dest='skipwm', help=h_skipwm, default=False, action="store_true")
    parser.add_option('--tal', dest='tal', help=h_tal, default=None)
    parser.add_option('--threads', dest='threads', help=h_threads, default=1, type="int")
    (options, args) = parser.parse_args()

    if options.invol is None or options.outvol is None:
        sys.exit(
            '\nERROR: Please specify --in input.nii --out output.nii as nifti\n   or use --help to see all options.\n')

    return options


def N4correctITK(itkimage, itkmask=None, shrink=4, levels=4, numiter=50, thres=0.0, rescale=True):
    """
    Perform the bias field correction.

    Args:
        itkimage: itkImage[sitk.sitkFloat32]
        itkmask: Optional[itkImage[sitk.sitk.Uint8]
        shrink: Union[numbers.Rational]
        levels: int
        numiter: int
        thres: float
        rescale: bool

    Returns: itkImage[sitk.sitkFloat32]
    """

    # if no mask is passed, create a simple mask from the image
    if not itkmask:
        itkmask = itkimage > 0
        itkmask.CopyInformation(itkimage)
        print("- mask: default (>0)")
    else:
        # binarize mask
        itkmask = itkmask > 0

    itkorig = itkimage

    # subsample image
    if shrink > 1:
        itkimage = sitk.Shrink(itkimage, [shrink] * itkimage.GetDimension())
        itkmask = sitk.Shrink(itkmask, [shrink] * itkimage.GetDimension())

    # init corrector
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([numiter] * levels)
    corrector.SetConvergenceThreshold(thres)

    # bias correct image
    sitk.ProcessObject.SetGlobalDefaultCoordinateTolerance(1e-04)
    corrector.Execute(itkimage, itkmask)

    # we need to apply bias field to original input
    log_bias_field = corrector.GetLogBiasFieldAsImage(itkorig)
    itkcorrected = itkorig / sitk.Exp(log_bias_field)

    # normalize to average input intensity
    if rescale:
        stats = sitk.StatisticsImageFilter()
        stats.Execute(itkcorrected)
        m1 = stats.GetMean()
        stats.Execute(itkorig)
        m0 = stats.GetMean()
        print("- rescale")
        print("   mean input: {:.4f} , mean corrected {:.4f} , image rescale: {:.4f}".format(m0, m1, m0 / m1))
        itkcorrected = itkcorrected * (m0 / m1)

    # should already be Float 32
    # itkcorrected = sitk.Cast(itkcorrected, sitk.sitkFloat32)
    if itkimage.GetPixelIDTypeAsString() != "32-bit float":
        print(f"The data type is: {itkimage.GetPixelIDTypeAsString()}")
    return itkcorrected


def normalizeWM(itkimage, itkmask=None, radius=50, centroid=None, targetWM=110):
    # print("\nnormalizeWM:")

    mask_passed = True
    if not itkmask:
        itkmask = sitk.OtsuThreshold(itkimage, 0, 1, 200)
        print("- mask: default (otsu)")
        # itkmask = itkimage>0
        # print("- mask: default (>0)")
        itkmask.CopyInformation(itkimage)
        mask_passed = False
    else:
        # binarize mask
        itkmask = itkmask > 0

    if not centroid:
        # extract centroid from mask
        label_stats = sitk.LabelShapeStatisticsImageFilter()
        label_stats.Execute(itkmask)
        # centroid_world = label_stats.GetCenterGravity(1)
        centroid_world = label_stats.GetCentroid(1)
        #print("centroid pyhsical: {}".format(centroid_world))
        #centroidf = itkmask.TransformPhysicalPointToContinuousIndex(centroid_world)
        #print("centroid voxel: {}".format(centroidf))
        centroid = itkmask.TransformPhysicalPointToIndex(centroid_world)

    print("- centroid: {}".format(centroid))
    print("- size: {}".format(itkimage.GetSize()))
    print("- spacing: "+ ' '.join(format(f, '.2f') for f in itkimage.GetSpacing()))

    # distance image
    isize = itkimage.GetSize()
    ispace = itkimage.GetSpacing()
    zz, yy, xx = np.meshgrid(range(isize[2]), range(isize[1]), range(isize[0]), indexing='ij')
    xx = ispace[0] * (xx - centroid[0])
    yy = ispace[1] * (yy - centroid[1])
    zz = ispace[2] * (zz - centroid[2])
    distance = xx * xx + yy * yy + zz * zz
    ball = distance < radius * radius

    # make sure to crop non-brain regions (if masked was passed)
    #  warning do not use otsu mask as it is cropping low-intensity values
    if mask_passed:
        ball = ball * sitk.GetArrayFromImage(itkmask)
    # for debugging the ball location and size:
    #balli = sitk.GetImageFromArray(1.0 * ball)
    #balli.CopyInformation(itkimage)
    #sitk.WriteImage(balli, "ball.nii.gz")

    # get 1 and 90 percentiles of intensities in ball
    mask = np.extract(ball, sitk.GetArrayFromImage(itkimage))
    percentiles = np.percentile(mask, [1, 90])
    percentile01, percentile90 = percentiles.tolist()
    print("-  1st percentile: {:.2f}\n- 90th percentile: {:.2f}".format(percentile01, percentile90))

    # compute intensity transformation
    m = (targetWM - 2.55) / (percentile90 - percentile01)
    b = targetWM - m * percentile90
    print("- m: {:.4f}  b: {:.4f}".format(m, b))

    # itkImage already is Float32 and output should be also Float32, we clamp outside as well
    normed = itkimage * m + b

    return normed


def readTalairachXFM(fname):
    with open(fname) as f:
        lines = f.readlines()
    f.close()
    counter = 0
    tal = []
    for line in lines:
        counter += 1
        firstword = line.split(sep="_", maxsplit=1)[0].lower()
        if firstword == "linear":
            taltxt = np.char.replace(lines[counter:counter + 3], ';', ' ')
            tal = np.genfromtxt(taltxt)
            tal = np.vstack([tal, [0, 0, 0, 1]])
            # tal = np.loadtxt(fname,skiprows=counter,delimiter=)

    print(" tal: {}".format(tal))
    return tal


def getTalOriginVox(tal, image):
    talinv = np.linalg.inv(tal)
    tal_origin = np.array(talinv[0:3, 3]).ravel()
    # print("Tal Physical Origin: {}".format(tal_origin))
    vox_origin = image.TransformPhysicalPointToIndex(tal_origin)
    print("Tal Voxel Origin: {}".format(vox_origin))
    return vox_origin


if __name__ == "__main__":

    # Command Line options are error checking done here
    options = options_parse()

    print()
    print("N4 Bias Correction Parameters:")
    print()
    print("- input volume: {}".format(options.invol))
    print("- output volume: {}".format(options.outvol))
    if options.mask:
        print("- mask: {}".format(options.mask))
    else:
        print("- mask: default (>0)")
    print("- shrink factor: {}".format(options.shrink))
    print("- number fitting levels: {}".format(options.levels))
    print("- number iterations: {}".format(options.numiter))
    print("- convergence threshold: {}".format(options.thres))
    print("- skipwm: {}".format(bool(options.skipwm)))
    if options.tal:
        print("- talairach: {}".format(options.tal))
    print("- threads: {}".format(options.threads))

    # set number of threads
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(options.threads)

    # read image (only nii supported) and convert to float32
    print("\nreading {}".format(options.invol))
    # itkimage = sitk.ReadImage(options.invol, sitk.sitkFloat32)
    itkimage, header = iio.readITKimage(options.invol, sitk.sitkFloat32, with_header=True)

    # if talaraich is passed
    talorig = None
    if options.tal:
        taltrans = readTalairachXFM(options.tal)
        talorig = getTalOriginVox(taltrans, itkimage)

    # read mask (as uchar)
    if options.mask:
        itkmask = iio.readITKimage(options.mask, sitk.sitkUInt8)
    else:
        itkmask = None

    # call N4 correct
    print("\nexecuting N4 correction ...")
    itkcorrected = N4correctITK(itkimage, itkmask, options.shrink, options.levels, options.numiter, options.thres,
                                rescale=options.skipwm)

    if not options.skipwm:
        print("\nnormalize WM to 110")
        itkcorrected = normalizeWM(itkcorrected, itkmask, centroid=talorig)

    print("\nconverting to UCHAR")
    itkcorrected = sitk.Cast(sitk.Clamp(itkcorrected, lowerBound=0, upperBound=255), sitk.sitkUInt8)

    # write image
    print("writing: {}".format(options.outvol))
    # sitk.WriteImage(itkcorrected, options.outvol)
    iio.writeITKimage(itkcorrected, options.outvol, header)

    sys.exit(0)
