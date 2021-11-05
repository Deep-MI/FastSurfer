
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
import nibabel as nib
import sys
import argparse

HELPTEXT = """
Script to add corpus callosum segmentation (CC, FreeSurfer IDs 251-255) to
deep-learning prediction (e.g. aparc.DKTatlas+aseg.deep.mgz).


USAGE:
paint_cc_into_pred  -in_cc <input_seg_with_cc> -in_pred <input_seg_without_cc> -out <output_seg>


Dependencies:
    Python 3.5

    Nibabel to read and write FreeSurfer data
    http://nipy.org/nibabel/

Original Author: Leonie Henschel
Date: Jul-10-2020

"""


def argument_parse():
    """
    Command line option parser for reduce_to_aseg.py
    """
    parser = argparse.ArgumentParser(usage=HELPTEXT)
    parser.add_argument('--input_cc', '-in_cc', dest='input_cc',
                        help="path to input segmentation with Corpus Callosum (IDs 251-255 in FreeSurfer space)")
    parser.add_argument('--input_pred', '-in_pred', dest='input_pred',
                        help="path to input segmentation Corpus Callosum shoud be added to.")
    parser.add_argument('--output', '-out', dest='output', help="path to output (input segmentation + added CC)")

    args = parser.parse_args()

    if args.input_cc is None or args.input_pred is None or args.output is None:
        sys.exit('ERROR: Please specify input and output segmentations')

    return args


def paint_in_cc(pred, aseg_cc):
    """
    Function to paint corpus callosum segmentation into prediction. Note, that this function
    modifies the original array and does not create a copy.

    :param np.ndarray pred: deep-learning prediction
    :param np.ndarray aseg_cc: aseg segmentation with CC
    :return np.ndarray: prediction with added CC
    """
    cc_mask = (aseg_cc >= 251) & (aseg_cc <= 255)
    pred[cc_mask] = aseg_cc[cc_mask]
    return pred


if __name__ == "__main__":
    # Command Line options are error checking done here
    options = argument_parse()

    print("Reading inputs: {} {}...".format(options.input_cc, options.input_pred))
    aseg_image = np.asanyarray(nib.load(options.input_cc).dataobj)
    prediction = nib.load(options.input_pred)
    pred_with_cc = paint_in_cc(np.asanyarray(prediction.dataobj), aseg_image)

    print ("Writing segmentation with corpus callosum to: {}".format(options.output))
    pred_with_cc_fin = nib.MGHImage(pred_with_cc, prediction.affine, prediction.header)
    pred_with_cc_fin.to_filename(options.output)

    sys.exit(0)
