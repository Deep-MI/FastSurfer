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
from functools import partial
from pathlib import Path
from typing import TypeVar, cast

import nibabel as nib
import numpy as np
import scipy.ndimage
from skimage.filters import gaussian
from skimage.measure import label

from FastSurferCNN.utils import AffineMatrix4x4, ShapeType, logging, nibabelHeader, nibabelImage
from FastSurferCNN.utils.brainvolstats import mask_in_array
from FastSurferCNN.utils.logging import setup_logging
from FastSurferCNN.utils.parallel import thread_executor

_T = TypeVar("_T", bound=np.number)
_TDType = np.dtype[_T]


LOGGER = logging.getLogger(__name__)

HELPTEXT = """
Script to reduce aparc+aseg to aseg by mapping cortex labels back to left/right GM.

If --outmask is used, it also creates a brainmask by dilating (5) and eroding (4) 
the segmentation, and then selecting the largest component. In that case also the 
segmentation is masked (to remove small components outside the main brain region).

If --flipwm is passed, disconnected WM islands will be checked and potentially
swapped to the other hemisphere. Sometimes these islands carry the wrong label 
and are far from the main body into the other hemisphere. This will cause mri_cc
to become really slow as it needs to cover a large search box. 


USAGE:
reduce_to_aseg  -i <input_seg> -o <output_seg>

    
Dependencies:
    Python 3.8+

    Numpy
    http://www.numpy.org

    Nibabel to read and write FreeSurfer data
    http://nipy.org/nibabel/
    
    skimage for erosion, dilation, connected component
    https://scikit-image.org/
"""

h_input = "path to input segmentation"
h_output = "path to output segmentation"
h_outmask = "path to output mask"
h_fixwm = "whether to try to flip labels of disconnected WM island to other hemi"


def options_parse():
    """
    Command line option parser.

    Returns
    -------
    options
        Object holding options.
    """
    parser = optparse.OptionParser(
        version="$Id: reduce_to_aseg.py,v 1.0 2018/06/24 11:34:08 mreuter Exp $",
        usage=HELPTEXT,
    )
    parser.add_option("--input", "-i", dest="input_seg", help=h_input)
    parser.add_option("--output", "-o", dest="output_seg", help=h_output)
    parser.add_option("--outmask", dest="output_mask", help=h_outmask)
    parser.add_option(
        "--fixwm", dest="fix_wm", help=h_fixwm, default=False, action="store_true"
    )
    (options, args) = parser.parse_args()

    if options.input_seg is None or options.output_seg is None:
        sys.exit("ERROR: Please specify input and output segmentations")

    return options


def reduce_to_aseg(data_inseg: np.ndarray[ShapeType, _TDType]) -> np.ndarray[ShapeType, _TDType]:
    """
    Reduce the input segmentation to a simpler segmentation (for all data orientations, LIA/etc).

    Parameters
    ----------
    data_inseg : np.ndarray, torch.Tensor
        The input segmentation. This should be a 3D array where the value at each position represents the segmentation
        label for that position.

    Returns
    -------
    data_inseg : np.ndarray, torch.Tensor
        The reduced segmentation.
    """
    LOGGER.info("Reducing to aseg ...")
    cortical_fill = np.full_like(data_inseg, 3)
    cortical_fill[data_inseg >= 2000] = 42
    return np.where(data_inseg >= 1000, cortical_fill, data_inseg)


def create_mask(aseg_data: np.ndarray[ShapeType, _TDType], dnum: int, enum: int) \
        -> np.ndarray[ShapeType, np.dtype[np.bool_]]:
    """
    Create dilated mask (works for all data orientations, LIA/etc).

    Parameters
    ----------
    aseg_data : int np.ndarray
        The input segmentation data.
    dnum : int
        The number of iterations for the dilation operation.
    enum : int
        The number of iterations for the erosion operation.

    Returns
    -------
    bool np.ndarray same shape as aseg_data
        Returns mask.
    """
    LOGGER.info("Creating dilated mask ...")

    # treat lateral orbital frontal and parsorbitalis special to avoid capturing too much of eye nerve
    lat_orb_front_mask = [2012, 1012]
    parsorbitalis_mask = [2019, 1019]
    frontal_mask = mask_in_array(aseg_data, lat_orb_front_mask + parsorbitalis_mask)
    LOGGER.info(f"Frontal region special treatment: {np.sum(frontal_mask)}")

    # reduce to binary
    datab = aseg_data > 0
    datab[frontal_mask] = 0
    # dilate and erode
    datab = scipy.ndimage.binary_dilation(datab, np.ones((3, 3, 3)), iterations=dnum)
    datab = scipy.ndimage.binary_erosion(datab, np.ones((3, 3, 3)), iterations=enum)
    # for x in range(dnum):
    #    datab = binary_dilation(datab, np.ones((3, 3, 3)))
    # for x in range(enum):
    #    datab = binary_erosion(datab, np.ones((3, 3, 3)))

    # extract largest component
    labels = label(datab)
    assert labels.max() != 0  # assume at least 1 real connected component
    LOGGER.info(f"  Found {labels.max()} connected component(s)!")

    if labels.max() > 1:
        LOGGER.info("  Selecting largest component!")
        datab = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1

    # add frontal regions back to mask
    datab[frontal_mask] = 1

    # set mask
    return datab.astype(np.uint8)


def flip_wm_islands(aseg_data: np.ndarray[ShapeType, _TDType]) -> np.ndarray[ShapeType, _TDType]:
    """
    Flip labels of disconnected white matter islands to the other hemisphere (works for all data orientations, LIA/etc).

    Parameters
    ----------
    aseg_data : numpy.ndarray
        The input segmentation data.

    Returns
    -------
    flip_data : numpy.ndarray
        The segmentation data with flipped WM labels.
    """
    # Sometimes WM is far in the other hemisphere, but with a WM label from the other hemi
    # These are usually islands, not connected to the main hemi WM component
    # Here we decide to flip assignment based on proximity to other WM and GM labels

    # label ids
    lh_wm = 2
    lh_gm = 3
    rh_wm = 41
    rh_gm = 42

    def _islands(data: np.ndarray[ShapeType, _TDType], _label: int) -> np.ndarray[ShapeType, np.dtype[np.bool_]]:
        # for lh get largest component and islands
        mask = data == _label
        labels = label(mask, background=0)
        assert labels.max() != 0  # assume at least 1 connected component
        bc = np.bincount(labels.flat)[1:]
        largest_id = np.argmax(bc) + 1
        largest_cc = labels == largest_id
        return (~largest_cc) & (labels > 0)

    lh_islands, rh_islands = thread_executor().map(partial(_islands, aseg_data), [lh_wm, rh_wm])


    # get signed probability for lh and rh (by smoothing joined GM+WM labels)
    lhmask = np.logical_or(aseg_data == lh_wm, aseg_data == lh_gm)
    rhmask = np.logical_or(aseg_data == rh_wm, aseg_data == rh_gm)
    ii = gaussian(lhmask.astype(float) * (-1) + rhmask.astype(float), sigma=1.5)

    # flip island
    rhswap = rh_islands & (ii < 0.0)
    lhswap = lh_islands & (ii > 0.0)
    flip_data = aseg_data.copy()
    flip_data[rhswap] = lh_wm
    flip_data[lhswap] = rh_wm
    LOGGER.info(f"FlipWM: rh {rhswap.sum()} and lh {lhswap.sum()} flipped.")

    return flip_data


def create_mask_and_save(
        seg: np.ndarray[ShapeType, np.dtype],
        seg_affine: AffineMatrix4x4,
        seg_header: nibabelHeader,
        filename: Path | None = None,
) -> np.ndarray[ShapeType, np.dtype[np.uint8]]:
    """Convenience function for brainmask generation plus saving."""
    mask_data = create_mask(seg, 5, 4)
    if filename is not None:
        LOGGER.info(f"Outputting mask: {filename}")
        mask = nib.MGHImage(mask_data, seg_affine, seg_header)
        mask.to_filename(filename)
    return mask_data


def reduce_to_aseg_and_save(
        seg: np.ndarray[ShapeType, np.dtype],
        seg_affine: AffineMatrix4x4,
        seg_header: nibabelHeader,
        filename: Path | None = None,
) -> np.ndarray[ShapeType, np.dtype[np.uint8]]:
    """Convenience function for reduce_to_aseg plus saving."""
    _data = reduce_to_aseg(seg)

    if filename is not None:
        LOGGER.info(f"Outputting aseg: {filename}")
        mask = nib.MGHImage(_data, seg_affine, seg_header)
        mask.to_filename(filename)
    return _data


if __name__ == "__main__":
    # Command Line options are error checking done here
    options = options_parse()

    setup_logging()

    LOGGER.info(f"Reading in aparc+aseg: {options.input_seg} ...")
    inseg = cast(nibabelImage, nib.load(options.input_seg))
    inseg_data = np.asanyarray(inseg.dataobj)
    inseg_header = inseg.header
    inseg_affine = inseg.affine

    # Change datatype to np.uint8
    inseg_header.set_data_dtype(np.uint8)

    # get mask
    if options.output_mask:
        io_fut_mask = thread_executor().submit(
            create_mask_and_save,
            inseg_data,
            inseg_affine,
            inseg_header,
            options.output_mask,
        )
    else:
        io_fut_mask = None

    # reduce aparc to aseg and mask regions
    aseg = reduce_to_aseg(inseg_data)

    if options.output_mask:
        # mask aseg also
        # wait and report errors in saving the mask
        bm = io_fut_mask.result()
        aseg[bm == 0] = 0

    if options.fix_wm:
        aseg = flip_wm_islands(aseg)

    LOGGER.info(f"Outputting aseg: {options.output_seg}")
    aseg_fin = nib.MGHImage(aseg, inseg_affine, inseg_header)
    aseg_fin.to_filename(options.output_seg)

    sys.exit(0)
