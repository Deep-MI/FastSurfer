# Copyright 2023 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
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
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import nibabel as nib
import numpy as np
import pandas as pd
import scipy.ndimage.morphology as morphology
import torch
from nibabel.filebasedimages import FileBasedHeader as _Header
from nibabel.freesurfer.mghformat import MGHError, MGHHeader
from numpy import typing as npt
from scipy.ndimage import (
    binary_closing,
    binary_erosion,
    filters,
    generate_binary_structure,
    uniform_filter,
)
from skimage.measure import label, regionprops

from FastSurferCNN.data_loader.conform import check_affine_in_nifti, conform, is_conform
from FastSurferCNN.utils import AffineMatrix4x4, Shape1d, logging, nibabelImage

##
# Global Vars
##
SUPPORTED_OUTPUT_FILE_FORMATS = ("mgz", "nii", "nii.gz")
# the integer types an MGH file can store, narrowest first, so the first that fits is the choice
MGH_INT_DTYPES = (np.uint8, np.uint16, np.int16, np.int32)
LOGGER = logging.getLogger(__name__)

##
# Helper Functions
##


# Conform an MRI brain image to UCHAR, RAS orientation, and 1mm or minimal isotropic
# voxels
def load_and_conform_image(
        img_filename: Path | str,
        order: int = 1,
        logger: logging.Logger = LOGGER,
        **conform_kwargs,
) -> tuple[_Header, np.ndarray, np.ndarray]:
    """
    Load MRI image and conform it to UCHAR, RAS orientation and 1mm or minimum isotropic
    voxels size.

    Only, if it does not already have this format.

    Parameters
    ----------
    img_filename : Path, str
        Path and name of volume to read.
    order : int, default=1
        Interpolation order for image conformation (0=nearest, 1=linear(default), 2=quadratic, 3=cubic).
    logger : logging.Logger, default=<local logger>
        Logger to write output to (default = STDOUT).
    **conform_kwargs
        Additional parameters to conform and is_conform.

    Returns
    -------
    nibabel.Header header_info
        Header information of the conformed image.
    numpy.ndarray affine_info
        Affine information of the conformed image.
    numpy.ndarray orig_data
        Conformed image data.

    Raises
    ------
    RuntimeError
        If input has multiple input frames or inconsistent nifti headers.
    """
    img_file = Path(img_filename)
    orig = cast(nibabelImage, nib.load(img_file))
    # is_conform and conform accept numeric values and the string 'min' instead of the bool value
    if not is_conform(orig, **conform_kwargs):

        logger.info("Conforming image to UCHAR, RAS orientation, and minimum isotropic voxels")

        if len(orig.shape) > 3 and orig.shape[3] != 1:
            raise RuntimeError(f"Multiple input frames ({orig.shape[3]}) not supported!")

        # Check affine if image is nifti image
        if img_file.suffix == ".nii" or img_file.suffixes[-2:] == [".nii", ".gz"]:
            if not check_affine_in_nifti(cast(nib.nifti1.Nifti1Image | nib.nifti2.Nifti1Image, orig), logger=logger):
                raise RuntimeError("Inconsistency in nifti-header!")

        # conform ; orig will remain the same class
        orig = conform(orig, order=order, **conform_kwargs)

    # Return header and affine information
    return orig.header, orig.affine, np.asanyarray(orig.dataobj)


def load_image(
        file: str | Path,
        name: str = "image",
        **kwargs,
) -> tuple[nibabelImage, np.ndarray]:
    """
    Load file 'file' with nibabel, including all data.

    Parameters
    ----------
    file : Path, str
        Path to the file to load.
    name : str, default="image"
        Name of the file (optional), only effects error messages.
    **kwargs :
        Additional keyword arguments.

    Returns
    -------
    the_image : nibabelImage
        The SpatialImage object from nibabel of the conformed image (including updated affine).
    the_data : np.ndarray
        The data of the conformed image.

    Raises
    ------
    IOError
        Failed loading the file
        nibabel releases the GIL, so the following is a parallel example.
        {
        >>> from concurrent.futures import ThreadPoolExecutor
        >>> with ThreadPoolExecutor() as pool:
        >>>     future1 = pool.submit(load_image, filename1)
        >>>     future2 = pool.submit(load_image, filename2)
        >>>     image, data = future1.result()
        >>>     image2, data2 = future2.result()
        }
    """
    try:
        img = cast(nibabelImage, nib.load(file, **kwargs))
    except (OSError, FileNotFoundError) as e:
        raise OSError(f"Failed loading the {name} '{file}' with error: {e.args[0]}") from e
    return img, np.asarray(img.dataobj)


def load_maybe_conform(
        file: Path | str,
        alt_file: Path | str,
        **conform_kwargs,
) -> tuple[Path, nibabelImage, np.ndarray]:
    """
    Load an image by file, check whether it is conformed to vox_size and conform to
    vox_size if it is not.

    Parameters
    ----------
    file : Path, str
        Path to the file to load.
    alt_file : Path, str
        Alternative file to interpolate from.
    **conform_kwargs
        Additional parameters to conform and is_conform.

    Returns
    -------
    Path
        The path to the file.
    nibabelImage
        The file container object including the corrected header.
    np.ndarray
        The data loaded from the file.

    See Also
    --------
    FastSurferCNN.data_loader.conform.conform
        For additional parameters supported via `conform_kwargs`.
    """
    file = Path(file)
    alt_file = Path(alt_file)
    conform_kwargs_is_conform = dict(conform_kwargs.items())
    del conform_kwargs_is_conform["order"]

    _is_conform, img = False, None
    if file.is_file():
        # see if the file is 1mm
        img = cast(nibabelImage, nib.load(file))
        # is_conform only needs the header, not the data
        _is_conform = is_conform(img, **conform_kwargs_is_conform, verbose=False, vox_eps=0.1)

    if _is_conform and img is not None:
        # calling np.asarray here, forces the load of img.dataobj into memory
        # (which is parallel with other operations, if done here)
        data = np.asarray(img.dataobj)
        dst_file = file
    else:
        # the image is not conformed to 1mm, do this now.
        fileext = [ext for ext in SUPPORTED_OUTPUT_FILE_FORMATS if file.name.endswith("." + ext)]
        if len(fileext) != 1:
            raise RuntimeError(
                f"Invalid file extension of conf_name: {file}, must be one of {SUPPORTED_OUTPUT_FILE_FORMATS}."
            )
        file_no_fileext = str(file)[:-len(fileext[0]) - 1]
        vox_size = conform_kwargs.get("vox_size", 1.0)
        vox_suffix = ".min" if vox_size == "min" else f".{str(vox_size).replace('.', '')}mm"
        if not file_no_fileext.endswith(vox_suffix):
            file_no_fileext += vox_suffix
        # if the orig file is neither absolute nor in the subject path, use the conformed file
        src_file = alt_file if alt_file.is_file() else file
        if not alt_file.is_file():
            LOGGER.warning(
                f"No valid alternative file (e.g. orig, here: {alt_file}) was given to interpolate from, so we might "
                f"lose quality due to multiple chained interpolations. "
            )

        dst_file = Path(file_no_fileext + "." + fileext[0])
        # conform to 1mm
        header, affine, data = load_and_conform_image(
            src_file, logger=logging.getLogger(__name__ + ".conform"), **conform_kwargs,
        )

        # after conforming, save the conformed file
        save_image(header, affine, data, dst_file)
        img = as_mgh_image(data, affine, header)
    return dst_file, img, data


def fits_dtype(array: np.ndarray, dtype: npt.DTypeLike) -> bool:
    """
    Whether every value of `array` survives being stored as `dtype`.

    Only integer targets are checked, since that is where this pipeline loses data: a float rounded
    into an integer loses its fraction, and a label outside the range is clipped. A floating-point
    target is reported as fitting, which ignores the precision a very large integer or a float64
    would lose, because MGH offers nothing wider than float32 to move it to.

    Parameters
    ----------
    array : np.ndarray
        The data to store.
    dtype : npt.DTypeLike
        The type to store it as.

    Returns
    -------
    bool
        False if storing `array` as `dtype` would round or clip a value.
    """
    dtype = np.dtype(dtype)
    if not np.issubdtype(dtype, np.integer):
        return True
    if not np.issubdtype(array.dtype, np.integer):
        return False
    if np.can_cast(array.dtype, dtype):
        return True
    limits = np.iinfo(dtype)
    return array.size == 0 or bool(limits.min <= array.min() and array.max() <= limits.max)


def as_mgh_image(
        data: np.ndarray,
        affine: AffineMatrix4x4,
        header: _Header | None = None,
        prefer_dtype: npt.DTypeLike | None = None,
) -> nib.MGHImage:
    """
    Build an MGHImage from data, affine and header, keeping what the conversion would drop.

    `MGHHeader.from_header` carries neither the field of view nor the data type over from a non-MGH
    header, so an .mgz written with a header that came from a .nii ended up with `fov=0` and stored
    as float32, whatever the data or the caller asked for. Both are restored here, so the file does
    not depend on the container its header came from. The fov is the largest of the three extents,
    which is what FreeSurfer keeps, and deriving it from `data` also gets it right when the header
    is inherited from a volume of a different shape.

    The type is never narrowed past what the data holds: floating-point data is not stored as an
    integer type, and neither is a value the header's type cannot represent, since both lose data
    silently. Where the header does not fit, the narrowest type MGH can store that does is used.

    Parameters
    ----------
    data : np.ndarray
        An array containing image data.
    affine : AffineMatrix4x4
        Image affine information.
    header : _Header, optional
        Image header information; a non-MGH header is converted.
    prefer_dtype : npt.DTypeLike, optional
        A narrower type to use if the data fits it, for callers who know what their output should
        be, such as the uchar aseg files. Ignored when it would round or clip a value, so it cannot
        be used to force a lossy type; `save_image` takes a `dtype` for that.

    Returns
    -------
    nib.MGHImage
        The image, with `fov` and the data type set.
    """
    array = np.asanyarray(data)
    # without a header, nibabel raises while constructing an image whose type MGH cannot store, so
    # give it a default one and settle the type below, where the fallback is
    img = nib.MGHImage(array, affine, MGHHeader() if header is None else header)
    zooms = img.header.get_zooms()
    img.header["fov"] = max(d * z for d, z in zip(img.shape[:3], zooms[:3], strict=True))

    data_dtype = np.dtype(array.dtype if header is None else header.get_data_dtype())
    if prefer_dtype is not None and fits_dtype(array, prefer_dtype):
        data_dtype = np.dtype(prefer_dtype)
    elif np.issubdtype(array.dtype, np.floating) and np.issubdtype(data_dtype, np.integer):
        # an integer type would round floating-point data away, as it did to the CC soft labels,
        # which are probabilities written with the header of the conformed image
        data_dtype = np.dtype(np.float32)
    elif not fits_dtype(array, data_dtype):
        # a header narrower than its data would clip it, silently. Widening to the array's own type
        # is not enough: MGH cannot store int64, and falling back would clip after all.
        wider = next(
            (np.dtype(t) for t in MGH_INT_DTYPES if fits_dtype(array, t)), np.dtype(np.float32)
        )
        LOGGER.warning(f"The data does not fit {data_dtype}, writing {wider} instead to avoid clipping.")
        data_dtype = wider
    try:
        img.set_data_dtype(data_dtype)
    except MGHError:
        # MGH stores uint8, uint16, int16, int32 and float32 only
        LOGGER.warning(
            f"An MGH file cannot store {data_dtype}, writing {img.get_data_dtype()} instead."
        )
    return img


# Save image routine
def save_image(
        header_info: _Header,
        affine_info: AffineMatrix4x4,
        img_array: np.ndarray,
        save_as: str | Path,
        dtype: npt.DTypeLike | None = None
) -> None:
    """
    Save an image (nibabel MGHImage), according to the desired output file format.

    Supported formats are defined in supported_output_file_formats. Saves predictions to save_as.

    Parameters
    ----------
    header_info : _Header
        Image header information.
    affine_info : AffineMatrix4x4
        Image affine information.
    img_array : np.ndarray
        An array containing image data.
    save_as : Path, str
        Name under which to save prediction; this determines output file format.
    dtype : npt.DTypeLike, optional
        Image array type; if provided, the image object is explicitly set to match this type.
    """
    save_as = Path(save_as)
    valid_ext = save_as.suffix[1:] in SUPPORTED_OUTPUT_FILE_FORMATS or save_as.suffixes[-2:] == [".nii", ".gz"]
    assert valid_ext, f"Output filename does not contain a supported file format {SUPPORTED_OUTPUT_FILE_FORMATS}!"

    if save_as.suffix == ".mgz":
        mgh_img = as_mgh_image(img_array, affine_info, header_info)
    elif save_as.suffix == ".nii" or save_as.suffixes[-2:] == [".nii", ".gz"]:
        mgh_img = nib.nifti1.Nifti1Pair(img_array, affine_info, header_info)
    else:
        raise ValueError(f"Invalid file extension of {save_as}, must be .mgz, .nii or .nii.gz!")

    if dtype is not None:
        # an explicit dtype is the caller's decision, so it is applied as given even where it loses
        # data, but it says so, and it falls back like the inferred one rather than aborting the run
        if not fits_dtype(np.asanyarray(img_array), dtype):
            LOGGER.warning(f"The data does not fit the requested {np.dtype(dtype)}, values will clip.")
        try:
            mgh_img.set_data_dtype(dtype)
        except MGHError:
            LOGGER.warning(
                f"An MGH file cannot store {np.dtype(dtype)}, writing {mgh_img.get_data_dtype()} "
                f"instead."
            )

    if save_as.suffix in (".mgz", ".nii"):
        nib.save(mgh_img, save_as)
    elif save_as.suffixes[-2:] == [".nii", ".gz"]:
        # For correct outputs, nii.gz files should be saved using the nifti1
        # sub-module's save():
        nib.nifti1.save(mgh_img, str(save_as))


# Transformation for mapping
def transform_axial(
        vol: npt.NDArray,
        coronal2axial: bool = True
) -> np.ndarray:
    """
    Transform volume into Axial axis and back.

    Parameters
    ----------
    vol : npt.NDArray
        Image volume to transform.
    coronal2axial : bool
        Transform from coronal to axial = True (default).

    Returns
    -------
    np.ndarray
        Transformed image.
    """
    if coronal2axial:
        return np.moveaxis(vol, [0, 1, 2], [1, 2, 0])
    else:
        return np.moveaxis(vol, [0, 1, 2], [2, 0, 1])


def transform_sagittal(
        vol: npt.NDArray,
        coronal2sagittal: bool = True
) -> np.ndarray:
    """
    Transform volume into Sagittal axis and back.

    Parameters
    ----------
    vol : npt.NDArray
        Image volume to transform.
    coronal2sagittal : bool
        Transform from coronal to sagittal = True (default).

    Returns
    -------
    np.ndarray:
        Transformed image.
    """
    if coronal2sagittal:
        return np.moveaxis(vol, [0, 1, 2], [2, 1, 0])
    else:
        return np.moveaxis(vol, [0, 1, 2], [2, 1, 0])


# Thick slice generator (for eval) and blank slices filter (for training)
def get_thick_slices(
        img_data: npt.NDArray,
        slice_thickness: int = 3
) -> np.ndarray:
    """
    Extract thick slices from the image.

    Feed slice_thickness preceding and succeeding slices to network,
    label only middle one.

    Parameters
    ----------
    img_data : npt.NDArray
        3D MRI image read in with nibabel.
    slice_thickness : int
        Number of slices to stack on top and below slice of interest (default=3).

    Returns
    -------
    np.ndarray
        Image data with the thick slices of the n-th axis appended into the n+1-th axis.
    """
    img_data_pad = np.pad(
        img_data, ((0, 0), (0, 0), (slice_thickness, slice_thickness)), mode="edge"
    )
    from numpy.lib.stride_tricks import sliding_window_view

    # sliding_window_view will automatically create thick slices through a sliding window, but as this in only a view,
    # less memory copies are required
    return sliding_window_view(img_data_pad, 2 * slice_thickness + 1, axis=2)


def filter_blank_slices_thick(
        img_vol: npt.NDArray,
        label_vol: npt.NDArray,
        weight_vol: npt.NDArray,
        threshold: int = 50
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Filter blank slices from the volume using the label volume.

    Parameters
    ----------
    img_vol : npt.NDArray
        Orig image volume.
    label_vol : npt.NDArray
        Label images (ground truth).
    weight_vol : npt.NDArray
        Weight corresponding to labels.
    threshold : int
        Threshold for number of pixels needed to keep slice (below = dropped). (Default value = 50).

    Returns
    -------
    filtered img_vol : np.ndarray
        Image volume with blank slices removed.
    label_vol : np.ndarray
        Label volume with blank slices removed.
    weight_vol : np.ndarray
        Weight volume with blank slices removed.
    """
    # Get indices of all slices with more than threshold labels/pixels
    select_slices = np.sum(label_vol, axis=(0, 1)) > threshold

    # Retain only slices with more than threshold labels/pixels
    img_vol = img_vol[:, :, select_slices, :]
    label_vol = label_vol[:, :, select_slices]
    weight_vol = weight_vol[:, :, select_slices]

    return img_vol, label_vol, weight_vol


# weight map generator
def create_weight_mask(
        mapped_aseg: npt.NDArray,
        max_weight: int = 5,
        max_edge_weight: int = 5,
        max_hires_weight: int | None = None,
        ctx_thresh: int = 33,
        mean_filter: bool = False,
        cortex_mask: bool = True,
        gradient: bool = True
) -> np.ndarray:
    """
    Create weighted mask - with median frequency balancing and edge-weighting.

    Parameters
    ----------
    mapped_aseg : np.ndarray
        Segmentation to create weight mask from.
    max_weight : int
        Maximal weight on median weights (cap at this value). (Default value = 5).
    max_edge_weight : int
        Maximal weight on gradient weight (cap at this value). (Default value = 5).
    max_hires_weight : int
        Maximal weight on hires weight (cap at this value). (Default value = None).
    ctx_thresh : int
        Label value of cortex (above = cortical parcels). (Default value = 33).
    mean_filter : bool
        Flag, set to add mean_filter mask (default = False).
    cortex_mask : bool
        Flag, set to create cortex weight mask (default=True).
    gradient : bool
        Flag, set to create gradient mask (default = True).

    Returns
    -------
    np.ndarray
        Weights.
    """
    unique, counts = np.unique(mapped_aseg, return_counts=True)

    # Median Frequency Balancing
    class_wise_weights = np.median(counts) / counts
    class_wise_weights[class_wise_weights > max_weight] = max_weight
    (h, w, d) = mapped_aseg.shape

    weights_mask = np.reshape(class_wise_weights[mapped_aseg.ravel()], (h, w, d))

    # Gradient Weighting
    if gradient:
        (gx, gy, gz) = np.gradient(mapped_aseg)
        grad_weight = max_edge_weight * np.asarray(
            np.power(np.power(gx, 2) + np.power(gy, 2) + np.power(gz, 2), 0.5) > 0,
            dtype="float",
        )

        weights_mask += grad_weight

    if max_hires_weight is not None:
        # High-res Weighting
        print(
            "Adding hires weight mask deep sulci and WM with weight ", max_hires_weight
        )
        mask1 = deep_sulci_and_wm_strand_mask(
            mapped_aseg, structure=np.ones((3, 3, 3)), ctx_thresh=ctx_thresh
        )
        weights_mask += mask1 * max_hires_weight

        if cortex_mask:
            print("Adding cortex mask with weight ", max_hires_weight)
            mask2 = cortex_border_mask(
                mapped_aseg, structure=np.ones((3, 3, 3)), ctx_thresh=ctx_thresh
            )
            weights_mask += mask2 * (max_hires_weight) // 2

    if mean_filter:
        weights_mask = uniform_filter(weights_mask, size=3)

    return weights_mask


def cortex_border_mask(
        label: npt.NDArray,
        structure: npt.NDArray,
        ctx_thresh: int = 33
) -> np.ndarray:
    """
    Erode the cortex of a given mri image to create the inner gray matter mask (outer most cortex voxels).

    Parameters
    ----------
    label : npt.NDArray
        Ground truth labels.
    structure : npt.NDArray
        Structuring element to erode with.
    ctx_thresh : int
        Label value of cortex (above = cortical parcels). Defaults to 33.

    Returns
    -------
    np.ndarray
        Inner grey matter layer.
    """
    # create aseg brainmask, erode it and subtract from itself
    bm = np.clip(label, a_max=1, a_min=0)
    eroded = binary_erosion(bm, structure=structure)
    diff_im = np.logical_xor(eroded, bm)

    # only keep values associated with the cortex
    diff_im[(label <= ctx_thresh)] = 0  # > 33 (>19) = > 1002 in FS space (full (sag)),
    print("Remaining voxels cortex border: ", np.unique(diff_im, return_counts=True))
    return diff_im


def deep_sulci_and_wm_strand_mask(
        volume: npt.NDArray,
        structure: npt.NDArray,
        iteration: int = 1,
        ctx_thresh: int = 33
) -> np.ndarray:
    """
    Get a binary mask of deep sulci and small white matter strands by using binary closing (erosion and dilation).

    Parameters
    ----------
    volume : npt.NDArray
        Loaded image (aseg, label space).
    structure : npt.NDArray
        Structuring element (e.g. np.ones((3, 3, 3))).
    iteration : int
        Number of times mask should be dilated + eroded. Defaults to 1.
    ctx_thresh : int
        Label value of cortex (above = cortical parcels). Defaults to 33.

    Returns
    -------
    np.ndarray
        Sulcus + wm mask.
    """
    # Binarize label image (cortex = 1, everything else = 0)
    empty_im = np.zeros(shape=volume.shape)
    empty_im[volume > ctx_thresh] = 1  # > 33 (>19) = >1002 in FS LUT (full (sag))

    # Erode the image
    eroded = binary_closing(empty_im, iterations=iteration, structure=structure)

    # Get difference between eroded and original image
    diff_image = np.logical_xor(empty_im, eroded)
    print("Remaining voxels sulci/wm strand: ", np.unique(diff_image, return_counts=True))
    return diff_image


# Label mapping functions (to aparc (eval) and to label (train))
def read_classes_from_lut(lut_file: str | Path):
    """
    Modify from datautils to allow support for FreeSurfer-distributed ColorLUTs.

    Read in **FreeSurfer-like** LUT table.

    Parameters
    ----------
    lut_file : Path, str
        The path and name of FreeSurfer-style LUT file with classes of interest.
        Example entry:
        ID LabelName  R   G   B   A
        0   Unknown   0   0   0   0
        1   Left-Cerebral-Exterior 70  130 180 0
        ...

    Returns
    -------
    pandas.DataFrame
        DataFrame with ids present, name of ids, color for plotting.
    """
    if not isinstance(lut_file, Path):
        lut_file = Path(lut_file)
    if lut_file.suffix == ".tsv":
        return pd.read_csv(lut_file, sep="\t")

    # Read in file, default factory must be a dtype factory
    names: defaultdict[str, str] = defaultdict(lambda: "str",
        ID="int",
        LabelName="str",
        Red="int",
        Green="int",
        Blue="int",
        Alpha="int",
    )
    if lut_file.suffix == ".csv":
        _sep = ","
    elif lut_file.suffix == ".txt":
        _sep = "\\s+"
    else:
        raise RuntimeError(f"Unknown LUT file extension {lut_file}, must be csv, txt or tsv.")
    return pd.read_csv(
        lut_file,
        index_col=False,
        skip_blank_lines=True,
        comment="#",
        header=None,
        names=list(names.keys()),
        dtype=names,
        sep=_sep,
    )


def map_label2aparc_aseg(
        mapped_aseg: torch.Tensor,
        labels: torch.Tensor | npt.NDArray
) -> torch.Tensor:
    """
    Perform look-up table mapping from sequential label space to LUT space.

    Parameters
    ----------
    mapped_aseg : torch.Tensor
        Label space segmentation (aparc.DKTatlas + aseg).
    labels : Union[torch.Tensor, npt.NDArray]
        List of labels defining LUT space.

    Returns
    -------
    torch.Tensor
        Labels in LUT space.
    """
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)
    labels = labels.to(mapped_aseg.device)
    return labels[mapped_aseg]


def clean_cortex_labels(aparc: npt.NDArray) -> np.ndarray:
    """
    Clean up aparc segmentations.

    Map undetermined and optic chiasma to BKG
    Map Hypointensity classes to one
    Vessel to WM
    5th Ventricle to CSF
    Remaining cortical labels to BKG.

    Parameters
    ----------
    aparc : npt.NDArray
        Aparc segmentations.

    Returns
    -------
    np.ndarray
        Cleaned aparc.
    """
    aparc[aparc == 80] = 77  # Hypointensities Class
    aparc[aparc == 85] = 0  # Optic Chiasma to BKG
    aparc[aparc == 62] = 41  # Right Vessel to Right WM
    aparc[aparc == 30] = 2  # Left Vessel to Left WM
    aparc[aparc == 72] = 24  # 5th Ventricle to CSF
    aparc[aparc == 29] = 0  # left-undetermined to 0
    aparc[aparc == 61] = 0  # right-undetermined to 0

    aparc[aparc == 3] = 0  # Map Remaining Cortical labels to background
    aparc[aparc == 42] = 0
    return aparc


def fill_unknown_labels_per_hemi(
        gt: npt.NDArray,
        unknown_label: int,
        cortex_stop: int
) -> np.ndarray:
    """
    Replace label 1000 (lh unknown) and 2000 (rh unknown) with closest class for each voxel.

    Parameters
    ----------
    gt : npt.NDArray
        Ground truth segmentation with class unknown.
    unknown_label : int
        Class label for unknown (lh: 1000, rh: 2000).
    cortex_stop : int
        Class label at which cortical labels of this hemi stop (lh: 2000, rh: 3000).

    Returns
    -------
    np.ndarray
        Ground truth segmentation with all classes.
    """
    # Define shape of image and dilation element
    h, w, d = gt.shape
    struct1 = generate_binary_structure(3, 2)

    # Get indices of unknown labels, dilate them to get closest surrounding parcels
    unknown = gt == unknown_label
    unknown = morphology.binary_dilation(unknown, struct1) ^ unknown
    list_parcels = np.unique(gt[unknown])

    # Mask all subcortical structures (fill unknown with closest cortical parcels only)
    mask = (list_parcels > unknown_label) & (list_parcels < cortex_stop)
    list_parcels = list_parcels[mask]

    # For each closest parcel, blur label with gaussian filter (spread), append resulting blurred images
    blur_vals = np.ndarray((h, w, d, 0), dtype=float)
    for idx in range(len(list_parcels)):
        aseg_blur = filters.gaussian_filter(
            1000 * np.asarray(gt == list_parcels[idx], dtype=float), sigma=5
        )
        blur_vals = np.append(blur_vals, np.expand_dims(aseg_blur, axis=3), axis=3)

    # Get for each position parcel with maximum value after blurring (= closest parcel)
    unknown = np.argmax(blur_vals, axis=3)
    unknown = np.reshape(list_parcels[unknown.ravel()], (h, w, d))

    # Assign the determined closest parcel to the unknown class (case-by-case basis)
    mask = gt == unknown_label
    gt[mask] = unknown[mask]

    return gt


def fuse_cortex_labels(aparc: npt.NDArray) -> np.ndarray:
    """
    Fuse cortical parcels on left/right hemisphere (reduce aparc classes).

    Parameters
    ----------
    aparc : npt.NDArray
        Anatomical segmentation with cortical parcels.

    Returns
    -------
    np.ndarray
        Anatomical segmentation with reduced number of cortical parcels.
    """
    aparc_temp = aparc.copy()

    # Map undetermined classes
    aparc = clean_cortex_labels(aparc)

    # Fill label unknown
    if np.any(aparc == 1000):
        aparc = fill_unknown_labels_per_hemi(aparc, 1000, 2000)
    if np.any(aparc == 2000):
        aparc = fill_unknown_labels_per_hemi(aparc, 2000, 3000)

    # De-lateralize parcels
    cortical_label_mask = (aparc >= 2000) & (aparc <= 2999)
    aparc[cortical_label_mask] = aparc[cortical_label_mask] - 1000

    # Re-lateralize Cortical parcels in close proximity
    aparc[aparc_temp == 2014] = 2014
    aparc[aparc_temp == 2028] = 2028
    aparc[aparc_temp == 2012] = 2012
    aparc[aparc_temp == 2016] = 2016
    aparc[aparc_temp == 2002] = 2002
    aparc[aparc_temp == 2023] = 2023
    aparc[aparc_temp == 2017] = 2017
    aparc[aparc_temp == 2024] = 2024
    aparc[aparc_temp == 2010] = 2010
    aparc[aparc_temp == 2013] = 2013
    aparc[aparc_temp == 2025] = 2025
    aparc[aparc_temp == 2022] = 2022
    aparc[aparc_temp == 2021] = 2021
    aparc[aparc_temp == 2005] = 2005

    return aparc


def split_cortex_labels(aparc: npt.NDArray) -> np.ndarray:
    """
    Splot cortex labels to completely de-lateralize structures.

    Parameters
    ----------
    aparc : npt.NDArray
        Anatomical segmentation and parcellation from network.

    Returns
    -------
    np.ndarray
        Re-lateralized aparc.
    """
    # Post processing - Splitting classes
    # Quick Fix for 2026 vs 1026; 2029 vs. 1029; 2025 vs. 1025
    rh_wm = get_largest_cc(aparc == 41)
    lh_wm = get_largest_cc(aparc == 2)
    rh_wm = regionprops(label(rh_wm, background=0))
    lh_wm = regionprops(label(lh_wm, background=0))
    centroid_rh = np.asarray(rh_wm[0].centroid)
    centroid_lh = np.asarray(lh_wm[0].centroid)

    labels_list = np.array(
        [
            1003,
            1006,
            1007,
            1008,
            1009,
            1011,
            1015,
            1018,
            1019,
            1020,
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

    for label_current in labels_list:

        label_img = label(aparc == label_current, connectivity=3, background=0)

        for region in regionprops(label_img):

            if region.label != 0:  # To avoid background

                if np.linalg.norm(
                    np.asarray(region.centroid) - centroid_rh
                ) < np.linalg.norm(np.asarray(region.centroid) - centroid_lh):
                    mask = label_img == region.label
                    aparc[mask] = label_current + 1000

    # Quick Fixes for overlapping classes
    aseg_lh = filters.gaussian_filter(
        1000 * np.asarray(aparc == 2, dtype=float), sigma=3
    )
    aseg_rh = filters.gaussian_filter(
        1000 * np.asarray(aparc == 41, dtype=float), sigma=3
    )

    lh_rh_split = np.argmax(
        np.concatenate(
            (np.expand_dims(aseg_lh, axis=3), np.expand_dims(aseg_rh, axis=3)), axis=3
        ),
        axis=3,
    )

    # Problematic classes: 1026, 1011, 1029, 1019
    for prob_class_lh in [1011, 1019, 1026, 1029]:
        prob_class_rh = prob_class_lh + 1000
        mask_prob_class = (aparc == prob_class_lh) | (aparc == prob_class_rh)
        mask_lh = np.logical_and(mask_prob_class, lh_rh_split == 0)
        mask_rh = np.logical_and(mask_prob_class, lh_rh_split == 1)

        aparc[mask_lh] = prob_class_lh
        aparc[mask_rh] = prob_class_rh

    return aparc


def unify_lateralized_labels(
        lut: str | pd.DataFrame,
        combi: tuple[str, str] = ("Left-", "Right-")
) -> Mapping:
    """
    Generate lookup dictionary of left-right labels.

    Parameters
    ----------
    lut : Union[str, pd.DataFrame]
        Either lut-file string to load or pandas dataframe
        Example entry:
        ID LabelName  R   G   B   A
        0   Unknown   0   0   0   0
        1   Left-Cerebral-Exterior 70  130 180 0.
    combi : Tuple[str, str]
        Prefix or labelnames to combine. Default: Left- and Right-.

    Returns
    -------
    Mapping
        Dictionary mapping between left and right hemispheres.
    """
    if isinstance(lut, str):
        lut = read_classes_from_lut(lut)
    left = lut[["ID", "LabelName"]][lut["LabelName"].str.startswith(combi[0])]
    right = lut[["ID", "LabelName"]][lut["LabelName"].str.startswith(combi[1])]
    left["LabelName"] = left["LabelName"].str.removeprefix(combi[0])
    right["LabelName"] = right["LabelName"].str.removeprefix(combi[1])
    mapp = left.merge(right, on="LabelName")
    return pd.Series(mapp.ID_y.values, index=mapp.ID_x).to_dict()


def get_labels_from_lut(
        lut: str | pd.DataFrame,
        label_extract: tuple[str, str] = ("Left-", "ctx-rh")
) -> tuple[np.ndarray[Shape1d, np.dtype[np.integer]], np.ndarray[Shape1d, np.dtype[np.integer]]]:
    """
    Extract labels from the lookup tables.

    Parameters
    ----------
    lut : Union[str, pd.DataFrame]
        FreeSurfer like LookUp Table (either path to it
        or already loaded as pandas DataFrame.
        Example entry:
        ID LabelName  R   G   B   A
        0   Unknown   0   0   0   0
        1   Left-Cerebral-Exterior 70  130 180 0.
    label_extract : Tuple[str, str]
        Suffix of label names to mask for sagittal labels
        Default: "Left-" and "ctx-rh".

    Returns
    -------
    np.ndarray
        Full label list.
    np.ndarray
        Sagittal label list.
    """
    _lut = read_classes_from_lut(lut) if isinstance(lut, str) else lut
    mask = _lut["LabelName"].str.startswith(label_extract)
    return np.asarray(_lut["ID"].values), np.asarray(_lut["ID"][~mask].values)


def map_aparc_aseg2label(
        aseg: npt.NDArray,
        labels: npt.NDArray,
        labels_sag: npt.NDArray,
        sagittal_lut_dict: Mapping,
        aseg_nocc: npt.NDArray | None = None,
        processing:  str = "aparc"
) ->  tuple[np.ndarray, np.ndarray]:
    """
    Perform look-up table mapping of aparc.DKTatlas+aseg.mgz data to label space.

    Parameters
    ----------
    aseg : npt.NDArray
        Ground truth aparc+aseg.
    labels : npt.NDArray
        Labels to use (extracted from LUT with get_labels_from_lut).
    labels_sag : npt.NDArray
        Sagittal labels to use (extracted from LUT with
        get_labels_from_lut).
    sagittal_lut_dict : Mapping
        Left-right label mapping (can be extracted with
        unify_lateralized_labels from LUT).
    aseg_nocc : Optional[npt.NDArray]
        Ground truth aseg without corpus callosum segmentation (Default value = None).
    processing : str
        Should be set to "aparc" or "aseg" for additional mappings (hard-coded) (Default value = "aparc").

    Returns
    -------
    np.ndarray
        Mapped aseg for coronal and axial.
    np.ndarray
        Mapped aseg for sagittal.
    """
    # If corpus callosum is not removed yet, do it now
    if aseg_nocc is not None:
        cc_mask = (aseg >= 251) & (aseg <= 255)
        aseg[cc_mask] = aseg_nocc[cc_mask]

    if processing == "aparc":
        LOGGER.info("APARC PROCESSING")
        aseg = fuse_cortex_labels(aseg)

    elif processing == "aseg":
        LOGGER.info("ASEG PROCESSING")
        aseg[aseg == 1000] = 3  # Map unknown to cortex
        aseg[aseg == 2000] = 42
        aseg[aseg == 80] = 77  # Hypointensities Class
        aseg[aseg == 85] = 0  # Optic Chiasma to BKG
        aseg[aseg == 62] = 41  # Right Vessel to Right WM
        aseg[aseg == 30] = 2  # Left Vessel to Left WM
        aseg[aseg == 72] = 24  # 5th Ventricle to CSF

        assert not np.any(
            251 <= aseg
        ), f"Error: CC classes (251-255) still exist in aseg {np.unique(aseg)}"
        assert np.any(aseg == 3) and np.any(
            aseg == 42
        ), f"Error: no cortical marker detected {np.unique(aseg)}"

    assert set(labels).issuperset(
        np.unique(aseg)
    ), f"Error: segmentation image contains classes not listed in the labels: \n{np.unique(aseg)}\n{labels}"

    h, w, d = aseg.shape
    lut_aseg = np.zeros(max(labels) + 1, dtype="int")
    for idx, value in enumerate(labels):
        lut_aseg[value] = idx

    # Remap Label Classes - Perform LUT Mapping - Coronal, Axial
    mapped_aseg = lut_aseg.ravel()[aseg.ravel()]
    mapped_aseg = mapped_aseg.reshape((h, w, d))

    if processing == "aparc":
        cortical_label_mask = (aseg >= 2000) & (aseg <= 2999)
        aseg[cortical_label_mask] = aseg[cortical_label_mask] - 1000

    # For sagittal, all Left hemispheres will be mapped to right, ctx the otherway round
    # If you use your own LUT, make sure all per-hemi labels have the corresponding prefix
    # Map Sagittal Labels
    for left, right in sagittal_lut_dict.items():
        aseg[aseg == left] = right

    h, w, d = aseg.shape
    lut_aseg = np.zeros(max(labels_sag) + 1, dtype="int")
    for idx, value in enumerate(labels_sag):
        lut_aseg[value] = idx

    # Remap Label Classes - Perform LUT Mapping - Coronal, Axial
    mapped_aseg_sag = lut_aseg.ravel()[aseg.ravel()]
    mapped_aseg_sag = mapped_aseg_sag.reshape((h, w, d))

    return mapped_aseg, mapped_aseg_sag


def sagittal_coronal_remap_lookup(x: int) -> int:
    """
    Convert left labels to corresponding right labels for aseg with dictionary mapping.

    Parameters
    ----------
    x : int
        Label to look up.

    Returns
    -------
    np.ndarray
        Mapped label.
    """
    return {
        2: 41,
        3: 42,
        4: 43,
        5: 44,
        7: 46,
        8: 47,
        10: 49,
        11: 50,
        12: 51,
        13: 52,
        17: 53,
        18: 54,
        26: 58,
        28: 60,
        31: 63,
    }[x]


def infer_mapping_from_lut(
        num_classes_full: int,
        lut: str | pd.DataFrame
) -> np.ndarray:
    """
    Guess the mapping from a lookup table.

    Parameters
    ----------
    num_classes_full : int
        Number of classes.
    lut : Union[str, pd.DataFrame]
        Look-up table listing class labels.

    Returns
    -------
    np.ndarray
        List of indexes for.
    """
    labels, labels_sag = unify_lateralized_labels(lut)
    idx_list = np.ndarray(shape=(num_classes_full,), dtype=np.int16)
    for idx in range(len(labels)):
        idx_in_sag = np.where(labels_sag == labels[idx])[0]
        if idx_in_sag.size == 0:  # Empty not subcortical
            idx_in_sag = np.where(labels_sag == (labels[idx] - 1000))[0]

        if idx_in_sag.size == 0:
            current_label_sag = sagittal_coronal_remap_lookup(labels[idx])
            idx_in_sag = np.where(labels_sag == current_label_sag)[0]

        idx_list[idx] = idx_in_sag
    return idx_list


def map_prediction_sagittal2full(
        prediction_sag: npt.NDArray,
        num_classes: int = 51,
        lut: str | None = None
) -> np.ndarray:
    """
    Remap the prediction on the sagittal network to full label space used by coronal and axial networks.

    Create full aparc.DKTatlas+aseg.mgz.

    Parameters
    ----------
    prediction_sag : npt.NDArray
        Sagittal prediction (labels).
    num_classes : int
        Number of SAGITTAL classes (96 for full classes, 51 for hemi split, 21 for aseg) (Default value = 51).
    lut : Optional[str]
        Look-up table listing class labels (Default value = None).

    Returns
    -------
    np.ndarray
        Remapped prediction.
    """
    r = range
    _idx = []
    if num_classes == 96:
        _idx = [[0], r(5, 14), r(1, 4), [14, 15, 4], r(16, 19), r(5, 51), r(20, 51)]
    elif num_classes == 51:
        _idx = [[0], r(5, 14), r(1, 4), [14, 15, 4], r(16, 19), r(5, 51)]
        _idx.extend([[20, 22, 27], r(29, 32), [33, 34], r(38, 43), [45]])
    elif num_classes == 21:
        _idx = [[0], r(5, 15), r(1, 4), [15, 16, 4], r(17, 20), r(5, 21)]
    if _idx:
        from itertools import chain
        idx_list = list(chain(*_idx))
    else:
        assert lut is not None, "lut is not defined!"
        idx_list = infer_mapping_from_lut(num_classes, lut)
    return prediction_sag[:, idx_list, :, :]


# Clean up and class separation
def bbox_3d(
        img: npt.NDArray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract the three-dimensional bounding box coordinates.

    Parameters
    ----------
    img : npt.NDArray
        Mri image.

    Returns
    -------
    np.ndarray
        Rmin.
    np.ndarray
        Rmax.
    np.ndarray
        Cmin.
    np.ndarray
        Cmax.
    np.ndarray
        Zmin.
    np.ndarray
        Zmax.
    """
    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return rmin, rmax, cmin, cmax, zmin, zmax


def get_largest_cc(segmentation: npt.NDArray) -> np.ndarray:
    """
    Find the largest connected component of segmentation.

    Parameters
    ----------
    segmentation : npt.NDArray
        Segmentation.

    Returns
    -------
    np.ndarray
        Largest connected component of segmentation (binary mask).
    """
    labels = label(segmentation, connectivity=3, background=0)

    bincount = np.bincount(labels.flat)
    background = np.argmax(bincount)
    bincount[background] = -1

    largest_cc = labels == np.argmax(bincount)

    return largest_cc
