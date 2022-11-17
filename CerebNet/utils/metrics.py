
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
import numpy as np
import torch
from scipy.ndimage import _ni_support
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion, generate_binary_structure

from FastSurferCNN.utils import logging

logger = logging.get_logger(__name__)


class DiceScore:
    """
        Accumulating the component of the dice coefficient i.e. the union and intersection
    Args:
        op (callable): a callable to update accumulator. Method's signature is `(accumulator, output)`.
            For example, to compute arithmetic mean value, `op = lambda a, x: a + x`.
        output_transform (callable, optional): a callable that is used to transform the
            :class:`~ignite.engine.Engine`'s `process_function`'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
        device (str of torch.device, optional): device specification in case of distributed computation usage.
            In most of the cases, it can be defined as "cuda:local_rank" or "cuda"
            if already set `torch.cuda.set_device(local_rank)`. By default, if a distributed process group is
            initialized and available, device is set to `cuda`.
    """

    def __init__(self,
                 num_classes,
                 class_ids=None,
                 device=None,
                 one_hot=False,
                 output_transform=lambda y_pred, y: (y_pred.data.max(1)[1], y)):
        self._device = device
        self.out_transform = output_transform
        self.class_ids = class_ids
        if self.class_ids is None:
            self.class_ids = np.arange(num_classes)
        self.n_classes = num_classes
        assert len(self.class_ids)==self.n_classes, f"Number of class ids is not correct," \
                                                    f" given {len(self.class_ids)} but {self.n_classes} is needed."
        self.one_hot = one_hot
        self.union = torch.zeros(self.n_classes, self.n_classes)
        self.intersection = torch.zeros(self.n_classes, self.n_classes)

    def reset(self):
        self.union = torch.zeros(self.n_classes, self.n_classes)
        self.intersection = torch.zeros(self.n_classes, self.n_classes)

    def _check_output_type(self, output):
        if not (isinstance(output, tuple)):
            raise TypeError("Output should a tuple consist of of torch.Tensors, but given {}".format(type(output)))

    def _update_union_intersection(self, batch_output, labels_batch):
        # self.union.to(batch_output.device)
        # self.intersection.to(batch_output.device)
        for i, c1 in enumerate(self.class_ids):
            gt = (labels_batch == c1).float()
            for j, c2 in enumerate(self.class_ids):
                pred = (batch_output == c2).float()
                self.intersection[i, j] =  self.intersection[i, j]+ torch.sum(torch.mul(gt, pred))
                self.union[i, j] = self.union[i, j] + torch.sum(gt) + torch.sum(pred)

    def update(self, output):
        self._check_output_type(output)

        if self.out_transform is not None:
            y_pred, y = self.out_transform(*output)
        else:
            y_pred, y = output

        if not isinstance(y, torch.Tensor):
            y = torch.from_numpy(y)

        if not isinstance(y_pred, torch.Tensor):
            y_pred = torch.from_numpy(y_pred)

        if self._device is not None:
            y = y.to(self._device)
            y_pred = y_pred.to(self._device)

        self._update_union_intersection(y_pred, y)

    def compute(self, per_class=False, class_idxs=None):
        dice_cm_mat = self._dice_confusion_matrix(class_idxs)
        dice_score_per_class = dice_cm_mat.diagonal()
        dice_score = dice_score_per_class.mean()
        if per_class:
            return dice_score_per_class, dice_cm_mat
        else:
            return dice_score, dice_cm_mat

    def _dice_confusion_matrix(self, class_idxs):
        dice_intersection = self.intersection.cpu().numpy()
        dice_union = self.union.cpu().numpy()
        if class_idxs is not None:
            dice_union = dice_union[class_idxs[:,None], class_idxs]
            dice_intersection = dice_intersection[class_idxs[:,None], class_idxs]
        if not (dice_union > 0).all():
            logger.info("Union of some classes are all zero")
        dice_cnf_matrix = 2 * np.divide(dice_intersection, dice_union)
        return dice_cnf_matrix


def dice_score(pred, gt):
    """
    Calculates the Dice Similarity between pred and gt.
    """
    from scipy.spatial.distance import dice
    return dice(pred.flat, gt.flat)


def volume_similarity(pred, gt):
    """
    Calculate the Volume Similarity between pred and gt.
    """
    pred_vol, gt_vol = np.sum(pred), np.save(gt)
    return 1. - np.abs(pred_vol - gt_vol) / (pred_vol + gt_vol)


# https://github.com/amanbasu/3d-prostate-segmentation/blob/master/metric_eval.py
def hd(result, reference, voxelspacing=None, connectivity=1):
    """
    Hausdorff Distance.
    Computes the (symmetric) Hausdorff Distance (HD) between the binary objects in two
    images. It is defined as the maximum surface distance between the objects.
    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    voxelspacing : float or sequence of floats, optional
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.
    connectivity : int
        The neighbourhood/connectivity considered when determining the surface
        of the binary objects. This value is passed to
        `scipy.ndimage.morphology.generate_binary_structure` and should usually be :math:`> 1`.
        Note that the connectivity influences the result in the case of the Hausdorff distance.
    Returns
    -------
    hd : float
        The symmetric Hausdorff Distance between the object(s) in ```result``` and the
        object(s) in ```reference```. The distance unit is the same as for the spacing of
        elements along each dimension, which is usually given in mm.
    See also
    --------
    :func:`assd`
    :func:`asd`
    Notes
    -----
    This is a real metric. The binary images can therefore be supplied in any order.
    """
    hd1 = __surface_distances(result, reference, voxelspacing, connectivity)
    hd2 = __surface_distances(reference, result, voxelspacing, connectivity)
    hd = max(hd1.max(), hd2.max())
    hd95 = np.percentile(np.hstack((hd1, hd2)), 95)
    hd50 = np.percentile(np.hstack((hd1, hd2)), 50)
    return hd, hd50, hd95



def hd95(result, reference, voxelspacing=None, connectivity=1):
    """
    95th percentile of the Hausdorff Distance.
    Computes the 95th percentile of the (symmetric) Hausdorff Distance (HD) between the binary objects in two
    images. Compared to the Hausdorff Distance, this metric is slightly more stable to small outliers and is
    commonly used in Biomedical Segmentation challenges.
    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    voxelspacing : float or sequence of floats, optional
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.
    connectivity : int
        The neighbourhood/connectivity considered when determining the surface
        of the binary objects. This value is passed to
        `scipy.ndimage.morphology.generate_binary_structure` and should usually be :math:`> 1`.
        Note that the connectivity influences the result in the case of the Hausdorff distance.
    Returns
    -------
    hd : float
        The symmetric Hausdorff Distance between the object(s) in ```result``` and the
        object(s) in ```reference```. The distance unit is the same as for the spacing of
        elements along each dimension, which is usually given in mm.
    See also
    --------
    :func:`hd`
    Notes
    -----
    This is a real metric. The binary images can therefore be supplied in any order.
    """
    hd1 = __surface_distances(result, reference, voxelspacing, connectivity)
    hd2 = __surface_distances(reference, result, voxelspacing, connectivity)
    hd95 = np.percentile(np.hstack((hd1, hd2)), 95)
    return hd95



def __surface_distances(result, reference, voxelspacing=None, connectivity=1):
    """
    The distances between the surface voxel of binary objects in result and their
    nearest partner surface voxel of a binary object in reference.
    """
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))
    if voxelspacing is not None:
        voxelspacing = _ni_support._normalize_sequence(voxelspacing, result.ndim)
        voxelspacing = np.asarray(voxelspacing, dtype=np.float64)
        if not voxelspacing.flags.contiguous:
            voxelspacing = voxelspacing.copy()

    # binary structure
    footprint = generate_binary_structure(result.ndim, connectivity)

    # test for emptiness
    if 0 == np.count_nonzero(result):
        raise RuntimeError('The first supplied array does not contain any binary object.')
    if 0 == np.count_nonzero(reference):
        raise RuntimeError('The second supplied array does not contain any binary object.')

    # extract only 1-pixel border line of objects
    result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
    reference_border = reference ^ binary_erosion(reference, structure=footprint, iterations=1)

    # compute average surface distance
    # Note: scipys distance transform is calculated only inside the borders of the
    #       foreground objects, therefore the input has to be reversed
    dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
    sds = dt[result_border]

    return sds
