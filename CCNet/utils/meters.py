
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
import os

from ignite.metrics import Metric
import ignite
from matplotlib import axis
import numpy as np
import torch

from FastSurferCNN.utils import logging
from CCNet.utils.misc import calculate_centers_of_comissures, plot_confusion_matrix

from scipy.ndimage import _ni_support
from scipy.ndimage.morphology import (
    distance_transform_edt,
    binary_erosion,
    generate_binary_structure,
)


logger = logging.getLogger(__name__)


class DiceScore(Metric):
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

    def __init__(
        self,
        num_classes,
        class_ids=None,
        device=None,
        one_hot=False,
        output_transform=lambda y_pred, y: (y_pred.data.max(1)[1], y),
    ):
        self._device = device
        self.out_transform = output_transform
        self.class_ids = class_ids
        if self.class_ids is None:
            self.class_ids = np.arange(num_classes)
        self.n_classes = num_classes
        assert len(self.class_ids) == self.n_classes, (
            f"Number of class ids is not correct,"
            f" given {len(self.class_ids)} but {self.n_classes} is needed."
        )
        self.one_hot = one_hot
        self.reset()

    def reset(self):
        self.union = torch.zeros(self.n_classes, self.n_classes, device=self._device)
        self.intersection = torch.zeros(self.n_classes, self.n_classes, device=self._device)

    def _check_output_type(self, output):
        if not (isinstance(output, tuple)):
            raise TypeError(
                "Output should a tuple consist of of torch.Tensors, but given {}".format(
                    type(output)
                )
            )
        
    def _update_union_intersection(self, batch_output: torch.Tensor, labels_batch: torch.Tensor):
        """Update the union intersection.

        Parameters
        ----------
        batch_output : torch.Tensor
            batch output (prediction, labels)
        labels_batch : torch.Tensor

        """
        for i in range(self.n_classes):
            gt = (labels_batch == i).float()
            pred = (batch_output == i).float()
            self.intersection[i, i] += torch.sum(torch.mul(gt, pred))
            self.union[i, i] += torch.sum(gt) + torch.sum(pred)

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

        self.reset()
        if per_class:
            return dice_score_per_class #, dice_cm_mat
        else:
            return dice_score #, dice_cm_mat

    def _dice_confusion_matrix(self, class_idxs):
        dice_intersection = self.intersection.cpu().numpy()
        dice_union = self.union.cpu().numpy()
        if class_idxs is not None:
            dice_union = dice_union[class_idxs[:, None], class_idxs]
            dice_intersection = dice_intersection[class_idxs[:, None], class_idxs]
        if not (dice_union > 0).all():
            logger.info("Union of some classes are all zero")
        dice_cnf_matrix = 2 * np.divide(dice_intersection, dice_union)
        return dice_cnf_matrix

class LocDistance(Metric):
    """
    Calculates the Locational Distance between pred and reference.
    """

    def __init__(self, device=None, axis = 2):
        self._device = device
        self.axis = axis
        self.reset()

    def reset(self):
        self.loc_distance = 0.0

    def update(self, pred, reference):
        """
        Updates the Locational Distance metric with new predictions and reference.
        """
        if not isinstance(pred, torch.Tensor):
            pred = torch.from_numpy(pred)
        
        if not isinstance(reference, torch.Tensor):
            reference = torch.from_numpy(reference)

        if self._device is not None:
            reference = reference.to(self._device)
            pred = pred.to(self._device)
        
        self.loc_pred = pred[:,-1,:,:].detach()
        self.loc_ref = reference[:,reference.shape[1]//2:, :]

    def compute(self):
        """
        Computes the localication Distance metric.
        """
        self.reset()

        self.loc_distance = localisation_distance(self.loc_pred, self.loc_ref, axis=self.axis)
        return self.loc_distance
        


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
    return 1.0 - np.abs(pred_vol - gt_vol) / (pred_vol + gt_vol)


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
        raise RuntimeError(
            "The first supplied array does not contain any binary object."
        )
    if 0 == np.count_nonzero(reference):
        raise RuntimeError(
            "The second supplied array does not contain any binary object."
        )

    # extract only 1-pixel border line of objects
    result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
    reference_border = reference ^ binary_erosion(
        reference, structure=footprint, iterations=1
    )

    # compute average surface distance
    # Note: scipys distance transform is calculated only inside the borders of the
    #       foreground objects, therefore the input has to be reversed
    dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
    sds = dt[result_border]

    return sds

def localisation_distance(result, reference, axis=2):
    """
    Calculates the distances between the centers of the comissures of the predicted and reference volume

    Parameters
    ----------
    result : np.ndarray
        Predicted volume
    reference : np.ndarray
        Reference volume
    axis : int
        Axis along which the commisure centers are separated (default: 2)
    """

    assert result.shape == reference.shape, f'Localisation shapes do not match: {result.shape} vs {reference.shape}'
    
    batch_size = reference.shape[0]

    # calculate centers of comissures
    commissures_pred = []
    commissures_ref = []

    for i in range(batch_size):
        try:
            ac_pc_pred = calculate_centers_of_comissures(result[i, ...], axis)
            ac_pc_ref = calculate_centers_of_comissures(reference[i, ...], axis)

        except ValueError:
            logger.warn(f'No valid comissures found.')
            continue
        
        commissures_pred.append(ac_pc_pred)
        commissures_ref.append(ac_pc_ref)
    
    commissures_pred = np.array(commissures_pred, dtype=np.float32)
    commissures_ref = np.array(commissures_ref, dtype=np.float32)
        
    # catch case where all distances are None
    if len(commissures_pred) == 0:
        logger.warn(f'No valid distances found. Skipping.')
        return float('nan')

    # calculate distances
    distances = np.array([np.linalg.norm(commissures_pred - commissures_ref, axis=1)], dtype=np.float32)
    
    # calculate average distance
    avg_distance = distances[distances != None].flatten().mean()

    logger.debug(f'Average distance between predicted and reference points: {avg_distance}')

    if np.isnan(avg_distance):
        logger.warn(f'Average distance is NaN')
        return float('nan')

    return avg_distance

class Meter:
    def __init__(self,
                 cfg,
                 mode,
                 global_step,
                 total_iter=None,
                 total_epoch=None,
                 class_names=None,
                 #device=None,
                 writer=None,
                 ignite_metrics={}):
        self._cfg = cfg
        self.mode = mode.capitalize()
        self.class_names = class_names
        if self.class_names is None:
            self.class_names = [f"{c+1}" for c in range(cfg.MODEL.NUM_CLASSES)]

        self.batch_losses = []
        self.writer = writer
        self.global_iter = global_step
        self.total_iter_num = total_iter
        self.total_epochs = total_epoch

        self.ignite_metrics = ignite_metrics

        self.save_path = os.path.join(cfg.LOG_DIR, "pred", str(cfg.EXPR_NUM))

    def reset(self):
        self.batch_losses = []
        for n, metric in self.ignite_metrics.items():
            metric.reset()

    def add_ignitemetric(self, name, ignitemetric):
        self.ignite_metrics[name] = ignitemetric


    def update_stats(self, pred, labels, batch_loss):
        # self.dice_score.update((pred, labels))
        self.batch_losses.append(batch_loss.item())

    def write_summary(self, loss_total, lr=None, loss_ce=None, loss_dice=None, loss_seg = None, loc_loss=None, dist_loss=None):
        self.global_iter += 1
        if self.writer is None:
            raise ValueError("Writer is None. Cannot write summary.")

        # add standart metrics
        self.writer.add_scalar(f"{self.mode}/total_loss", loss_total.item(), self.global_iter)

        
        # add metrics only for training
        if self.mode == 'Train':
            if lr:
                self.writer.add_scalar("Train/lr", lr[0], self.global_iter)
            if loss_ce:
                self.writer.add_scalar("Train/ce_loss", loss_ce.item(), self.global_iter)
            if loss_dice:
                self.writer.add_scalar("Train/dice_loss", loss_dice.item(), self.global_iter)
            if loc_loss:
                self.writer.add_scalar("Train/loc_loss", loc_loss, self.global_iter)
            if dist_loss:
                self.writer.add_scalar("Train/dist_loss", dist_loss, self.global_iter)
            if loss_seg:
                self.writer.add_scalar("Train/seg_loss", loss_seg, self.global_iter)
        

    def log_iter(self, cur_iter, cur_epoch):
        if (cur_iter+1) % self._cfg.TRAIN.LOG_INTERVAL == 0:
            logger.info("{} Epoch [{}/{}] Iter [{}/{}] with loss {:.4f}".format(self.mode,
                cur_epoch + 1, self.total_epochs,
                cur_iter + 1, self.total_iter_num,
                np.array(self.batch_losses).mean()
            ))

    def log_epoch(self, cur_epoch, runtime=None):
        if self.writer is None:
            raise ValueError("Writer is None. Cannot log the epoch.")
        
        if runtime is not None:
            self.writer.add_scalar(f"{self.mode}/runtime_per_epoch", runtime, cur_epoch)

        for metric_name in self.ignite_metrics.keys():
            if metric_name == 'confusion_matrix':
                confusion_mat = self.ignite_metrics['confusion_matrix'].compute()
                # file_save_name = os.path.join(self.save_path, 'Epoch_' + str(cur_epoch) + f'_{self.mode}_ConfusionMatrix.pdf')
                # plot_confusion_matrix(cm=confusion_mat, file_save_name=file_save_name, classes=self.class_names)
                # #self.writer.add_figure(f"{self.mode}/confusion_mat", fig, cur_epoch)
                # #plt.close('all')
                continue

            if metric_name == 'locational_distance':
                loc_distance = self.ignite_metrics['locational_distance'].compute()
                self.writer.add_scalar(f"{self.mode}/locational_distance", loc_distance, cur_epoch)
                continue

            metric = self.ignite_metrics[metric_name]

            try:
                log_output = metric.compute()
            except ignite.exceptions.NotComputableError:
                logger.warn(f"{metric_name} is not computable. Skipping.")
                continue

            # todo: Change to different types of float or check length
            if isinstance(log_output, float):
                self.writer.add_scalar(f"{self.mode}/{metric_name}", log_output, cur_epoch)
            else:
                print(f"WARNING: {metric_name} has more than one value. Only the mean is logged.")
                self.writer.add_scalar(f"{self.mode}/{metric_name}", log_output.mean(), cur_epoch)

            
            
