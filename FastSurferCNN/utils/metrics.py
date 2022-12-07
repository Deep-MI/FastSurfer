
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
import torch
import numpy as np

from FastSurferCNN.utils import logging

logger = logging.getLogger(__name__)


def iou_score(pred_cls, true_cls, nclass=79):
    """
    compute the intersection-over-union score
    both inputs should be categorical (as opposed to one-hot)
    """
    intersect_ = []
    union_ = []

    for i in range(1, nclass):
        intersect = ((pred_cls == i).float() + (true_cls == i).float()).eq(2).sum().item()
        union = ((pred_cls == i).float() + (true_cls == i).float()).ge(1).sum().item()
        intersect_.append(intersect)
        union_.append(union)

    return np.array(intersect_), np.array(union_)


def precision_recall(pred_cls, true_cls, nclass=79):
    """
    Function to calculate recall (TP/(TP + FN) and precision (TP/(TP+FP) per class
    :param pytorch.Tensor pred_cls: network prediction (categorical)
    :param pytorch.Tensor true_cls: ground truth (categorical)
    :param int nclass: number of classes
    :return:
    """
    tpos_fneg = []
    tpos_fpos = []
    tpos = []

    for i in range(1, nclass):
        all_pred = (pred_cls == i).float()
        all_gt = (true_cls == i).float()

        tpos.append((all_pred + all_gt).eq(2).sum().item())
        tpos_fpos.append(all_pred.sum().item())
        tpos_fneg.append(all_gt.sum().item())

    return np.array(tpos), np.array(tpos_fneg), np.array(tpos_fpos)


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

    def __init__(self, num_classes,
                 device=None,
                 output_transform=lambda y_pred, y: (y_pred.data.max(1)[1], y)):
        self._device = device
        self.out_transform = output_transform
        self.n_classes = num_classes
        self.union = torch.zeros(self.n_classes, self.n_classes, device=device)
        self.intersection = torch.zeros(self.n_classes, self.n_classes, device=device)

    def reset(self):
        self.union = torch.zeros(self.n_classes, self.n_classes, device=self._device)
        self.intersection = torch.zeros(self.n_classes, self.n_classes, device=self._device)

    def _check_output_type(self, output):
        if not (isinstance(output, tuple)):
            raise TypeError("Output should a tuple consist of of torch.Tensors, but given {}".format(type(output)))

    def _update_union_intersection_matrix(self, batch_output, labels_batch):
        for i in range(self.n_classes):
            gt = (labels_batch == i).float()
            for j in range(self.n_classes):
                pred = (batch_output == j).float()
                self.intersection[i, j] += torch.sum(torch.mul(gt, pred))
                self.union[i, j] += (torch.sum(gt) + torch.sum(pred))

    def _update_union_intersection(self, batch_output, labels_batch):
        for i in range(self.n_classes):
            gt = (labels_batch == i).float()
            pred = (batch_output == i).float()
            self.intersection[i, i] += torch.sum(torch.mul(gt, pred))
            self.union[i, i] += (torch.sum(gt) + torch.sum(pred))

    def update(self, output, cnf_mat):
        self._check_output_type(output)

        if self._device is not None:
            # Put output to the metric's device
            if isinstance(output, torch.Tensor) and (output.device != self._device):
                output = output.to(self._device)
        y_pred, y = self.out_transform(*output)

        if cnf_mat:
            self._update_union_intersection_matrix(y_pred, y)
        else:
            self._update_union_intersection(y_pred, y)

    def compute_dsc(self):
        dsc_per_class = self._dice_calculation()
        dsc = dsc_per_class.mean()
        return dsc

    def comput_dice_cnf(self):
        dice_cm_mat = self._dice_confusion_matrix()
        return dice_cm_mat

    def _dice_calculation(self):
        intersection = self.intersection.diagonal()
        union = self.union.diagonal()
        dsc = 2 * torch.div(intersection, union)
        return dsc

    def _dice_confusion_matrix(self):
        dice_intersection = self.intersection.cpu().numpy()
        dice_union = self.union.cpu().numpy()
        if not (dice_union > 0).all():
            logger.info("Union of some classes are all zero")
        dice_cnf_matrix = 2 * np.divide(dice_intersection, dice_union)
        return dice_cnf_matrix


def dice_score(cm):
    pass