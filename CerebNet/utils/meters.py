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
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
import torch

from FastSurferCNN.utils import logging

from CerebNet.utils.metrics import DiceScore, hd, dice_score, volume_similarity
from CerebNet.utils.misc import plot_confusion_matrix, plot_predictions
from CerebNet.data_loader.data_utils import VERMIS_NAMES, GRAY_MATTER

logger = logging.get_logger(__name__)


class TestMeter:
    def __init__(self, classname_to_ids):

        # class_id: class_name
        self.classname_to_ids = classname_to_ids
        self.measure_func = lambda pred, gt: {
            "Dice": dice_score(pred, gt),
            "VS": volume_similarity(pred, gt),
        }

    def _compute_hd(self, pred_bin, gt_bin):
        hd_dict = {}
        if np.count_nonzero(pred_bin) == 0:
            hd_dict["HD_Max"] = np.nan
            hd_dict["HD95"] = np.nan
        else:
            hd_max, _, hd95 = hd(pred_bin, gt_bin)
            hd_dict["HD_Max"] = hd_max
            hd_dict["HD95"] = hd95

        return hd_dict

    def _get_binray_map(self, lbl_map, class_names):
        bin_map = np.logical_or.reduce(list(map(lambda l: lbl_map == l, class_names)))
        return bin_map

    def metrics_per_class(self, pred, gt):
        metrics = {"Label": [], "Dice": [], "HD95": [], "HD_Max": [], "VS": []}
        for lbl_name, lbl_id in self.classname_to_ids.items():
            # ignoring background
            if lbl_id == 0:
                continue

            metrics["Label"].append(lbl_name)
            if lbl_name == "Vermis":
                pred_bin_lbl = self._get_binray_map(pred, VERMIS_NAMES.values())
                gt_bin_lbl = self._get_binray_map(gt, VERMIS_NAMES.values())
            elif lbl_name == "L_Gray_Matter":
                pred_bin_lbl = self._get_binray_map(pred, GRAY_MATTER["Left"].values())
                gt_bin_lbl = self._get_binray_map(gt, GRAY_MATTER["Left"].values())
            elif lbl_name == "R_Gray_Matter":
                pred_bin_lbl = self._get_binray_map(pred, GRAY_MATTER["Right"].values())
                gt_bin_lbl = self._get_binray_map(gt, GRAY_MATTER["Right"].values())
            else:
                pred_bin_lbl = pred == lbl_id
                gt_bin_lbl = gt == lbl_id
            measures = self.measure_func(gt_bin_lbl, pred_bin_lbl)
            for key, val in measures.items():
                metrics[key].append(val)

            hd_dict = self._compute_hd(pred_bin_lbl, gt_bin_lbl)

            for key, val in hd_dict.items():
                metrics[key].append(val)

        return metrics


class Meter:
    def __init__(
        self,
        cfg,
        mode,
        global_step,
        total_iter=None,
        total_epoch=None,
        class_names=None,
        device=None,
        writer=None,
    ):
        self._cfg = cfg
        self.mode = mode.capitalize()
        self.confusion_mat = self.mode == "Val"
        self.class_names = class_names
        if self.class_names is None:
            self.class_names = [f"{c+1}" for c in range(cfg.MODEL.NUM_CLASSES)]

        self.dice_score = DiceScore(cfg.MODEL.NUM_CLASSES, device=device)
        self.batch_losses = {}
        self.writer = writer
        self.global_iter = global_step
        self.total_iter_num = total_iter
        self.total_epochs = total_epoch
        self.multi_gpu = cfg.NUM_GPUS > 1

    def reset(self):
        self.batch_losses = {}
        self.dice_score.reset()

    def update_stats(self, pred, labels, loss_dict=None):
        self.dice_score.update((pred, labels))
        if loss_dict is None:
            return
        for name, loss in loss_dict.items():
            self.batch_losses.setdefault(name, []).append(loss.item())

    def write_summary(self, loss_dict):
        if self.writer is None:
            return
        for name, loss in loss_dict.items():
            self.writer.add_scalar(f"{self.mode}/{name}", loss.item(), self.global_iter)
        self.global_iter += 1

    def prediction_visualize(
        self, cur_iter, cur_epoch, img_batch, label_batch, pred_batch
    ):
        if self.writer is None:
            return
        if cur_iter == 1:
            plt_title = "Results Epoch " + str(cur_epoch)
            _, batch_output = torch.max(pred_batch, dim=1)

            fig = plot_predictions(img_batch, label_batch, batch_output, plt_title)
            self.writer.add_figure("Val/Prediction", fig, cur_epoch)
            plt.close("all")

    def log_iter(self, cur_iter, cur_epoch):
        if (cur_iter + 1) % self._cfg.TRAIN.LOG_INTERVAL == 0:
            out_losses = {}
            for name, loss in self.batch_losses.items():
                out_losses[name] = np.around(np.array(loss).mean(), decimals=4)
            dice_score_per_class, confusion_mat = self.dice_score.compute(
                per_class=True
            )

            logger.info(
                "{} Epoch [{}/{}] Iter [{}/{}] [Dice Score: {:.4f}]  [{}]".format(
                    self.mode,
                    cur_epoch + 1,
                    self.total_epochs,
                    cur_iter + 1,
                    self.total_iter_num,
                    dice_score_per_class[1:].mean(),
                    out_losses,
                )
            )

    def log_lr(self, lr, step=None):
        if step is None:
            step = self.global_iter
        self.writer.add_scalar("Train/lr", lr[0], step)

    def log_epoch(self, cur_epoch):
        dice_score_per_class, confusion_mat = self.dice_score.compute(per_class=True)
        dice_score = dice_score_per_class[1:].mean()
        if self.writer is None:
            return dice_score
        self.writer.add_scalar(f"{self.mode}/mean_dice_score", dice_score, cur_epoch)

        if self.mode == "Val":
            if self.confusion_mat:
                fig = plot_confusion_matrix(confusion_mat, self.class_names)
                self.writer.add_figure(f"{self.mode}/confusion_mat", fig, cur_epoch)
                plt.close("all")
                plt.close(fig)

        logger.info(
            f"{self.mode} epoch {cur_epoch + 1} ended with [Mean Dice Score: {dice_score:.4f}]"
        )
        return dice_score
