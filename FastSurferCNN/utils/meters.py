
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
import matplotlib.pyplot as plt

from FastSurferCNN.utils import logging
from FastSurferCNN.utils.metrics import DiceScore
from FastSurferCNN.utils.misc import plot_confusion_matrix


logger = logging.getLogger(__name__)


class Meter:
    def __init__(self,
                 cfg,
                 mode,
                 global_step,
                 total_iter=None,
                 total_epoch=None,
                 class_names=None,
                 device=None,
                 writer=None):
        self._cfg = cfg
        self.mode = mode.capitalize()
        self.confusion_mat = False
        self.class_names = class_names
        if self.class_names is None:
            self.class_names = [f"{c+1}" for c in range(cfg.MODEL.NUM_CLASSES)]

        self.dice_score = DiceScore(cfg.MODEL.NUM_CLASSES, device=device)
        self.batch_losses = []
        self.writer = writer
        self.global_iter = global_step
        self.total_iter_num = total_iter
        self.total_epochs = total_epoch

    def reset(self):
        self.batch_losses = []
        self.dice_score.reset()

    def enable_confusion_mat(self):
        self.confusion_mat = True

    def disable_confusion_mat(self):
        self.confusion_mat = False

    def update_stats(self, pred, labels, batch_loss):
        self.dice_score.update((pred, labels), self.confusion_mat)
        self.batch_losses.append(batch_loss.item())

    def write_summary(self, loss_total, lr=None, loss_ce=None, loss_dice=None):
        self.writer.add_scalar(f"{self.mode}/total_loss", loss_total.item(), self.global_iter)
        if self.mode == 'Train':
            self.writer.add_scalar("Train/lr", lr[0], self.global_iter)
            if loss_ce:
                self.writer.add_scalar("Train/ce_loss", loss_ce.item(), self.global_iter)
            if loss_dice:
                self.writer.add_scalar("Train/dice_loss", loss_dice.item(), self.global_iter)

        self.global_iter += 1

    def log_iter(self, cur_iter, cur_epoch):
        if (cur_iter+1) % self._cfg.TRAIN.LOG_INTERVAL == 0:
            logger.info("{} Epoch [{}/{}] Iter [{}/{}] with loss {:.4f}".format(self.mode,
                cur_epoch + 1, self.total_epochs,
                cur_iter + 1, self.total_iter_num,
                np.array(self.batch_losses).mean()
            ))

    def log_epoch(self, cur_epoch):
        dice_score = self.dice_score.compute_dsc()
        self.writer.add_scalar(f"{self.mode}/mean_dice_score", dice_score, cur_epoch)
        if self.confusion_mat:
            confusion_mat = self.dice_score.comput_dice_cnf()
            fig = plot_confusion_matrix(confusion_mat, self.class_names)
            self.writer.add_figure(f"{self.mode}/confusion_mat", fig, cur_epoch)
            plt.close('all')