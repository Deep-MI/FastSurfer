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
import math
import numbers
import typing as _T

import torch
import torch.optim.lr_scheduler as scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

from FastSurferCNN.utils import logging

logger = logging.get_logger(__name__)


class ReduceLROnPlateauWithRestarts(ReduceLROnPlateau):
    """
    Extends the ReduceLROnPlateau class with the restart ability.
    """
    def __init__(self, optimizer, *args, T_0=10, Tmult=1, lr_restart=None, **kwargs):
        """
        Create a ReduceLROnPlateauWithRestarts learning rate scheduler.

        Restarts the learning rate scheduler after T_i epochs, where T_i = T_{i-1} * Tmult.

        At restart, the learning rate gets reset to the initial learning rate. If lr_restart is set
        and a number, it is reset to initial lr * (lr_restart) ^ i, if lr_restart is a function,
        the lr gets reset to lr_restart(initial_lr, i).

        Args:
            ...: same as ReduceLROnPlateau
            T_0 (optional): number of epochs until first restart (default: 10)
            Tmult (optional): multiplicative factor for future restarts (default: 1)
            lr_restart (optinoal): multiplicative factor for learning rate adjustment at restart.
        """
        # from torch.optim.lr_scheduler._LRSchduler
        # if last_epoch == -1:
        for group in optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])
        # else:
        # for i, group in enumerate(optimizer.param_groups):
        # if 'initial_lr' not in group:
        # raise KeyError("param 'initial_lr' is not specified "
        # "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(
            map(lambda group: group["initial_lr"], optimizer.param_groups)
        )

        super().__init__(optimizer, *args, **kwargs)
        self.T_0 = T_0
        self.Tmult = Tmult
        self.lr_restart = lr_restart
        self.T_i = T_0
        self.Tcur = -1
        self.i = 0
        if self.lr_restart is None:
            self.lr_restart = 1

    def step(self, metrics, epoch=None):
        """
        Performs an optimization step.

        Parameters
        ----------
        metrics : float
            The value of metrics is used to determine learning rate adjustments.
        epoch : int, default=None
            Number of epochs.
        
        Notes
        -----
        For details, refer to the PyTorch documentation for `ReduceLROnPlateau` at:
        https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html
        """
        self.Tcur += 1
        super().step(metrics, epoch)
        if self.Tcur >= self.T_i:
            self._reset_lr()
        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    def _reset_lr(self):
        """
        Internal method to reset the learning rate.
        """
        self.Tcur = 0
        self.T_i *= self.Tmult
        self.i += 1

        # reset the best numbers
        self._reset()

        for i, param_group in enumerate(self.optimizer.param_groups):
            # old_lr = float(param_group["lr"])
            lr_r = (
                self.lr_restart[i]
                if isinstance(self.lr_restart, _T.Sequence)
                else self.lr_restart
            )
            if isinstance(lr_r, numbers.Number):
                new_lr = param_group["initial_lr"] * lr_r**i
            else:
                new_lr = lr_r(param_group["initial_lr"], self.i)
            new_lr = self.min_lrs[i] if new_lr < self.min_lrs[i] else new_lr
            param_group["lr"] = new_lr
            if self.verbose:
                logger.info(
                    f"Epoch {self.last_epoch:5d}: restarting learning rate with "
                    f"{new_lr:.4e} for group {i}."
                )


# https://detectron2.readthedocs.io/_modules/detectron2/solver/lr_scheduler.html
class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Learning Rate scheduler that combines a cosine schedule with a warmup phase.
    """
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_iters: int,
        warmup_factor: float = 0.001,
        warmup_iters: int = 1000,
        warmup_method: str = "linear",
        last_epoch: int = -1,
    ):
        self.max_iters = max_iters
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        """
        Get the learning rates at the current epoch.
        """
        warmup_factor = _get_warmup_factor_at_iter(
            self.warmup_method, self.last_epoch, self.warmup_iters, self.warmup_factor
        )
        # Different definitions of half-cosine with warmup are possible. For
        # simplicity we multiply the standard half-cosine schedule by the warmup
        # factor. An alternative is to start the period of the cosine at warmup_iters
        # instead of at 0. In the case that warmup_iters << max_iters the two are
        # very close to each other.
        return [
            base_lr
            * warmup_factor
            * 0.5
            * (1.0 + math.cos(math.pi * self.last_epoch / self.max_iters))
            for base_lr in self.base_lrs
        ]

    def _compute_values(self) -> list[float]:
        # The new interface
        return self.get_lr()


def _get_warmup_factor_at_iter(
    method: str, iter: int, warmup_iters: int, warmup_factor: float
) -> float:
    """
    Return the learning rate warmup factor at a specific iteration.
    See :paper:`in1k1h` for more details.

    Args:
        method (str): warmup method; either "constant" or "linear".
        iter (int): iteration at which to calculate the warmup factor.
        warmup_iters (int): the number of warmup iterations.
        warmup_factor (float): the base warmup factor (the meaning changes according
            to the method used).

    Returns:
        float: the effective warmup factor at the given iteration.
    """
    if iter >= warmup_iters:
        return 1.0

    if method == "constant":
        return warmup_factor
    elif method == "linear":
        alpha = iter / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    else:
        raise ValueError(f"Unknown warmup method: {method}")


class CosineLR:
    """
    Learning rate scheduler that follows a Cosine trajectory.
    """
    def __init__(self, base_lr, eta_min, max_epoch):
        self.base_lr = base_lr
        self.max_epoch = max_epoch
        self.eta_min = eta_min

    def lr_func_cosine(self, cur_epoch):
        """
        Get the learning rate following a cosine pattern for the epoch `cur_epoch`.

        Parameters
        ----------
        cur_epoch : int
            The number of epoch of the current training stage.
        """
        return self.eta_min + (
            (self.base_lr - self.eta_min)
            * (math.cos(math.pi * cur_epoch / self.max_epoch) + 1.0)
            * 0.5
        )

    def set_lr(self, optimizer, epoch):
        """
        Sets the optimizer lr to the specified value.
        
        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            The optimizer using to optimize the current network.
        epoch : int
            The epoch for which to update the learning rate.
        """
        new_lr = self.get_epoch_lr(epoch)
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr

    def get_epoch_lr(self, cur_epoch):
        """
        Retrieves the lr for the given epoch (as specified by the lr policy).

        Parameters
        ----------
        cur_epoch : int
            The number of epoch of the current training stage.
        """
        return self.lr_func_cosine(cur_epoch)


class CosineAnnealingWarmRestartsDecay(scheduler.CosineAnnealingWarmRestarts):
    """
    Learning rate scheduler that combines a Cosine annealing with warm restarts pattern, but also adds a 
    decay factor for where the learning rate restarts at. 
    """
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1):
        super().__init__(
            optimizer, T_0, T_mult=T_mult, eta_min=eta_min, last_epoch=last_epoch
        )
        pass

    def decay_base_lr(self, curr_iter, n_epochs, n_iter):
        """
        Learning rate scheduler that combines a Cosine annealing with warm restarts pattern, 
        but also adds a decay factor for where the learning rate restarts at. 
        """
        if self.T_cur + 1 == self.T_i:
            annealed_lrs = []
            for base_lr in self.base_lrs:
                annealed_lr = (
                    base_lr
                    * (1 + math.cos(math.pi * curr_iter / (n_epochs * n_iter)))
                    / 2
                )
                annealed_lrs.append(annealed_lr)
            self.base_lrs = annealed_lrs


def get_lr_scheduler(optimizer, cfg):
    """
    Build a learning rate scheduler object from the config data in cfg.
    """
    scheduler_type = cfg.OPTIMIZER.LR_SCHEDULER
    if scheduler_type == "step_lr":
        return scheduler.StepLR(
            optimizer=optimizer,
            step_size=cfg.OPTIMIZER.STEP_SIZE,
            gamma=cfg.OPTIMIZER.GAMMA,
        )
    elif scheduler_type == "cosineWarmRestarts":
        return scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=cfg.OPTIMIZER.T_ZERO,
            T_mult=cfg.OPTIMIZER.T_MULT,
            eta_min=cfg.OPTIMIZER.ETA_MIN,
        )
    elif scheduler_type == "reduceOnPlateau":
        return scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode="max",
            patience=cfg.OPTIMIZER.PATIENCE,
            verbose=True,
            factor=cfg.OPTIMIZER.GAMMA,
            threshold=0.001,
            threshold_mode="rel",
            min_lr=cfg.OPTIMIZER.ETA_MIN,
        )
    elif scheduler_type == "reduceOnPlateauRestart":
        return ReduceLROnPlateauWithRestarts(
            optimizer=optimizer,
            mode="max",
            patience=cfg.OPTIMIZER.PATIENCE,
            verbose=True,
            factor=cfg.OPTIMIZER.GAMMA,
            threshold=0.005,
            threshold_mode="rel",
            cooldown=2,
            min_lr=cfg.OPTIMIZER.ETA_MIN,
            eps=1e-08,
            T_0=10,
            Tmult=2,
            lr_restart=0.2,
        )
    elif scheduler_type == "multiStep":
        return scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=cfg.OPTIMIZER.MILESTONES,
            gamma=cfg.OPTIMIZER.GAMMA,
        )
    else:
        raise ValueError(f"{scheduler_type} lr scheduler is not supported ")
