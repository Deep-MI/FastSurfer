
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
import torch.optim.lr_scheduler as scheduler


def get_lr_scheduler(optimzer, cfg):
    scheduler_type = cfg.OPTIMIZER.LR_SCHEDULER
    if scheduler_type == 'step_lr':
        return scheduler.StepLR(
            optimizer=optimzer,
            step_size=cfg.OPTIMIZER.STEP_SIZE,
            gamma=cfg.OPTIMIZER.GAMMA,
        )
    elif scheduler_type == 'cosineWarmRestarts':
        return scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimzer,
            T_0=cfg.OPTIMIZER.T_ZERO,
            T_mult=cfg.OPTIMIZER.T_MULT,
            eta_min=cfg.OPTIMIZER.ETA_MIN,
        )
    elif scheduler_type == "NoScheduler" or scheduler_type is None:
        return None
    else:
        raise ValueError(f"{scheduler_type} lr scheduler is not supported ")