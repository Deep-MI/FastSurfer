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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import yacs.config

from CerebNet.config import get_cfg_cerebnet
from FastSurferCNN.utils import PLANES


def get_config(args) -> "yacs.config.CfgNode":
    """
    Given the arguments, load and initialize the config_files.
    """
    # Setup cfg.
    cfg = get_cfg_cerebnet()
    # Load config from cfg.
    # Load config from command line, overwrite config from opts.
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    if hasattr(args, "rng_seed"):
        cfg.RNG_SEED = args.rng_seed
    if hasattr(args, "out_dir"):
        cfg.LOG_DIR = str(args.out_dir)

    path_ax, path_sag, path_cor = (
        getattr(args, name) for name in ["ckpt_ax", "ckpt_sag", "ckpt_cor"]
    )

    for plane, path in zip(PLANES, (path_ax, path_cor, path_sag), strict=False):
        setattr(cfg.TEST, f"{plane.upper()}_CHECKPOINT_PATH", str(path))

    # overwrite the batch size if it is passed as a parameter
    batch_size = getattr(args, "batch_size", None)
    if batch_size is not None:
        cfg.TEST.BATCH_SIZE = batch_size

    return cfg
