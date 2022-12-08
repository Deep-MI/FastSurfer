
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
from os.path import join
import sys
import argparse
import json

from FastSurferCNN.utils import misc
from FastSurferCNN.utils.load_config import get_config
from FastSurferCNN.train import Trainer


def setup_options():
    # Training settings
    parser = argparse.ArgumentParser(description='Segmentation')

    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path to the config file",
        default="config/FastSurferVINN.yaml",
        type=str,
    )
    parser.add_argument("--aug", action='append', help="List of augmentations to use.", default=None)

    parser.add_argument("--opt", action='append', help="List of augmentations to use.")

    parser.add_argument(
        "opts",
        help="See FastSurferCNN/config/defaults.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )

    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()


def main():
    args = setup_options()
    cfg = get_config(args)

    if args.aug is not None:
        cfg.DATA.AUG = args.aug

    if args.opt:
        cfg.DATA.CLASS_OPTIONS = args.opt

    summary_path = misc.check_path(join(cfg.LOG_DIR, 'summary'))
    if cfg.EXPR_NUM == "Default":
        cfg.EXPR_NUM = str(misc.find_latest_experiment(join(cfg.LOG_DIR, 'summary')) + 1)

    if cfg.TRAIN.RESUME and cfg.TRAIN.RESUME_EXPR_NUM != "Default":
        cfg.EXPR_NUM = cfg.TRAIN.RESUME_EXPR_NUM

    cfg.SUMMARY_PATH = misc.check_path(join(summary_path, '{}'.format(cfg.EXPR_NUM)))
    cfg.CONFIG_LOG_PATH = misc.check_path(join(cfg.LOG_DIR, "config", '{}'.format(cfg.EXPR_NUM)))

    with open(join(cfg.CONFIG_LOG_PATH, "config.yaml"), "w") as json_file:
        json.dump(cfg, json_file, indent=2)

    trainer = Trainer(cfg=cfg)
    trainer.run()


if __name__ == '__main__':
    main()
