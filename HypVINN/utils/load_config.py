# Copyright 2024 AI in Medical Imaging, German Center for Neurodegenerative Diseases(DZNE), Bonn
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

from os.path import join, split, splitext

from HypVINN.config.hypvinn import get_cfg_hypvinn


def get_config(args):
    """
    Given the arguments, load and initialize the configs.

    Parameters
    ----------
    args : object
        The arguments object.
    Returns
    -------
    cfg : yacs.config.CfgNode
        The configuration node.
    """
    # Setup cfg.
    cfg = get_cfg_hypvinn()
    # Load config from cfg.
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)
    # Load config from command line, overwrite config from opts.
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    if hasattr(args, "rng_seed"):
        cfg.RNG_SEED = args.rng_seed
    if hasattr(args, "output_dir"):
        cfg.LOG_DIR = args.LOG_dir

    cfg_file_name = splitext(split(args.cfg_file)[1])[0]
    cfg.LOG_DIR = join(cfg.LOG_DIR, cfg_file_name)

    return cfg

def load_config(cfg_file):
    """
    Load and initialize the configuration from a given file.

    Parameters
    ----------
    cfg_file : str
        The path to the configuration file. The function will load configurations from this file.

    Returns
    -------
    cfg : yacs.config.CfgNode
        The configuration node, loaded and initialized with the given file.
    """
    # setup base
    cfg = get_cfg_hypvinn()
    cfg.EXPR_NUM = None
    cfg.SUMMARY_PATH = ""
    cfg.CONFIG_LOG_PATH = ""
    cfg.TRAIN.RESUME_EXPR_NUM = None
    # Overwrite with stored arguments
    cfg.merge_from_file(cfg_file)
    return cfg