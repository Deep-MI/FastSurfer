#!/usr/bin/env python3

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

import argparse

from CerebNet.utils.checkpoint import (
    YAML_DEFAULT as CEREBNET_YAML,
)
from FastSurferCNN.utils import PLANES
from FastSurferCNN.utils.checkpoint import (
    YAML_DEFAULT as VINN_YAML,
)
from FastSurferCNN.utils.checkpoint import (
    check_and_download_ckpts,
    get_checkpoints,
    load_checkpoint_config_defaults,
)
from HypVINN.utils.checkpoint import (
    YAML_DEFAULT as HYPVINN_YAML,
)


class ConfigCache:
    def vinn_url(self):
        return load_checkpoint_config_defaults("url", filename=VINN_YAML)

    def cerebnet_url(self):
        return load_checkpoint_config_defaults("url", filename=CEREBNET_YAML)

    def hypvinn_url(self):
        return load_checkpoint_config_defaults("url", filename=HYPVINN_YAML)

    def all_urls(self):
        return self.vinn_url() + self.cerebnet_url() + self.hypvinn_url()


defaults = ConfigCache()


def make_arguments():
    parser = argparse.ArgumentParser(
        description="Check and Download Network Checkpoints"
    )
    parser.add_argument(
        "--all",
        default=False,
        action="store_true",
        help="Check and download all default checkpoints",
    )
    parser.add_argument(
        "--vinn",
        default=False,
        action="store_true",
        help="Check and download VINN default checkpoints",
    )
    parser.add_argument(
        "--cerebnet",
        default=False,
        action="store_true",
        help="Check and download CerebNet default checkpoints",
    )

    parser.add_argument(
        "--hypvinn",
        default=False,
        action="store_true",
        help="Check and download HypVinn default checkpoints",
    )

    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help=f"Specify you own base URL. This is applied to all models. \n"
             f"Default for VINN: {defaults.vinn_url()} \n"
             f"Default for CerebNet: {defaults.cerebnet_url()} \n"
             f"Default for HypVINN: {defaults.hypvinn_url()}",
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Checkpoint file paths to download, e.g. "
             "checkpoints/aparc_vinn_axial_v2.0.0.pkl ...",
    )
    return parser.parse_args()


def main(
        vinn: bool,
        cerebnet: bool,
        hypvinn: bool,
        all: bool,
        files: list[str],
        url: str | None = None,
) -> int | str:
    if not vinn and not files and not cerebnet and not hypvinn and not all:
        return ("Specify either files to download or --vinn, --cerebnet, "
                "--hypvinn, or --all, see help -h.")

    try:
        # FastSurferVINN checkpoints
        if vinn or all:
            vinn_config = load_checkpoint_config_defaults(
                "checkpoint",
                filename=VINN_YAML,
            )
            get_checkpoints(
                *(vinn_config[plane] for plane in PLANES),
                urls=defaults.vinn_url() if url is None else [url]
            )
        # CerebNet checkpoints
        if cerebnet or all:
            cerebnet_config = load_checkpoint_config_defaults(
                "checkpoint",
                filename=CEREBNET_YAML,
            )
            get_checkpoints(
                *(cerebnet_config[plane] for plane in PLANES),
                urls=defaults.cerebnet_url() if url is None else [url],
            )
        # HypVINN checkpoints
        if hypvinn or all:
            hypvinn_config = load_checkpoint_config_defaults(
                "checkpoint",
                filename=HYPVINN_YAML,
            )
            get_checkpoints(
                *(hypvinn_config[plane] for plane in PLANES),
                urls=defaults.hypvinn_url() if url is None else [url],
            )
        for fname in files:
            check_and_download_ckpts(
                fname,
                urls=defaults.all_urls() if url is None else [url],
            )
    except Exception as e:
        from traceback import print_exception
        print_exception(e)
        return e.args[0]
    return 0


if __name__ == "__main__":
    import sys
    from logging import INFO, basicConfig

    basicConfig(stream=sys.stdout, level=INFO)
    args = make_arguments()
    sys.exit(main(**vars(args)))
