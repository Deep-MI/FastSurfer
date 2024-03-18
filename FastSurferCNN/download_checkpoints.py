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

from FastSurferCNN.utils.checkpoint import (
    check_and_download_ckpts,
    get_checkpoints,
    get_plane_default,
    YAML_DEFAULT as VINN_YAML,
    )

from CerebNet.utils.checkpoint import (
    YAML_DEFAULT as CEREBNET_YAML)


if __name__ == "__main__":
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
        "--url",
        type=str,
        default=None,
        help="Specify you own base URL. This is applied to all models. \n"
        "Default for VINN: {} \n"
        "Default for CerebNet: {}".format(get_plane_default('URL', None, filename=VINN_YAML), get_plane_default('URL', None, filename=CEREBNET_YAML)),
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Checkpoint file paths to download, e.g. checkpoints/aparc_vinn_axial_v2.0.0.pkl ...",
    )
    args = parser.parse_args()

    if not args.vinn and not args.files and not args.cerebnet and not args.all:
        print(
            "Specify either files to download or --vinn, --cerebnet or --all, see help -h."
        )
        exit(1)

    # FastSurferVINN checkpoints
    if args.vinn or args.all:
        get_checkpoints(
            get_plane_default('CKPT', 'axial', filename=VINN_YAML),
            get_plane_default('CKPT', 'coronal', filename=VINN_YAML),
            get_plane_default('CKPT', 'sagittal', filename=VINN_YAML),
            urls=get_plane_default('URL', None, filename=VINN_YAML) if args.url is None else [args.url]
        )

    # CerebNet checkpoints
    if args.cerebnet or args.all:
        get_checkpoints(
            get_plane_default('CKPT', 'axial', filename=CEREBNET_YAML),
            get_plane_default('CKPT', 'coronal', filename=CEREBNET_YAML),
            get_plane_default('CKPT', 'sagittal', filename=CEREBNET_YAML),
            urls=get_plane_default('URL', None, filename=CEREBNET_YAML) if args.url is None else [args.url],
        )

    # later we can add more defaults here (for other sub-segmentation networks, or old CNN)
    for fname in args.files:
        check_and_download_ckpts(fname, get_plane_default('URL', None, filename=VINN_YAML) if args.url is None else [args.url])
