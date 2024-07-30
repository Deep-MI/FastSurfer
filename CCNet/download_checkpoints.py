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

from FastSurferCNN.utils.checkpoint import check_and_download_ckpts, get_checkpoints, VINN_AXI, VINN_COR, VINN_SAG, URL


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Check and Download Network Checkpoints')
    parser.add_argument("--all", default=False, action="store_true",
                        help="Check and download all default checkpoints")
    parser.add_argument("--vinn", default=False, action="store_true",
                        help="Check and download VINN default checkpoints")
    parser.add_argument("--url", type=str, default=URL,
                        help="Specify you own base URL. Default: {}".format(URL))
    parser.add_argument('files', nargs='*',
                        help="Checkpoint file paths to download, e.g. checkpoints/aparc_vinn_axial_v2.0.0.pkl ...")
    args = parser.parse_args()

    # download all sets of weights:
    if args.all:
        args.vinn = True

    if not args.vinn and not args.files:
        print("Specify either files to download or --vinn, see help -h.")
        exit(1)

    if args.vinn or args.all:
        get_checkpoints(VINN_AXI, VINN_COR, VINN_SAG, args.url)

    # later we can add more defaults here (for other sub-segmentation networks, or old CNN)

    for fname in args.files:
        check_and_download_ckpts(fname, args.url)
