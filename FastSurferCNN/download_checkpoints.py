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

from functools import lru_cache
from itertools import chain
from pathlib import Path

from FastSurferCNN.utils.checkpoint import (
    check_and_download_ckpts,
    get_checkpoints,
    get_config_file,
    load_checkpoint_config_defaults,
)
from FastSurferCNN.utils.parallel import thread_executor


class ConfigCache:
    @classmethod
    @lru_cache
    def url(cls, module: str) -> list[str]:
        return load_checkpoint_config_defaults("url", get_config_file(module))

    @classmethod
    @lru_cache
    def checkpoint(cls, module: str) -> dict[str, Path]:
        return load_checkpoint_config_defaults("checkpoint", get_config_file(module))

    @classmethod
    def all_urls(cls) -> list[str]:
        return list(chain(*(cls.url(mod) for mod in ("FastSurferCNN", "CorpusCallosum", "CerebNet", "HypVINN"))))


defaults = ConfigCache()


def make_parser():
    import argparse

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
        "--cc",
        default=False,
        action="store_true",
        help="Check and download Corpus Callosum default checkpoints",
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
             f"Default for VINN: {defaults.url('FastSurferCNN')} \n" + \
             "\n".join(f"Default for {mod}: {defaults.url(mod)}" for mod in ("CerebNet", "CorpusCallosum", "HypVINN")),
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Checkpoint file paths to download, e.g. "
             "checkpoints/aparc_vinn_axial_v2.0.0.pkl ...",
    )
    return parser


def main(
        vinn: bool = False,
        cerebnet: bool = False,
        hypvinn: bool = False,
        cc: bool = False,
        all: bool = False,
        files: list[str] = (),
        url: str | None = None,
) -> int | str:
    if not vinn and not files and not cerebnet and not hypvinn and not cc and not all:
        return "Specify either files to download or --vinn, --cerebnet, --cc, --hypvinn, or --all, see help -h."

    futures = []
    all_errors = []
    try:
        for mod, sel in (("FastSurferCNN", vinn), ("CerebNet", cerebnet), ("HypVINN", hypvinn), ("CorpusCallosum", cc)):
            if sel or all:
                urls = defaults.url(mod) if url is None else [url]
                futures.extend(thread_executor().submit(get_checkpoints, file, urls=urls)
                               for key, file in defaults.checkpoint(mod).items())
        for fname in files:
            check_and_download_ckpts(
                fname,
                urls=defaults.all_urls() if url is None else [url],
            )
    except Exception as e:
        from traceback import print_exception
        print_exception(e)
        all_errors = [e.args[0]]
    for f in futures:
        if e := f.exception():
            from traceback import print_exception
            print_exception(e)
            all_errors.append(f.exception().args[0])
    return "\n".join(all_errors) or 0


if __name__ == "__main__":
    import sys
    from logging import INFO, basicConfig

    basicConfig(stream=sys.stdout, level=INFO)
    parser = make_parser()
    args = parser.parse_args()
    sys.exit(main(**vars(args)))
