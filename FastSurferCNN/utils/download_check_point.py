#!/usr/bin/env python3

import argparse

from utils.checkpoint import download_checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # TODO: Add default URL
    parser.add_argument('--download_url', type=str, default="",
                        help="URL for downloading checkpoints. Default: ")
    parser.add_argument('--checkpoint_name', type=str, required=True,
                        help="Name of the checkpoint to be downloaded")
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help="Path to save the downloaded checkpoint file in")

    args = parser.parse_args()

    download_checkpoint(args.download_url, args.checkpoint_name, args.checkpoint_path)
