#!/usr/bin/env python3

import argparse

import nibabel as nib
import numpy as np


WM_AND_HYPO_LABELS = (2, 41, 77, 78, 79, 87, 88, 89)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge asynchronously generated wmparc labels into aparc+aseg."
    )
    parser.add_argument("--aseg", required=True, help="aseg.mgz used as wmparc input")
    parser.add_argument(
        "--aparc",
        required=True,
        help="aparc+aseg volume carrying the final cortical labels",
    )
    parser.add_argument(
        "--wmparc",
        required=True,
        help="wmparc volume generated from aseg.mgz with --label-wm",
    )
    parser.add_argument("--out", required=True, help="merged wmparc output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    aseg_img = nib.load(args.aseg)
    aparc_img = nib.load(args.aparc)
    wmparc_img = nib.load(args.wmparc)

    aseg = np.asanyarray(aseg_img.dataobj)
    aparc = np.asanyarray(aparc_img.dataobj).copy()
    wmparc = np.asanyarray(wmparc_img.dataobj)

    wm_mask = np.isin(aseg, WM_AND_HYPO_LABELS)
    aparc[wm_mask] = wmparc[wm_mask]

    out_img = nib.MGHImage(aparc, aparc_img.affine, aparc_img.header)
    nib.save(out_img, args.out)


if __name__ == "__main__":
    main()
