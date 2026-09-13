# Copyright 2026 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
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
import argparse
import os
import sys
from pathlib import Path

import nibabel as nib
import nibabel.freesurfer.io as fsio
import numpy as np

HELPTEXT = """
Label each voxel by its position with respect to the white and pial surfaces,
writing the cortical ribbon volumes. This replaces FreeSurfer's mris_volmask.

Inside and outside are decided with a generalized winding number, which is a
solid angle integral evaluated per point rather than a fill. Coincident sheets,
self-intersections and small holes therefore change its value smoothly instead
of letting a region escape, which matters because white and pial coincide
exactly along the medial wall.
"""


def make_parser() -> argparse.ArgumentParser:
    """
    Create a command line interface and return command line options.

    Returns
    -------
    argparse.ArgumentParser
        The parser object.
    """
    parser = argparse.ArgumentParser(
        description=HELPTEXT,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--sd", type=Path, required=True, help="subjects directory")
    parser.add_argument("--sid", type=str, required=True, help="subject id")
    parser.add_argument(
        "--aseg_name",
        type=str,
        default="aseg.presurf",
        help="volume in mri/ supplying the output geometry, without the .mgz suffix (default: aseg.presurf)",
    )
    parser.add_argument("--surf_white", type=str, default="white", help="white surface name")
    parser.add_argument("--surf_pial", type=str, default="pial", help="pial surface name")
    parser.add_argument(
        "--out_root",
        type=str,
        default="ribbon",
        help="output basename, written as <out_root>.mgz and <hemi>.<out_root>.mgz",
    )
    parser.add_argument("--label_left_white", type=int, default=2)
    parser.add_argument("--label_left_ribbon", type=int, default=3)
    parser.add_argument("--label_right_white", type=int, default=41)
    parser.add_argument("--label_right_ribbon", type=int, default=42)
    parser.add_argument("--lh-only", action="store_true", dest="lh_only", help="only left hemisphere")
    parser.add_argument("--rh-only", action="store_true", dest="rh_only", help="only right hemisphere")
    parser.add_argument(
        "--margin",
        type=float,
        default=2.0,
        help="margin in mm added around each surface's bounding box (default: 2.0)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="number of threads, 0 for one per core (default: 1)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s, part of FastSurfer, see 'run_fastsurfer.sh --version'",
    )
    return parser


def inside_surface(
    surf_file: Path,
    ref: nib.freesurfer.mghformat.MGHImage,
    margin: float,
) -> np.ndarray:
    """
    Mark the voxels of the reference grid that lie inside a closed surface.

    The winding number is only evaluated inside the surface's bounding box; a
    point outside that box cannot be inside the surface.

    Parameters
    ----------
    surf_file : Path
        FreeSurfer surface to test against.
    ref : nibabel.freesurfer.mghformat.MGHImage
        Volume supplying the output grid.
    margin : float
        Margin in mm added around the bounding box.

    Returns
    -------
    numpy.ndarray
        Boolean volume of the reference shape, True inside the surface.
    """
    import igl

    vertices, faces = fsio.read_geometry(surf_file)
    # surfaces are in tkrRAS, the grid is in voxels
    to_vox = np.linalg.inv(ref.header.get_vox2ras_tkr())
    verts = np.ascontiguousarray(nib.affines.apply_affine(to_vox, vertices), dtype=np.float64)
    tris = np.ascontiguousarray(faces, dtype=np.int32)

    shape = np.array(ref.shape[:3])
    voxel_margin = margin / float(np.mean(ref.header.get_zooms()[:3]))
    low = np.maximum(np.floor(verts.min(axis=0) - voxel_margin).astype(int), 0)
    high = np.minimum(np.ceil(verts.max(axis=0) + voxel_margin).astype(int), shape - 1)
    grid = np.meshgrid(*[np.arange(a, b + 1) for a, b in zip(low, high, strict=True)], indexing="ij")
    points = np.ascontiguousarray(np.stack([g.ravel() for g in grid], axis=1), dtype=np.float64)

    winding = np.asarray(igl.fast_winding_number(verts, tris, points))
    # FreeSurfer writes faces in either order depending on the sign of the surface's
    # embedded vox2ras determinant, so take the orientation from the mesh itself.
    corners = verts[tris]
    if np.einsum("ij,ij->i", corners[:, 0], np.cross(corners[:, 1], corners[:, 2])).sum() < 0:
        winding = -winding

    mask = np.zeros(ref.shape[:3], dtype=bool)
    mask[low[0] : high[0] + 1, low[1] : high[1] + 1, low[2] : high[2] + 1] = (winding > 0.5).reshape(grid[0].shape)
    return mask


def main(args: argparse.Namespace) -> int | str:
    """
    Write the ribbon volumes for one subject.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line options.

    Returns
    -------
    int or str
        0 on success, an error message otherwise.
    """
    # Left to itself libigl takes one thread per core, ignoring the budget the pipeline hands
    # out. The published wheels parallelise with std::thread and read IGL_NUM_THREADS, while a
    # build from source can use OpenMP instead, so set both. Either way the value is read when
    # the library is first used, which is why this has to happen before igl is imported.
    if args.threads > 0:
        os.environ["IGL_NUM_THREADS"] = str(args.threads)
        os.environ["OMP_NUM_THREADS"] = str(args.threads)

    try:
        import igl  # noqa: F401
    except ImportError:
        return "volmask.py needs libigl, install it with 'pip install libigl'."

    if args.lh_only and args.rh_only:
        return "Pass at most one of --lh-only and --rh-only."

    mri_dir = args.sd / args.sid / "mri"
    surf_dir = args.sd / args.sid / "surf"
    ref_file = mri_dir / f"{args.aseg_name}.mgz"
    if not ref_file.is_file():
        return f"Could not find the reference volume {ref_file}."

    ref = nib.load(ref_file)
    hemis = ["lh", "rh"]
    if args.lh_only:
        hemis = ["lh"]
    elif args.rh_only:
        hemis = ["rh"]

    inside: dict[tuple[str, str], np.ndarray] = {}
    for hemi in hemis:
        for surf in (args.surf_white, args.surf_pial):
            surf_file = surf_dir / f"{hemi}.{surf}"
            if not surf_file.is_file():
                return f"Could not find the surface {surf_file}."
            inside[(hemi, surf)] = inside_surface(surf_file, ref, args.margin)
            print(f"{surf_file.name}: {int(inside[(hemi, surf)].sum())} voxels inside")

    labels = {
        "lh": (args.label_left_white, args.label_left_ribbon),
        "rh": (args.label_right_white, args.label_right_ribbon),
    }

    def write(data: np.ndarray, path: Path) -> None:
        image = nib.MGHImage(data, ref.affine, ref.header)
        image.set_data_dtype(np.uint8)
        nib.save(image, path)
        print(f"wrote {path}")

    # rh first, so that a voxel both hemispheres claim ends up left, as mris_volmask does
    ribbon = np.zeros(ref.shape[:3], dtype=np.uint8)
    for hemi in [h for h in ("rh", "lh") if h in hemis]:
        white, pial = inside[(hemi, args.surf_white)], inside[(hemi, args.surf_pial)]
        wm_label, gm_label = labels[hemi]
        ribbon[white | pial] = 0
        ribbon[pial & ~white] = gm_label
        ribbon[white] = wm_label
        # the per-hemisphere ribbon is written before that tie-break, again as mris_volmask does
        write((pial & ~white).astype(np.uint8), mri_dir / f"{hemi}.{args.out_root}.mgz")

    if len(hemis) == 2:
        overlap = int(
            (
                (inside[("lh", args.surf_white)] | inside[("lh", args.surf_pial)])
                & (inside[("rh", args.surf_white)] | inside[("rh", args.surf_pial)])
            ).sum()
        )
        print(f"hemi masks overlap voxels = {overlap}")

    if ribbon[0, 0, 0] != 0:
        return "Voxel (0, 0, 0) is labelled, the surfaces do not match the reference volume."

    write(ribbon, mri_dir / f"{args.out_root}.mgz")
    return 0


if __name__ == "__main__":
    sys.exit(main(make_parser().parse_args()))
