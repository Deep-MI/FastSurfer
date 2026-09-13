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

A few hundred voxels near the medial wall fall inside both hemispheres. Rather
than hand all of them to the left as mris_volmask does, which biases left cortex
in the same direction in every subject, each goes to the hemisphere whose pial
surface it lies deeper inside.
"""

# query points per call to libigl, chosen so the coordinates stay around 100 MB whatever the
# voxel size; the per-call cost of rebuilding the mesh BVH is small against a slab this size
QUERY_CHUNK = 4_000_000


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
        help="number of threads, 0 to leave the thread count to the environment (default: 1)",
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

    vertices, faces, meta = fsio.read_geometry(surf_file, read_metadata=True)
    # A surface records the volume it was built against. get_vox2ras_tkr derives the transform
    # below from the dimensions and voxel sizes alone, so those are what have to agree: reading
    # surfaces against a volume on another grid would place every vertex wrongly and produce a
    # ribbon that looks plausible and is not.
    surf_dims = tuple(int(v) for v in meta["volume"])
    ref_dims = tuple(int(v) for v in ref.shape[:3])
    if surf_dims != ref_dims or not np.allclose(meta["voxelsize"], ref.header.get_zooms()[:3], atol=1e-4):
        raise ValueError(
            f"{surf_file.name} was built against a {list(surf_dims)} grid at "
            f"{np.round(np.asarray(meta['voxelsize']), 4).tolist()} mm, but the reference volume is "
            f"{list(ref_dims)} at {np.round(np.asarray(ref.header.get_zooms()[:3]), 4).tolist()} mm"
        )

    # surfaces are in tkrRAS, the grid is in voxels
    to_vox = np.linalg.inv(ref.header.get_vox2ras_tkr())
    verts = np.ascontiguousarray(nib.affines.apply_affine(to_vox, vertices), dtype=np.float64)
    tris = np.ascontiguousarray(faces, dtype=np.int64)

    # FreeSurfer writes faces in either order depending on the sign of the surface's
    # embedded vox2ras determinant, so take the orientation from the mesh itself.
    corners = verts[tris]
    inward = np.einsum("ij,ij->i", corners[:, 0], np.cross(corners[:, 1], corners[:, 2])).sum() < 0
    del corners

    shape = np.array(ref.shape[:3])
    voxel_margin = margin / float(np.mean(ref.header.get_zooms()[:3]))
    low = np.maximum(np.floor(verts.min(axis=0) - voxel_margin).astype(int), 0)
    high = np.minimum(np.ceil(verts.max(axis=0) + voxel_margin).astype(int), shape - 1)

    mask = np.zeros(ref.shape[:3], dtype=bool)
    ys = np.arange(low[1], high[1] + 1, dtype=np.float64)
    zs = np.arange(low[2], high[2] + 1, dtype=np.float64)
    plane = ys.size * zs.size
    if plane == 0 or high[0] < low[0]:
        return mask

    # Evaluate in slabs rather than building the whole query grid at once: the points cost
    # 24 bytes each, so at 0.4 mm a single array would be hundreds of MB per surface. Each call
    # rebuilds the mesh BVH, about 50 ms, which against a slab of this size is around 1%.
    rows = max(1, int(QUERY_CHUNK // plane))
    for start in range(low[0], high[0] + 1, rows):
        xs = np.arange(start, min(start + rows, high[0] + 1), dtype=np.float64)
        block = np.empty((xs.size, ys.size, zs.size, 3), dtype=np.float64)
        block[..., 0] = xs[:, None, None]
        block[..., 1] = ys[None, :, None]
        block[..., 2] = zs[None, None, :]
        # C-contiguous, so this reshape is a view and igl gets the layout it wants
        winding = np.asarray(igl.fast_winding_number(verts, tris, block.reshape(-1, 3)))
        if inward:
            winding = -winding
        mask[start : start + xs.size, low[1] : high[1] + 1, low[2] : high[2] + 1] = (winding > 0.5).reshape(
            xs.size, ys.size, zs.size
        )
    return mask


def surface_distance(
    surf_file: Path,
    ref: nib.freesurfer.mghformat.MGHImage,
    points: np.ndarray,
) -> np.ndarray:
    """
    Distance from each point to the nearest point on a surface, in voxels.

    For a point known to be inside the surface this is how deep inside it lies.

    Parameters
    ----------
    surf_file : Path
        FreeSurfer surface to measure against.
    ref : nibabel.freesurfer.mghformat.MGHImage
        Volume whose grid the points are given in.
    points : numpy.ndarray
        Voxel coordinates, shape (n, 3).

    Returns
    -------
    numpy.ndarray
        Distance per point, shape (n,).
    """
    import igl

    vertices, faces = fsio.read_geometry(surf_file)
    to_vox = np.linalg.inv(ref.header.get_vox2ras_tkr())
    verts = np.ascontiguousarray(nib.affines.apply_affine(to_vox, vertices), dtype=np.float64)
    tris = np.ascontiguousarray(faces, dtype=np.int64)
    squared, _, _ = igl.point_mesh_squared_distance(np.ascontiguousarray(points), verts, tris)
    return np.sqrt(np.asarray(squared))


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
    if args.surf_white == args.surf_pial:
        return "--surf_white and --surf_pial name the same surface, the ribbon would be empty."

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

    # check every surface before doing any work, so a missing file fails immediately
    surf_files = {}
    for hemi in hemis:
        for surf in (args.surf_white, args.surf_pial):
            surf_files[(hemi, surf)] = surf_dir / f"{hemi}.{surf}"
            if not surf_files[(hemi, surf)].is_file():
                return f"Could not find the surface {surf_files[(hemi, surf)]}."

    labels = {
        "lh": (args.label_left_white, args.label_left_ribbon),
        "rh": (args.label_right_white, args.label_right_ribbon),
    }

    def write(data: np.ndarray, path: Path) -> None:
        image = nib.MGHImage(data, ref.affine, ref.header)
        image.set_data_dtype(np.uint8)
        nib.save(image, path)
        print(f"wrote {path}")

    # One hemisphere at a time, so only its two masks are held rather than all four; at 0.4 mm
    # a single mask over a 640^3 grid is already 260 MB.
    ribbon = np.zeros(ref.shape[:3], dtype=np.uint8)
    hemi_ribbons: dict[str, np.ndarray] = {}
    previous = None
    for hemi in [h for h in ("rh", "lh") if h in hemis]:
        masks = {}
        for surf in (args.surf_white, args.surf_pial):
            try:
                masks[surf] = inside_surface(surf_files[(hemi, surf)], ref, args.margin)
            except ValueError as error:
                return str(error)
            print(f"{surf_files[(hemi, surf)].name}: {int(masks[surf].sum())} voxels inside")
        white, pial = masks[args.surf_white], masks[args.surf_pial]

        # The per-hemisphere ribbon records this hemisphere's own claim, before any of the
        # arbitration below, which is what mris_volmask writes too.
        hemi_ribbons[hemi] = (pial & ~white).astype(np.uint8)

        claim = white | pial
        if previous is not None:
            # Whatever is already labelled belongs to the hemisphere done before this one, so
            # anything claimed twice has to be arbitrated. mris_volmask hands all of it to the
            # left, which inflates left cortex by a small amount in every subject. Give each
            # voxel to the hemisphere whose pial surface it lies deeper inside instead, which
            # is symmetric and needs nothing beyond the surfaces already at hand.
            contested = claim & (ribbon != 0)
            count = int(contested.sum())
            if count:
                points = np.argwhere(contested).astype(np.float64)
                mine = surface_distance(surf_files[(hemi, args.surf_pial)], ref, points)
                theirs = surface_distance(surf_files[(previous, args.surf_pial)], ref, points)
                lost = points[mine <= theirs].astype(int)
                claim[lost[:, 0], lost[:, 1], lost[:, 2]] = False
                print(
                    f"hemi masks overlap voxels = {count}, {count - len(lost)} to {hemi} and "
                    f"{len(lost)} to {previous} by depth inside the pial surface"
                )

        wm_label, gm_label = labels[hemi]
        ribbon[claim] = 0
        ribbon[claim & pial & ~white] = gm_label
        ribbon[claim & white] = wm_label
        previous = hemi

    if ribbon[0, 0, 0] != 0:
        return "Voxel (0, 0, 0) is labelled, the surfaces do not match the reference volume."

    # everything is checked before anything is written, so a failure cannot leave a subject
    # directory holding some of the ribbon files and not the rest
    for hemi, data in hemi_ribbons.items():
        write(data, mri_dir / f"{hemi}.{args.out_root}.mgz")
    write(ribbon, mri_dir / f"{args.out_root}.mgz")
    return 0


if __name__ == "__main__":
    sys.exit(main(make_parser().parse_args()))
