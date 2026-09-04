#!/usr/bin/env python3

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

"""
Compare two FastSurfer subject directories by content.

``diff`` and checksums are not usable for this: the file headers record timestamps, the command
line, the user name and the FreeSurfer version, so two runs that produced exactly the same
measurements still differ byte for byte. This compares what the files mean instead, and reports how
large each difference is, which is what turns "the runs look slightly different" into a number.

What is compared, per output type:

- volumes (``mri/*.mgz``, ``mri/*.nii.gz``): the voxel array, and the geometry (the affine)
  separately, so a pure header change is not reported as a data change
- surfaces (``surf/?h.white``, ``?h.pial``, ...): the vertex coordinates, the face list and the
  volume geometry in the header, again separately. The vertex count is checked first, because a
  different count means an earlier step diverged and a per-vertex comparison would be meaningless
- per-vertex data (``surf/?h.curv``, ``?h.thickness``, ``?h.w-g.pct.mgh``, ...): the scalar array
- transforms (``mri/transforms/*.lta``, ``*.xfm``): the RMS distance between the two transforms in
  mm, and the source and destination volume geometry separately
- statistics (``stats/*.stats``): the numeric columns, ignoring the header comment block
- annotations (``label/*.annot``): the per-vertex label array
- the rest of ``surf/`` (``callosum.surf``, ``autodet.gw.stats.?h.dat``, ``*.w``): the surface, the
  text or the bytes, whichever the format allows

Transforms are read with ``neuroreg``, which FastSurfer depends on anyway, so that the distance is
measured on the RAS-to-RAS form and a transform stored as vox-to-vox in one run and as RAS-to-RAS
in the other is not reported as a difference. Without it, the files fall back to a text comparison.

Typical uses are checking whether two runs of the same input agree, and checking whether a change
to FastSurfer, to FreeSurfer or to the hardware moved any measurement.

Exits 0 when everything compared is identical and 1 otherwise, so it can be used in a script.
"""

import argparse
import re
import sys
from collections.abc import Callable
from pathlib import Path

import nibabel as nib
import numpy as np

# surf/ mixes three formats in one directory, so route by name rather than feeding everything to
# the surface reader. A morph token anywhere wins: lh.area.pial is per-vertex data, not a surface.
MORPH = re.compile(r"(^|\.)(curv|sulc|thickness|area|volume|crv|H|K|mid|avg_curv|jacobian_white|w-g\.pct)(\.|$)")
GEOMETRY = re.compile(r"\.(orig|white|pial|inflated|sphere|smoothwm|qsphere|premesh|nofix|reg|preaparc)(\.|$)")
TEXT_SUFFIXES = (".txt", ".pointset", ".label", ".log", ".dat")

# the fields of a FreeSurfer volume geometry header, carried both by an LTA and by a surface file.
# filename is left out on purpose: it records where the volume sat, so two runs in different
# directories differ in it without anything having changed.
GEOMETRY_FIELDS = ("valid", "volume", "voxelsize", "xras", "yras", "zras", "cras")

# converting a vox-to-vox transform to RAS-to-RAS costs a few 1e-14 mm of rounding, so a run that
# only stored the same transform differently must not be reported as different. A picometre is far
# below anything a registration could mean and far above that rounding.
TRANSFORM_TOLERANCE_MM = 1e-9


def _geometry_notes(label: str, ga: dict, gb: dict) -> str:
    """Name the volume geometry fields that differ, or return an empty string."""
    if not ga and not gb:
        return ""
    if not ga or not gb:
        return f"{label} geometry is recorded on one side only"
    differing = [
        field for field in GEOMETRY_FIELDS if not np.array_equal(np.asarray(ga.get(field)), np.asarray(gb.get(field)))
    ]
    return f"{label} geometry differs: {', '.join(differing)}" if differing else ""


def compare_volumes(a: Path, b: Path) -> str:
    """Voxel data and geometry of two volumes, reported separately."""
    ia, ib = nib.load(str(a)), nib.load(str(b))
    da, db = np.asarray(ia.dataobj), np.asarray(ib.dataobj)
    if da.shape != db.shape:
        return f"shape {da.shape} vs {db.shape}"
    notes = []
    if not np.array_equal(ia.affine, ib.affine):
        notes.append(f"geometry differs, max|d|={np.abs(ia.affine - ib.affine).max():.3g}")
    if not np.array_equal(da, db):
        diff = np.abs(da.astype(np.float64) - db.astype(np.float64))
        notes.append(f"{int((da != db).sum())} voxels differ, max|d|={diff.max():.6g}")
    return "; ".join(notes)


def compare_surfaces(a: Path, b: Path) -> str:
    """Vertex coordinates, faces and header geometry of two surfaces."""
    va, fa, ga = nib.freesurfer.read_geometry(str(a), read_metadata=True)
    vb, fb, gb = nib.freesurfer.read_geometry(str(b), read_metadata=True)
    if va.shape != vb.shape:
        return f"VERTEX COUNT {va.shape[0]} vs {vb.shape[0]}"
    notes = []
    geometry = _geometry_notes("volume", ga, gb)
    if geometry:
        notes.append(geometry)
    if not np.array_equal(fa, fb):
        notes.append(f"{int((fa != fb).any(axis=1).sum())} faces differ")
    if not np.array_equal(va, vb):
        diff = np.abs(va - vb)
        notes.append(
            f"{int((diff.max(axis=1) > 0).sum())} vertices differ, max|d|={diff.max():.6g}, mean|d|={diff.mean():.3g}"
        )
    return "; ".join(notes)


def compare_morph(a: Path, b: Path) -> str:
    """Per-vertex scalar data, such as curv, sulc or thickness."""
    da = nib.freesurfer.read_morph_data(str(a))
    db = nib.freesurfer.read_morph_data(str(b))
    if da.shape != db.shape:
        return f"length {da.shape[0]} vs {db.shape[0]}"
    if np.array_equal(da, db):
        return ""
    diff = np.abs(da - db)
    return f"{int((diff > 0).sum())} values differ, max|d|={diff.max():.6g}"


def _read_transform(path: Path):
    """A transform as an LTA, whichever of the two ASCII formats holds it."""
    # imported here rather than at module scope: neuroreg pulls in torch, which takes seconds, and
    # a comparison that touches no transform should not pay for it
    from neuroreg.transforms import LTA, XFM

    if path.suffix == ".xfm":
        return XFM.read(path).to_lta()
    return LTA.read(path)


def _compare_transforms_as_text(a: Path, b: Path) -> str:
    """Fall back to the file text, for when neuroreg is not installed."""

    def content(path: Path) -> str:
        lines = path.read_text(errors="replace").splitlines()
        return "\n".join(line for line in lines if "filename" not in line and not line.startswith("%"))

    if content(a) == content(b):
        return ""
    return "text differs, install neuroreg for a distance in mm"


def compare_transforms(a: Path, b: Path) -> str:
    """Two transforms, as an RMS distance in mm, and the geometry they refer to."""
    try:
        la, lb = _read_transform(a), _read_transform(b)
    except ImportError:
        return _compare_transforms_as_text(a, b)
    notes = []
    # the RAS-to-RAS form, so storing the same transform as vox-to-vox in one run and as RAS-to-RAS
    # in the other is not a difference. This is dist type 2 of FreeSurfer's lta_diff.
    distance = la.affine_dist(lb)
    if distance > TRANSFORM_TOLERANCE_MM:
        notes.append(f"differs by {distance:.6g} mm RMS")
    for label, ga, gb in (("src", la.src, lb.src), ("dst", la.dst, lb.dst)):
        geometry = _geometry_notes(f"{label} volume", ga, gb)
        if geometry:
            notes.append(geometry)
    return "; ".join(notes)


def compare_matrix_text(a: Path, b: Path) -> str:
    """A plain-text matrix, such as the talairach t4 file, ignoring its comment header."""

    def matrix(path: Path) -> np.ndarray:
        lines = path.read_text(errors="replace").splitlines()
        rows = [line.split() for line in lines if line.strip() and not line.lstrip().startswith("#")]
        return np.array(rows, dtype=float)

    ma, mb = matrix(a), matrix(b)
    if ma.shape != mb.shape:
        return f"shape {ma.shape} vs {mb.shape}"
    if np.array_equal(ma, mb):
        return ""
    return f"matrix differs, max|d|={np.abs(ma - mb).max():.6g}"


def compare_annotations(a: Path, b: Path) -> str:
    """The per-vertex label array of two annotation files."""
    la = nib.freesurfer.read_annot(str(a))[0]
    lb = nib.freesurfer.read_annot(str(b))[0]
    if la.shape != lb.shape:
        return f"length {la.shape[0]} vs {lb.shape[0]}"
    relabelled = int((la != lb).sum())
    return f"{relabelled} vertices relabelled" if relabelled else ""


_NUMBER = re.compile(r"^-?\d+(\.\d+)?([eE][-+]?\d+)?$")


def _stats_numbers(path: Path) -> dict[str, list[float]]:
    """The numbers in a .stats file, keyed by row or measure name, without the header prose."""
    values: dict[str, list[float]] = {}
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("# Measure"):
            parts = [p.strip() for p in line[len("# Measure") :].split(",")]
            numbers = [float(p) for p in parts if _NUMBER.match(p)]
            if parts and numbers:
                values["measure:" + parts[0]] = numbers
        elif not line.startswith("#") and line.strip():
            fields = line.split()
            numbers = [float(f) for f in fields if _NUMBER.match(f)]
            if numbers:
                name = next((f for f in fields if not _NUMBER.match(f)), fields[0])
                values["row:" + name] = numbers
    return values


def compare_stats(a: Path, b: Path) -> str:
    """The numeric content of two .stats files, with the worst relative change."""
    va, vb = _stats_numbers(a), _stats_numbers(b)
    notes = []
    if va.keys() != vb.keys():
        notes.append(f"{len(set(va) ^ set(vb))} entries only on one side")
    worst, differing, ragged = 0.0, 0, 0
    for key in set(va) & set(vb):
        # a differing column count means one side gained or lost a value, which zip would drop
        if len(va[key]) != len(vb[key]):
            ragged += 1
            continue
        for x, y in zip(va[key], vb[key], strict=True):
            if x != y:
                differing += 1
                worst = max(worst, abs(x - y) / max(abs(x), 1e-12))
    if ragged:
        notes.append(f"{ragged} entries have a different number of values")
    if differing:
        notes.append(f"{differing} values differ, worst relative {worst * 100:.4f}%")
    return "; ".join(notes)


def compare_text(a: Path, b: Path) -> str:
    """Plain text files, compared verbatim."""
    if a.read_text(errors="replace") == b.read_text(errors="replace"):
        return ""
    return "text differs"


def compare_bytes(a: Path, b: Path) -> str:
    """Formats that record nothing volatile, so the bytes themselves are the content."""
    return "" if a.read_bytes() == b.read_bytes() else "bytes differ"


def _route_surf(name: str) -> "Callable[[Path, Path], str] | None":
    """Pick the comparison for a file in surf/, or None to skip an unknown format."""
    if name.endswith(TEXT_SUFFIXES):
        return compare_text
    # a per-vertex map stored in volume format, such as ?h.w-g.pct.mgh, which nibabel reads as an
    # N x 1 x 1 volume
    if name.endswith(".mgh"):
        return compare_volumes
    if MORPH.search(name):
        return compare_morph
    if GEOMETRY.search(name) or name.endswith(".surf"):
        return compare_surfaces
    # the w file format holds a vertex count and per-vertex values and nothing volatile, so its
    # bytes can be compared as they are
    if name.endswith(".w"):
        return compare_bytes
    return None


def _route_transforms(name: str) -> "Callable[[Path, Path], str] | None":
    """Pick the comparison for a file in mri/transforms/, or None to skip an unknown format."""
    if name.endswith((".lta", ".xfm")):
        return compare_transforms
    if name.endswith(".txt"):
        return compare_matrix_text
    return None


# the third entry routes a file name to its comparison, because mri/transforms/ and surf/ each hold
# more than one format
GROUPS = (
    ("mri", ("*.mgz", "*.nii.gz"), lambda _name: compare_volumes),
    ("mri/transforms", ("*.lta", "*.xfm", "*_vox2vox.txt"), _route_transforms),
    ("surf", ("lh.*", "rh.*", "*.surf", "*.dat", "*.w"), _route_surf),
    ("stats", ("*.stats",), lambda _name: compare_stats),
    ("label", ("*.annot",), lambda _name: compare_annotations),
)


def find_subject(root: Path, subject: "str | None") -> Path:
    """The subject directory inside root, or root itself if it is already one."""
    if not root.is_dir():
        sys.exit(f"ERROR: {root} is not a directory.")
    if subject:
        # checked, so that a typo cannot end as "0 files compared" and an exit code of 0
        if not (root / subject / "mri").is_dir():
            sys.exit(f"ERROR: {root / subject} is not a subject directory, it has no mri.")
        return root / subject
    if (root / "mri").is_dir():
        return root
    candidates = [p for p in sorted(root.iterdir()) if p.is_dir() and (p / "mri").is_dir()]
    if len(candidates) != 1:
        names = [p.name for p in candidates]
        sys.exit(f"ERROR: cannot pick a subject in {root}, found {names}. Use --subject.")
    return candidates[0]


def make_parser() -> argparse.ArgumentParser:
    """Create the command line interface."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("a", type=Path, help="first subject directory, or a directory holding one")
    parser.add_argument("b", type=Path, help="second subject directory, or a directory holding one")
    parser.add_argument(
        "--subject",
        default=None,
        help="subject id, if the two paths hold more than one subject",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="also list the files that are identical, not only the differences",
    )
    return parser


def main() -> int:
    """Compare the two subject directories and report the differences."""
    args = make_parser().parse_args()
    a_dir, b_dir = find_subject(args.a, args.subject), find_subject(args.b, args.subject)
    print(f"A = {a_dir}\nB = {b_dir}\n")

    differing: list[str] = []
    only_one_side: list[str] = []
    compared = 0

    for group, patterns, route in GROUPS:
        if not (a_dir / group).is_dir():
            continue
        names = sorted({p.name for pattern in patterns for p in (a_dir / group).glob(pattern)})
        for name in names:
            compare = route(name)
            if compare is None:
                continue
            file_a, file_b = a_dir / group / name, b_dir / group / name
            if not file_b.exists():
                continue  # reported by the one-sided pass below
            try:
                note = compare(file_a, file_b)
            except Exception as error:  # noqa: BLE001  report, do not abort
                note = f"could not compare: {type(error).__name__}: {error}"
            compared += 1
            if note:
                differing.append(f"{group}/{name}: {note}")
            elif args.all:
                print(f"  same  {group}/{name}")

    # every name present in one directory and not in the other, whether or not there is a
    # comparison for its format: a missing output matters even where the content cannot be read
    for group, _, _ in GROUPS:
        in_a = {p.name for p in (a_dir / group).glob("*")} if (a_dir / group).is_dir() else set()
        in_b = {p.name for p in (b_dir / group).glob("*")} if (b_dir / group).is_dir() else set()
        only_one_side.extend(f"{group}/{name} (only in A)" for name in sorted(in_a - in_b))
        only_one_side.extend(f"{group}/{name} (only in B)" for name in sorted(in_b - in_a))

    for entry in differing:
        print(f"  DIFF  {entry}")
    for entry in only_one_side:
        print(f"  ONLY  {entry}")
    print(f"\n{compared} files compared, {len(differing)} differ, {len(only_one_side)} on one side only")
    if compared == 0:
        # not "identical": there was nothing to compare, so say so rather than exiting 0
        print("ERROR: no comparable files found, are these subject directories?", file=sys.stderr)
        return 2
    return 1 if differing or only_one_side else 0


if __name__ == "__main__":
    sys.exit(main())
