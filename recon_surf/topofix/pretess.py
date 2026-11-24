# Copyright 2026 DeepMI Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
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
Join a binary mask where its voxels touch only at an edge or a corner, as FreeSurfer's mri_pretess does.

Three configurations are resolved by adding voxels, chosen by an intensity volume:

* edge: two voxels diagonal within a plane whose two common face neighbours are empty; the brighter
  neighbour is added.
* corner: two voxels diagonal in space without a face path of two voxels between them; the path with
  the larger summed intensity is added.
* background corner: two empty voxels diagonal in space whose face paths are all set; the brighter of
  the two is added.

Voxels are only added, and only inside the bounding box of the mask.
"""

import heapq

import numpy as np

MODES = ("fixed", "freesurfer")

# (first, second) face neighbours of each edge configuration, in the order mri_pretess resolves them;
# the two voxels are diagonal along first + second, and ties go to second
EDGES = (
    ((0, 1, 0), (1, 0, 0)),
    ((0, 1, 0), (-1, 0, 0)),
    ((0, 1, 0), (0, 0, 1)),
    ((0, 1, 0), (0, 0, -1)),
    ((1, 0, 0), (0, 0, 1)),
    ((-1, 0, 0), (0, 0, 1)),
)
# signs of the second and third axis of the four space diagonals, in the order of mri_pretess
DIAGONALS = ((1, 1), (1, -1), (-1, -1), (-1, 1))


def _add(p, o):
    return p[0] + o[0], p[1] + o[1], p[2] + o[2]


def _corner(sy: int, sz: int):
    """Diagonal of a corner configuration and its six face paths, in the order of mri_pretess."""
    ex, ey, ez = (1, 0, 0), (0, sy, 0), (0, 0, sz)
    exy, exz, eyz = (1, sy, 0), (1, 0, sz), (0, sy, sz)
    paths = ((ex, exy), (ex, exz), (ey, eyz), (ey, exy), (ez, exz), (ez, eyz))
    return (1, sy, sz), paths


def _around(sy: int, sz: int, all_six: bool):
    """
    Face path voxels a background corner configuration requires to be set.

    mri_pretess checks only five of the six: it tests ex twice and never ez.
    """
    around = [(1, 0, 0), (1, sy, 0), (0, sy, 0), (0, sy, sz), (1, 0, sz)]
    return around + [(0, 0, sz)] if all_six else around


class _Volume:
    """A mask padded by one empty voxel, so that every configuration lies inside, and its intensity."""

    def __init__(self, mask: np.ndarray, intensity: np.ndarray):
        self.seg = np.pad(np.asarray(mask, dtype=bool), 1)
        self.val = np.pad(np.asarray(intensity, dtype=np.float32), 1)

    def view(self, a: np.ndarray, o) -> np.ndarray:
        """a[p + o] for every position p of the unpadded volume."""
        return a[tuple(slice(1 + o[d], a.shape[d] - 1 + o[d]) for d in range(3))]


def _fixed(vol: _Volume, max_iterations: int) -> None:
    """Resolve all configurations present at once, repeatedly; ties add both choices."""
    seg, val, view = vol.seg, vol.val, vol.view
    for _ in range(max_iterations):
        add = np.zeros_like(seg)
        here = view(seg, (0, 0, 0))
        for first, second in EDGES:
            m = here & view(seg, _add(first, second)) & ~view(seg, first) & ~view(seg, second)
            if m.any():
                va, vb = view(val, first), view(val, second)
                view(add, first)[...] |= m & (va >= vb)
                view(add, second)[...] |= m & (vb >= va)
        for sy, sz in DIAGONALS:
            diag, paths = _corner(sy, sz)
            m = here & view(seg, diag)
            for a, b in paths:
                m &= ~(view(seg, a) & view(seg, b))
            if m.any():
                dist = [view(val, a) + view(val, b) for a, b in paths]
                best = np.maximum.reduce(dist)
                for (a, b), d in zip(paths, dist, strict=True):
                    sel = m & (d == best)
                    view(add, a)[...] |= sel
                    view(add, b)[...] |= sel
        for sy, sz in DIAGONALS:
            diag, _ = _corner(sy, sz)
            m = ~here & ~view(seg, diag)
            for o in _around(sy, sz, all_six=True):
                m &= view(seg, o)
            if m.any():
                vp, vq = view(val, (0, 0, 0)), view(val, diag)
                view(add, (0, 0, 0))[...] |= m & (vp >= vq)
                view(add, diag)[...] |= m & (vq >= vp)
        add &= ~seg
        if not add.any():
            return
        seg |= add


class _Rule:
    """One configuration of mri_pretess, checked and resolved voxel by voxel."""

    def __init__(self, vol: _Volume, offsets):
        self.vol = vol
        self.offsets = offsets
        # the log position up to which this rule has seen every change, see _replicate
        self.seen = None

    def vector(self) -> np.ndarray:
        """Where the configuration is present, for every position of the unpadded volume."""
        raise NotImplementedError

    def present(self, p) -> bool:
        raise NotImplementedError

    def resolve(self, p) -> list:
        """Add the voxels that resolve the configuration at p and return the ones that were not set."""
        raise NotImplementedError


class _Edge(_Rule):
    def __init__(self, vol, first, second):
        self.first, self.second, self.diag = first, second, _add(first, second)
        super().__init__(vol, ((0, 0, 0), self.diag, first, second))

    def vector(self):
        s, v = self.vol.seg, self.vol.view
        return v(s, (0, 0, 0)) & v(s, self.diag) & ~v(s, self.first) & ~v(s, self.second)

    def present(self, p):
        s = self.vol.seg
        return s[p] and s[_add(p, self.diag)] and not s[_add(p, self.first)] and not s[_add(p, self.second)]

    def resolve(self, p):
        a, b = _add(p, self.first), _add(p, self.second)
        v = a if self.vol.val[a] > self.vol.val[b] else b
        self.vol.seg[v] = True
        return [v]


class _Corner(_Rule):
    def __init__(self, vol, sy, sz):
        self.diag, self.paths = _corner(sy, sz)
        super().__init__(vol, ((0, 0, 0), self.diag, *{o for path in self.paths for o in path}))

    def vector(self):
        s, v = self.vol.seg, self.vol.view
        m = v(s, (0, 0, 0)) & v(s, self.diag)
        for a, b in self.paths:
            m &= ~(v(s, a) & v(s, b))
        return m

    def present(self, p):
        s = self.vol.seg
        if not (s[p] and s[_add(p, self.diag)]):
            return False
        return not any(s[_add(p, a)] and s[_add(p, b)] for a, b in self.paths)

    def resolve(self, p):
        val = self.vol.val
        dist = [float(val[_add(p, a)]) + float(val[_add(p, b)]) for a, b in self.paths]
        best = 0
        for n in range(1, len(dist)):
            if dist[best] < dist[n]:
                best = n
        changed = []
        for o in self.paths[best]:
            v = _add(p, o)
            if not self.vol.seg[v]:
                self.vol.seg[v] = True
                changed.append(v)
        return changed


class _BackgroundCorner(_Rule):
    def __init__(self, vol, sy, sz):
        self.diag, _ = _corner(sy, sz)
        self.around = _around(sy, sz, all_six=False)
        super().__init__(vol, ((0, 0, 0), self.diag, *self.around))

    def vector(self):
        s, v = self.vol.seg, self.vol.view
        m = ~v(s, (0, 0, 0)) & ~v(s, self.diag)
        for o in self.around:
            m &= v(s, o)
        return m

    def present(self, p):
        s = self.vol.seg
        if s[p] or s[_add(p, self.diag)]:
            return False
        return all(s[_add(p, o)] for o in self.around)

    def resolve(self, p):
        q = _add(p, self.diag)
        v = p if self.vol.val[p] > self.vol.val[q] else q
        self.vol.seg[v] = True
        return [v]


def _replicate(vol: _Volume, max_iterations: int) -> None:
    """
    Reproduce mri_pretess exactly.

    mri_pretess resolves each configuration by scanning the volume (third axis outermost) and changing
    it in place, repeating the scan until it changes nothing, so the result depends on the scan order.
    Here each scan visits only the positions where the configuration can be present, in scan order: a
    configuration only appears next to a voxel that was added, so every added voxel goes into a log, and
    a scan starts from the positions around the voxels added since that rule last finished.
    """
    log = []
    shape = vol.seg.shape
    key = lambda p: (p[2], p[1], p[0])  # noqa: E731
    inside = lambda q: all(1 <= q[d] < shape[d] - 1 for d in range(3))  # noqa: E731
    rules = [_Edge(vol, *e) for e in EDGES]
    rules += [_Corner(vol, *d) for d in DIAGONALS]
    rules += [_BackgroundCorner(vol, *d) for d in DIAGONALS]

    def around(rule, added):
        return {q for v in added for o in rule.offsets if inside(q := (v[0] - o[0], v[1] - o[1], v[2] - o[2]))}

    def scan(rule, positions):
        heap = [(key(p), p) for p in positions]
        heapq.heapify(heap)
        done, nfound = set(), 0
        while heap:
            kp, p = heapq.heappop(heap)
            if p in done:
                continue
            done.add(p)
            if not rule.present(p):
                continue
            changed = rule.resolve(p)
            log.extend(changed)
            nfound += len(changed)
            # configurations the new voxels create later in this scan
            for q in around(rule, changed):
                if key(q) > kp and q not in done:
                    heapq.heappush(heap, (key(q), q))
        return nfound

    def run(rule):
        if rule.seen is None:
            positions = [tuple(int(x) + 1 for x in p) for p in np.argwhere(rule.vector())]
        else:
            positions = around(rule, log[rule.seen:])
        total = 0
        while True:
            start = len(log)
            nfound = scan(rule, positions)
            if nfound == 0:
                break
            total += nfound
            positions = around(rule, log[start:])
        rule.seen = len(log)
        return total

    for _ in range(max_iterations):
        if sum(run(rule) for rule in rules) == 0:
            return


def pretess(mask: np.ndarray, intensity: np.ndarray, mode: str = "fixed", max_iterations: int = 100) -> np.ndarray:
    """
    Add voxels to a binary mask so that its voxels touch at faces only, choosing them by intensity.

    Parameters
    ----------
    mask : np.ndarray
        Boolean volume.
    intensity : np.ndarray
        Intensity volume of the same shape, for example ``brain.mgz``.
    mode : {"fixed", "freesurfer"}, default="fixed"
        ``"freesurfer"`` reproduces mri_pretess voxel for voxel, including its dependence on the scan
        order and its background corner check of five of the six voxels. ``"fixed"`` resolves all
        configurations present at the same time, which makes the result independent of the order and
        the orientation of the voxel axes; it checks all six voxels and adds both choices on a tie.
    max_iterations : int, default=100
        Maximum number of rounds over all configurations, as in mri_pretess.

    Returns
    -------
    np.ndarray
        Boolean volume, a superset of ``mask``.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
    mask = np.asarray(mask, dtype=bool)
    out = mask.copy()
    if not mask.any():
        return out
    # nothing is added outside the bounding box of the mask
    idx = np.argwhere(mask)
    box = tuple(slice(lo, hi + 1) for lo, hi in zip(idx.min(axis=0), idx.max(axis=0), strict=True))
    vol = _Volume(mask[box], np.asarray(intensity)[box])
    (_fixed if mode == "fixed" else _replicate)(vol, max_iterations)
    out[box] = vol.seg[1:-1, 1:-1, 1:-1]
    return out
