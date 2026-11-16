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

"""Gray and white modes without rounding, written as mris_place_surface reads them."""

import numpy as np
import pytest
from scipy.stats import norm

from recon_surf.topofix.gwstats import derived_values, read_stats, refine_stats, smooth_mode, write_stats

# written by mris_autodet_gwstats for a 1 mm subject
STATS = """hemicode           1
white_border_hi    112.261520
white_border_low   64.000000
white_outside_low  53.961357
white_inside_hi    120.000000
white_outside_hi   112.261520
pial_border_hi    53.961357
pial_border_low   33.884075
pial_outside_low  10.000000
pial_inside_hi    93.738480
pial_outside_hi   48.942036
use_mode 1
variablesigma 3.000000
std_scale 1.000000
adWHITE_MATTER_MEAN 110.000000
MAX_WHITE 120.000000
MIN_BORDER_WHITE 85.000000
MAX_BORDER_WHITE 105.000000
MAX_GRAY 95.000000
MID_GRAY 63.811279
MIN_GRAY_AT_CSF_BORDER 40.000000
MAX_GRAY_AT_CSF_BORDER 75.000000
MIN_CSF 10.000000
adMAX_CSF 40.000000
white_mean 103.000000
white_mode 103.000000
white_std 9.261518
gray_mean 64.000000
gray_mode 64.000000
gray_std 10.038642
min_border_white 64.000000
max_border_white 112.261520
min_gray_at_white_border 53.961357
max_gray 93.738480
min_gray_at_csf_border 33.884075
max_gray_at_csf_border 53.961357
min_csf 10.000000
max_csf 43.922714
max_gray_scale 0.000000
max_scale_down 0.200000
"""


@pytest.fixture
def stats(tmp_path):
    path = tmp_path / "autodet.gw.stats.lh.dat"
    path.write_text(STATS)
    return read_stats(path)


def test_formulas_reproduce_freesurfer(stats):
    for key, value in derived_values(stats, stats["white_mode"], stats["gray_mode"]).items():
        assert value == pytest.approx(stats[key], abs=1e-5), key


def normal_samples(mean, std, n=150_000):
    """Evenly spaced quantiles of a normal distribution, a histogram without sampling noise."""
    return norm.ppf((np.arange(n) + 0.5) / n, mean, std)


def test_mode_between_two_integers():
    assert smooth_mode(normal_samples(63.5, 10.0)) == pytest.approx(63.5, abs=0.05)
    assert smooth_mode(normal_samples(102.3, 9.0)) == pytest.approx(102.3, abs=0.05)


def test_refined_file_keeps_its_layout(stats, tmp_path):
    refined = refine_stats(stats, normal_samples(102.3, 9.0), normal_samples(63.5, 10.0))
    out = tmp_path / "refined.dat"
    write_stats(refined, out)
    keys = [line.split()[0] for line in out.read_text().splitlines()]
    assert keys == [line.split()[0] for line in STATS.splitlines()]
    again = read_stats(out)
    assert again["gray_mode"] == pytest.approx(63.5, abs=0.05)
    assert again["white_border_low"] == again["gray_mode"]
    assert again["white_std"] == stats["white_std"] and again["gray_std"] == stats["gray_std"]


def test_gray_mode_ignores_clipped_white_matter(stats):
    # gray samples on white matter clipped to 110, enough to outgrow the smoothed gray peak
    gray = np.concatenate([normal_samples(63.5, 10.0), np.full(40_000, 110.0)])
    refined = refine_stats(stats, normal_samples(102.3, 9.0), gray)
    assert refined["gray_mode"] == pytest.approx(63.5, abs=0.1)


def test_values_set_by_hand_are_not_replaced(stats):
    stats["min_border_white"] = 70.0
    with pytest.raises(ValueError, match="min_border_white"):
        refine_stats(stats, np.full(10, 100.0), np.full(10, 60.0))
