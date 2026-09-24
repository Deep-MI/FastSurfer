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
Render the FastSurfer applet icon, fastsurfer-icon.png, from the logo.

The logo sits on a dark rounded square in the macOS icon grid. macOS shrinks an icon of any other
shape onto a light plate of its own. build_release_package.sh uses the committed PNG, so this only
runs when the icon changes. It needs Pillow, which the FastSurfer environment has.
"""
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
LOGO = HERE.parent.parent / "doc" / "images" / "fastsurfer.png"
OUT = HERE / "fastsurfer-icon.png"

CANVAS = 1024
# the macOS icon grid: an 824 px rounded square centred in the 1024 px canvas
TILE = 824
RADIUS = 185
BACKGROUND = (58, 58, 60, 255)
# the share of the tile's width the logo spans
LOGO_SHARE = 0.90
# drawn larger and scaled down, for smooth corners
SUPERSAMPLE = 4


def main() -> None:
    """Write fastsurfer-icon.png next to this script."""
    s = SUPERSAMPLE
    offset = (CANVAS - TILE) // 2 * s
    mask = Image.new("L", (CANVAS * s, CANVAS * s), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (offset, offset, offset + TILE * s - 1, offset + TILE * s - 1), radius=RADIUS * s, fill=255
    )
    mask = mask.resize((CANVAS, CANVAS), Image.LANCZOS)
    icon = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    icon.paste(Image.new("RGBA", (CANVAS, CANVAS), BACKGROUND), mask=mask)

    logo = Image.open(LOGO).convert("RGBA")
    # the visible logo, so the empty margin and the faint halo in the file neither shrink nor
    # off-centre it
    logo = logo.crop(logo.getchannel("A").point(lambda alpha: 255 if alpha > 20 else 0).getbbox())
    width = round(TILE * LOGO_SHARE)
    logo = logo.resize((width, round(width * logo.height / logo.width)), Image.LANCZOS)
    # centred on its coloured part: the drop shadow extends to the lower right and would pull the
    # logo up and to the left
    left, top, right, bottom = _coloured_bbox(logo)
    x = CANVAS // 2 - (left + right) // 2
    y = CANVAS // 2 - (top + bottom) // 2
    icon.alpha_composite(logo, (x, y))
    icon.save(OUT, optimize=True)


def _coloured_bbox(image: Image.Image) -> tuple[int, int, int, int]:
    """Return the bounding box of the visible pixels that are not the dark shadow."""
    visible = image.getchannel("A").point(lambda alpha: 255 if alpha > 20 else 0)
    # the brightest channel rather than luminance, which would count the dark red as shadow
    bright = image.convert("HSV").getchannel("V").point(lambda value: 255 if value > 60 else 0)
    return Image.composite(bright, Image.new("L", image.size, 0), visible).getbbox()


if __name__ == "__main__":
    main()
