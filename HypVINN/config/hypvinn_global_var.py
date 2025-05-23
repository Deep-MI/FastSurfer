# Copyright 2024 AI in Medical Imaging, German Center for Neurodegenerative Diseases(DZNE), Bonn
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

from typing import Literal

import numpy as np

Plane = Literal["axial", "coronal", "sagittal"]


HYPVINN_CLASS_NAMES = {
    0: "Background",

    1: "R-N.opticus",
    2: "L-N.opticus",

    3: "R-C.mammilare",
    6: "L-C.mammilare",

    4: "R-Optic-tract",
    5: "L-Optic-tract",

    7: "R-Chiasma-Opticum",
    8: "L-Chiasma-Opticum",

    9: "Ant-Commisure",
    10: "Third-Ventricle",

    11: "R-Fornix",
    12: "L-Fornix",

    14: "Epiphysis",
    16: "Hypophysis",
    17: "Infundibulum",

    13: "R-Globus-Pallidus",
    20: "L-Globus-pallidus",

    122: "Tuberal-Region",

    126: "L-Med-Hypothalamus",
    226: "R-Med-Hypothalamus",

    127: "L-Lat-Hypothalamus",
    227: "R-Lat-Hypothalamus",

    128: "L-Ant-Hypothalamus",
    228: "R-Ant-Hypothalamus",

    129: "L-Post-Hypothalamus",
    229: "R-Post-Hypothalamus",
}

FS_CLASS_NAMES = {
    "Background": 0,

    "R-N.opticus": 961,
    "L-N.opticus": 962,
    "R-C.mammilare": 963,
    "R-Optic-tract": 964,
    "L-Optic-tract": 965,
    "L-C.mammilare": 966,
    "R-Chiasma-Opticum": 967,
    "L-Chiasma-Opticum": 968,
    "Ant-Commisure": 969,
    "Third-Ventricle": 970,
    "R-Fornix": 971,
    "L-Fornix": 972,
    "Epiphysis": 973,
    "Hypophysis": 974,
    "Infundibulum": 975,
    "Tuberal-Region": 976,
    "L-Med-Hypothalamus": 977,
    "L-Lat-Hypothalamus": 978,
    "L-Ant-Hypothalamus": 979,
    "L-Post-Hypothalamus": 980,
    "R-Med-Hypothalamus": 981,
    "R-Lat-Hypothalamus": 982,
    "R-Ant-Hypothalamus": 983,
    "R-Post-Hypothalamus": 984,
    #excluded ids
    "R-Globus-Pallidus": 985,
    "L-Globus-pallidus": 986,
}

planes = ("axial", "coronal", "sagittal")

hyposubseg_labels = (
    np.array(list(HYPVINN_CLASS_NAMES.keys())),
    np.array([0, 1, 3, 4, 7, 9, 10, 11, 14, 16, 17, 13, 122, 226, 227, 228, 229]),
)

SAG2FULL_MAP = {
        # lbl: sag_lbl_index
        0: 0,
        1: 1,
        2: 1,
        3: 2,
        6: 2,
        4: 3,
        5: 3,
        7: 4,
        8: 4,
        9: 5,
        10: 6,
        11: 7,
        12: 7,
        14: 8,
        16: 9,
        17: 10,
        13: 11,
        20: 11,
        122: 12,
        126: 13,
        226: 13,
        127: 14,
        227: 14,
        128: 15,
        228: 15,
        129: 16,
        229: 16
    }
