import numpy as np
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
    10: "3rd-Ventricle",

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
   "Background" : 0,

    "R-N.opticus": 901,
    "L-N.opticus": 902,

    "R-C.mammilare": 903,
    "L-C.mammilare": 906,

    "R-Optic-tract": 904,
    "L-Optic-tract": 905,

    "R-Chiasma-Opticum": 907,
    "L-Chiasma-Opticum": 908,

    "Ant-Commisure": 909,
    "3rd-Ventricle": 910,

    "R-Fornix": 911,
    "L-Fornix": 912,

    "Epiphysis": 914,
    "Hypophysis": 916,
    "Infundibulum": 917,

    "R-Globus-Pallidus": 913,
    "L-Globus-pallidus": 920,

    "Tuberal-Region": 922,

    "L-Med-Hypothalamus": 926,
    "R-Med-Hypothalamus": 936,

    "L-Lat-Hypothalamus": 927,
    "R-Lat-Hypothalamus": 937,

    "L-Ant-Hypothalamus": 928,
    "R-Ant-Hypothalamus": 938,

    "L-Post-Hypothalamus": 929,
    "R-Post-Hypothalamus": 939,
}



hyposubseg_labels =  (np.array(list(HYPVINN_CLASS_NAMES.keys())),
                    np.array([0, 1, 3, 4, 7, 9, 10,
                              11, 14, 16, 17, 13, 122,
                              226, 227, 228, 229]))

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

