import abc
from pathlib import Path
from typing import Tuple, Union, TYPE_CHECKING, Sequence, List, cast, Literal

if TYPE_CHECKING:
    import numpy as np
    import lapy
    import nibabel as nib
    import pandas as pd


MeasureTuple = Tuple[str, str, Union[int, float], str]


class AbstractMeasure(metaclass=abc.ABCMeta):

    def __int__(self, name: str, description: str, unit: str):
        self._name: str = name
        self._description: str = description
        self._unit: str = unit
        self.__value: int | float | None = None
        self.__token: str | None = None

    def as_tuple(self, subject_dir: Path, subject_id: str) -> MeasureTuple:
        return self._name, self._description, self(subject_dir, subject_id), self._unit

    @abc.abstractmethod
    def __call__(self, subject_dir: Path, subject_id: str) -> int | float:
        ...


class Measure(AbstractMeasure, metaclass=abc.ABCMeta):
    def __call__(self, subject_dir: Path, subject_id: str) -> int | float:
        self._subject_dir = subject_dir
        self._subject_id = subject_id
        token = str(subject_dir / subject_id)
        if self.__value is None or self.__token != token:
            self.__value = self._compute()
            self.__token = token
        return self.__value

    @abc.abstractmethod
    def _compute(self) -> int | float:
        ...


class SurfaceMeasure(Measure, metaclass=abc.ABCMeta):

    def __init__(self, surface_file: str, name: str, description: str, unit: str):
        self._surface_file = surface_file
        super().__init__(name, description, unit)

    def _read_mesh(self) -> "lapy.TriaMesh":
        import lapy
        return lapy.TriaMesh.read_fssurf(
            str(self._subject_dir / self._subject_id / "surf" / self._surface_file)
        )


class SurfaceHoles(SurfaceMeasure):
    def _compute(self) -> int:
        return int(1 - self._read_mesh().euler()/2)


class SurfaceVolume(SurfaceMeasure):
    def _compute(self) -> float:
        return self._read_mesh().volume()


class TableMeasure(Measure):

    def __init__(self, classes: Sequence[int], name: str, description: str,
                 unit: Literal["mm^3"] = "mm^3"):
        if unit != "mm^3":
            raise ValueError("unit must be mm^3 for TableMeasure!")
        self._classes = classes
        super().__init__(name, description, unit)
        self._pv_value = None
        self._legacy = True
        self._vox_vol = 0.

    @property
    def legacy_freesurfer(self) -> bool:
        return self._legacy

    @legacy_freesurfer.setter
    def legacy_freesurfer(self, v: bool):
        self._legacy = v

    @property
    def vox_vol(self) -> float:
        return self._vox_vol

    @vox_vol.setter
    def vox_vol(self, v: float):
        self._vox_vol = v

    def generate_virtual_label(self) -> List[int]:
        return list(self._classes)

    def update_data(self, value: "pd.Series"):
        self._pv_value = value

    def _compute(self) -> float:
        if self._pv_value is None:
            raise RuntimeError(f"The partial volume of {self._name} has not been "
                               f"updated in the TableMeasure object yet!")
        if self._legacy:
            out = self._pv_value["NVoxels"].item() * self._vox_vol
        else:
            out = self._pv_value["Volume_mm3"].item()
        self._pv_value = None
        return out


class VolumeMeasure(Measure, metaclass=abc.ABCMeta):

    def __init__(self, segfile: str,
                 name: str, description: str, unit: str):
        self._segfile = segfile
        super().__init__(name, description, unit)

    def _read_file(self) -> Tuple["nib.analyze.SpatialImage", "np.ndarray"]:
        try:
            img = cast(nib.analyze.SpatialImage, nib.load(self._segfile))
            if not isinstance(img, nib.analyze.SpatialImage):
                raise RuntimeError(f"Loading the file '{self._segfile}' for Measure "
                                   f"{self._name} was invalid, no SpatialImage.")
        except (IOError, FileNotFoundError) as e:
            args = e.args[0]
            raise IOError(
                f"Failed loading the file '{self._segfile}' for {self._name} "
                f"with error: {args}") from e
        data = np.asarray(img.dataobj)
        return img, data

    def _get_vox_vol(self, img: "nib.analyze.SpatialImage") -> float:
        return np.prod(img.header.get_zooms()).item()


class VoxelCount(VolumeMeasure):
    """Counts the voxels belonging to """

    def __init__(self, segfile: str, classes: Sequence[int], name: str,
                 description: str, unit: Literal["unitless", "mm^3"] = "unitless"):
        self._classes = classes
        if unit not in ["unitless", "mm^3"]:
            raise ValueError("unit must be either 'mm^3' or 'unitless' for VoxelCount")
        super().__init__(segfile, name, description, unit)

    def _compute(self) -> Union[int, float]:
        import numpy as np
        img, seg = self._read_file()
        vox_vol = 1 if self._unit == "unitless" else self._get_vox_vol(img)
        in_classes = np.logical_or.reduce((seg == c for c in self._classes), axis=0)
        return in_classes.sum(dtype=int).item() * vox_vol


class MaskVolume(VolumeMeasure):

    def _compute(self) -> float:
        """Load full image and compute the volume of voxels > 0.
        Returns
        -------
        float
            volume of voxels > 0
        """
        img, data = self._read_file()
        vox_vol = 1 if self._unit == "unitless" else self._get_vox_vol(img)
        return np.sum(np.greater(data, 0), dtype=int).item() * vox_vol


class DerivedMeasure(AbstractMeasure):

    def __init__(self,
                 parents: list[Tuple[float, Measure] | Measure],
                 name: str, description: str, unit: str):
        self._parents = [p if isinstance(p, tuple) else (1, p) for p in parents]
        super().__init__(name, description, unit)

    def __call__(self, subject_dir: Path, subject_id: str) -> int | float:
        import numpy as np
        return np.sum([s * m(subject_dir, subject_id) for s, m in self._parents])

__FILE_LOADER = {
    "ribbon": ReadFile("ribbon.mgz"),
    # FixAsegWithRibbon is use ribbonid, where asegid is in 2, 41, 3, 42, 0
    "asegfixed": FixAsegWithRibbon(),
}

MEASURES = {
    "lhSurfaceHoles": SurfaceHoles(
        "lh.orig.nofix", "lhSurfaceHoles",
        "Number of defect holes in lh surfaces prior to fixing",
        "unitless"),
    "rhSurfaceHoles": SurfaceHoles(
        "rh.orig.nofix", "rhSurfaceHoles",
        "Number of defect holes in rh surfaces prior to fixing",
        "unitless"),
    "lhPialTotal": SurfaceVolume(
        "lh.pial", "lhPialTotalVol", "Left hemisphere total pial volume", "mm^3"
    ),
    "rhPialTotal": SurfaceVolume(
        "rh.pial", "rhPialTotalVol", "Right hemisphere total pial volume", "mm^3"
    ),
    "lhWhiteMatterTotal": SurfaceVolume(
        "lh.white", "lhPialVol", "Left hemisphere total white matter volume", "mm^3"
    ),
    "rhWhiteMatterTotal": SurfaceVolume(
        "rh.white", "rhPialVol", "Right hemisphere total white matter volume", "mm^3"
    ),
    "Mask": MaskVolume(
        "brainmask.mgz", "MaskVol", "Mask Volume", "mm^3"
    ),
    "SupraTentorialRibbon": Ribbon(),
    # Left-Cerebellum-Cortex Right-Cerebellum-Cortex Cbm_Left_I_IV Cbm_Right_I_IV
    # Cbm_Left_V Cbm_Right_V Cbm_Left_VI Cbm_Vermis_VI Cbm_Right_VI Cbm_Left_CrusI
    # Cbm_Vermis_CrusI Cbm_Right_CrusI Cbm_Left_CrusII Cbm_Vermis_CrusII
    # Cbm_Right_CrusII Cbm_Left_VIIb Cbm_Vermis_VIIb Cbm_Right_VIIb Cbm_Left_VIIIa
    # Cbm_Vermis_VIIIa Cbm_Right_VIIIa Cbm_Left_VIIIb Cbm_Vermis_VIIIb Cbm_Right_VIIIb
    # Cbm_Left_IX Cbm_Vermis_IX Cbm_Right_IX Cbm_Left_X Cbm_Vermis_X Cbm_Right_X
    # Cbm_Vermis_VII Cbm_Vermis_VIII Cbm_Vermis
    "CerebellarGM": TableMeasure(
        (8, 47, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614,
         615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 630,
         631, 632),
        "CerebellarGMVol", "Cerebellar gray matter volume", "mm^3"
    ),
    # Left-Thalamus Right-Thalamus Left-Caudate Right-Caudate Left-Putamen Right-Putamen
    # Left-Pallidum Right-Pallidum Left-Hippocampus Right-Hippocampus Left-Amygdala
    # Right-Amygdala Left-Accumbens-area Right-Accumbens-area Left-VentralDC
    # Right-VentralDC Left-Substantia-Nigra Right-Substantia-Nigra
    "SubCortGray": TableMeasure(
        (10, 11, 12, 13, 17, 18, 26, 27, 28, 49, 50, 51, 52, 53, 54, 58, 59, 60),
        "SubCortGrayVol", "Subcortical gray matter volume", "mm^3"
    ),
    # 3rd-Ventricle 4th-Ventricle 5th-Ventricle CSF
    "TFFC": TableMeasure(
        (14, 15, 72, 24),
        "Third-Fourth-Fifth-CSF", "volume of 3rd, 4th, 5th ventricle and CSF", "mm^3"
    ),
    # Left-Choroid-Plexus Right-Choroid-Plexus Left-Lateral-Ventricle
    # Right-Lateral-Ventricle Left-Inf-Lat-Vent Right-Inf-Lat-Vent
    "VentricleChoroidVol": TableMeasure(
        (31, 63, 4, 43, 5, 44),
        "VentricleChoroidVol", "Volume of ventricles and choroid plexus", "mm^3"),
    "BrainSeg": AsegFixed(# not 0, not brainstem
        "BrainSegVol", "Brain Segmentation Volume", "mm^3"),
    "EstimatedTotalIntraCranialVol": eTIVMeasure(
        "transforms/talairach.lta",
        "eTIV", "Estimated Total Intracranial Volume", "mm^3"),

}

MEASURES.update({
    "BrainSegNotVent": DerivedMeasure(
        [MEASURES["BrainSeg"], (-1, MEASURES["VentricleChoroidVol"])],
        "BrainSegVolNotVent","Brain Segmentation Volume Without Ventricles", "mm^3"),
    "lhCortex": DerivedMeasure(
        [(1., MEASURES["lhPialTotal"]), (-1., MEASURES["lhWhiteMatterTotal"]),
         (-1, MEASURES["lhCortexRibbon"])],
        "lhCortexVol", "Left hemisphere cortical gray matter volume", "mm^3"
    ),
    "rhCortex": DerivedMeasure(
        [(1., MEASURES["rhPialTotal"]), (-1., MEASURES["rhWhiteMatterTotal"]),
         (-1, MEASURES["rhCortexRibbon"])],
        "rhCortexVol", "Right hemisphere cortical gray matter volume", "mm^3"),
    "lhCerebralWhiteMatter": DerivedMeasure(
        [MEASURES["lhWhiteMatterTotal"], (-1, MEASURES["lhWhiteMatterRibbon"])],
        "lhCerebralWhiteMatterVol", "Left hemisphere cerebral white matter volume",
        "mm^3"),
    "rhCerebralWhiteMatter": DerivedMeasure(
        [MEASURES["lhWhiteMatterTotal"], (-1, MEASURES["lhWhiteMatterRibbon"])],
        "rhCerebralWhiteMatterVol", "Right hemisphere cerebral white matter volume",
        "mm^3"),
    "SupraTentorial": DerivedMeasure(
        [MEASURES["lhPialTotal"], MEASURES["rhPialTotal"],
         (-1, MEASURES["SupraTentorialRibbon"])],
        "SupraTentorialVol", "Supratentorial volume", "mm^3"),
    "SurfaceHoles": DerivedMeasure(
        [MEASURES["rhSurfaceHoles"], MEASURES["lhSurfaceHoles"]],
        "SurfaceHoles", "Total number of defect holes in surfaces prior to fixing",
        "unitless"),
})

MEASURES.update({
    "Cortex": DerivedMeasure(
        [MEASURES["lhCortex"], MEASURES["rhCortex"]],
        "CortexVol", f"Total cortical gray matter volume", "mm^3"
    ),
    "CerebralWhiteMatter": DerivedMeasure(
        [MEASURES["rhCerebralWhiteMatter"], MEASURES["lhCerebralWhiteMatter"]],
        "CerebralWhiteMatterVol", "Total cerebral white matter volume", "mm^3"),
    "SupraTentorialNotVent": DerivedMeasure(
        [MEASURES["SupraTentorial"], (-1, MEASURES["VentricleChoroidVol"])],
        "SupraTentorialVolNotVent", "Supratentorial volume", "mm^3"
    ),


})

MEASURES["TotalGray"] = DerivedMeasure(
    [MEASURES["SubCortGray"], MEASURES["Cortex"], MEASURES["CerebellarGM"]],
    "TotalGrayVol", "Total gray matter volume", "mm^3"
)

ETIV_SCALE_FACTOR = 1948106.  # 1948.106 cm^3 * 1e3 mm^3/cm^3
MEASURES = {
    "BrainSegVoltoeTIV": ("BrainSegVol-to-eTIV",
                         "Ratio of BrainSegVol to eTIV", "unitless"),
    "MaskVoltoeTIV": ("MaskVol-to-eTIV", "Ratio of MaskVol to eTIV", "unitless"),
    "SupraTentorialNotVentVox": ("SupraTentorialVolNotVentVox",
                                "Supratentorial volume voxel count", "mm^3"),
}

# Measure BrainSeg = BrainSegVol
# Measure BrainSegNotVent = BrainSegVolNotVent (BrainSegVol-VentChorVol-TFFC)
# Measure VentricleChoroidVol = VentChorVol
# Measure lhCortex (lhpialvolTot - lhwhitevolTot - lhCtxGMCor)
# Measure rhCortex (rhpialvolTot - rhwhitevolTot - rhCtxGMCor)
# Measure lhCerebralWhiteMatter
# Measure rhCerebralWhiteMatter
# Measure SubCortGray = SubCortGMVol
## Measure TotalGray = TotalGMVol
# SupraTentVolCor = SupraTentorialVolCorrection(aseg, ribbon);
# Measure SupraTentorial = SupraTentVol (lhpialvolTot + rhpialvolTot + SupraTentVolCor)
# Measure SupraTentorialNotVent = SupraTentVolNotVent (SupraTentVol - VentChorVol)
# Measure Mask = MaskVol
# Measure BrainSegVol-to-eTIV
# Measure MaskVol-to-eTIV
# Measure lhSurfaceHoles = (1-lheno/2) -- Euler number of /surf/lh.orig.nofix
# Measure rhSurfaceHoles = (1-rheno/2)
# Measure EstimatedTotalIntraCranialVol

# SupraTentorialNotVentVox
# BrainSegNotVentSurf
