import abc
import logging
import re
from concurrent.futures import Executor
from contextlib import contextmanager
from pathlib import Path
from typing import (Tuple, Union, TYPE_CHECKING, Sequence, List, cast, Literal,
                    Iterable, Callable, Optional, Dict, overload, TextIO, Protocol,
                    TypeVar, Generic, get_args)


import numpy as np
if TYPE_CHECKING:
    from numpy import typing as npt
    import lapy
    import nibabel as nib
    import pandas as pd
    from concurrent.futures import Future

    from CerebNet.datasets.utils import LTADict

MeasureTuple = Tuple[str, str, Union[int, float], str]
ImageTuple = Tuple["nib.analyze.SpatialImage", "np.ndarray"]
MeasureString = Union["Measure", str]
AnyBufferType = Union[dict[str, MeasureTuple], ImageTuple, "lapy.TriaMesh",
                      "npt.NDArray[float]"]
T_BufferType = TypeVar("T_BufferType",
                       bound=(ImageTuple | dict[str, MeasureTuple] | "lapy.TriaMesh" |
                              "np.ndarray"))
DerivedAggOperation = Literal["sum", "ratio"]
AnyMeasure = Union["AbstractMeasure", str]
PVMode = Literal["vox", "pv"]
ClassesType = Sequence[int]
ClassesOrCondType = ClassesType | Callable[["npt.NDArray[int]"], "npt.NDArray[bool]"]
MaskSign = Literal["abs", "pos", "neg"]
_ToBoolCallback = Callable[["npt.NDArray[int]"], "npt.NDArray[bool]"]


class ReadFileHook(Protocol[T_BufferType]):

    @overload
    def __call__(self, file: Path, blocking: True = True) -> T_BufferType: ...

    @overload
    def __call__(self, file: Path, blocking: False) -> None: ...

    def __call__(self, file: Path, b: bool = True) -> Optional[T_BufferType]: ...


def read_measure_file(path: Path) -> Dict[str, MeasureTuple]:
    if not path.exists():
        raise IOError(f"Measures could not be imported from {path}, "
                      f"the file does not exist.")
    with open(path, "r") as fp:
        lines = list(fp.readlines())
    lines = filter(lambda l: l.startswith("# Measure "), lines)

    def to_measure(line: str) -> Tuple[str, MeasureTuple]:
        data = line.removeprefix("# Measure ").strip()
        import re
        key, name, desc, sval, unit = re.split("\\s*,\\s*", data)
        value = float(sval) if "." in sval else int(sval)
        return key, (name, desc, value, unit)

    return dict(map(to_measure, lines))


def read_volume_file(path: Path) -> ImageTuple:
    """Read a volume from disk."""
    try:
        import nibabel as nib
        img = cast(nib.analyze.SpatialImage, nib.load(path))
        if not isinstance(img, nib.analyze.SpatialImage):
            raise RuntimeError(f"Loading the file '{path}' for Measure was invalid, "
                               f"no SpatialImage.")
    except (IOError, FileNotFoundError) as e:
        args = e.args[0]
        raise IOError(
            f"Failed loading the file '{path}' with error: {args}") from e
    data = np.asarray(img.dataobj)
    return img, data


def read_mesh_file(path: Path) -> "lapy.TriaMesh":
    """Read a mesh from disk."""
    try:
        import lapy
        mesh = lapy.TriaMesh.read_fssurf(str(path))
    except (IOError, FileNotFoundError) as e:
        args = e.args[0]
        raise IOError(
            f"Failed loading the file '{path}' with error: {args}") from e
    return mesh


def read_lta_transform_file(path: Path) -> "npt.NDArray[float]":
    """Read and extract the first lta transform from an LTA file.

    Parameters
    ----------
    path : Path
        Path of the LTA file

    Returns
    -------
    matrix : npt.NDArray[float]
        Matrix of shape (4, 4)
    """
    from CerebNet.datasets.utils import read_lta
    return read_lta(path)["lta"][0, 0]


def read_xfm_transform_file(path: Path) -> "npt.NDArray[float]":
    """Read Talairach transform.

    Parameters
    ----------
    path : str | Path
        Filename to transform

    Returns
    -------
    tal
        Talairach transform matrix

    Raises
    ------
    ValueError
        if the file is of an invalid format.

    """
    with open(path) as f:
        lines = f.readlines()

    try:
        transf_start = [l.lower().startswith("linear_") for l in lines].index(True) + 1
        tal_str = [l.replace(";", " ") for l in lines[transf_start:transf_start + 3]]
        tal = np.genfromtxt(tal_str)
        tal = np.vstack([tal, [0, 0, 0, 1]])

        return tal
    except Exception as e:
        err = ValueError(f"Could not find taiairach transform in {path}.")
        raise err from e


def read_transform_file(path: Path) -> "npt.NDArray[float]":
    """Read any transform file."""
    if path.suffix == ".lta":
        return read_lta_transform_file(path)
    elif path.suffix == ".xfm":
        return read_xfm_transform_file(path)
    else:
        raise NotImplementedError(
            f"The extension {path.suffix} is not '.xfm' or '.lta' and not recognized.")


def mask_in_array(arr: "npt.NDArray", items: "npt.ArrayLike") -> "npt.NDArray[bool]":
    """Efficient function to generate a mask of elements in `arr`, which are also in
    items.

    Parameters
    ----------
    arr : npt.NDArray
        array with data, most likely int
    items : npt.ArrayLike
        which items in arr should yield True

    Returns
    -------
    mask : npt.NDArray[bool]
        an array, true, where elements in arr are in items
    """
    _items = np.asarray(items)
    if _items.size == 0:
        return np.zeros_like(arr, dtype=bool)
    elif _items.size == 1:
        return np.asarray(arr == _items.flat[0])
    else:
        max_index = max(np.max(items), np.max(arr))
        if max_index >= 2 ** 16:
            logging.getLogger(__name__).warning(
                f"labels in arr are larger than {2 ** 16 - 1}, this is not recommended!"
            )
        lookup = np.zeros(max_index + 1, dtype=bool)
        lookup[_items] = True
        return lookup[arr]


def mask_not_in_array(arr: "npt.NDArray", items: "npt.ArrayLike") -> "npt.NDArray[bool]":
    """Inverse of mask_in_array

    Parameters
    ----------
    arr : npt.NDArray
        array with data, most likely int
    items : npt.ArrayLike
        which items in arr should yield True

    Returns
    -------
    mask : npt.NDArray[bool]
        an array, false, where elements in arr are in items
    """
    _items = np.asarray(items)
    if _items.size == 0:
        return np.ones_like(arr, dtype=bool)
    elif _items.size == 1:
        return np.asarray(arr != _items.flat[0])
    else:
        max_index = max(np.max(items), np.max(arr))
        if max_index >= 2 ** 16:
            logging.getLogger(__name__).warning(
                f"labels in arr are larger than {2 ** 16 - 1}, this is not recommended!"
            )
        lookup = np.ones(max_index + 1, dtype=bool)
        lookup[_items] = False
        return lookup[arr]


class AbstractMeasure(metaclass=abc.ABCMeta):

    __PATTERN = re.compile("^([^\s=]+)\s*=\s*(\S.*)$")

    def __init__(self, name: str, description: str, unit: str):
        self._name: str = name
        self._description: str = description
        self._unit: str = unit
        self._subject_dir: Path | None = None

    def as_tuple(self) -> MeasureTuple:
        return self._name, self._description, self(), self._unit

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def unit(self) -> str:
        return self._unit

    @property
    def subject_dir(self) -> Path:
        return self._subject_dir

    @abc.abstractmethod
    def __call__(self) -> int | float:
        ...

    def read_subject(self, subject_dir: Path) -> bool:
        """
        Perform IO required to compute/fill the Measure.

        Parameters
        ----------
        subject_dir : Path
            path to the directory of the subject_dir (often subject_dir/subject_id)

        Returns
        -------
        bool
            whether there was an update
        """
        updated = subject_dir != self.subject_dir
        if updated:
            self._subject_dir = subject_dir
        return updated

    @abc.abstractmethod
    def _parsable_args(self) -> list[str]: ...

    def set_args(self, **kwargs: str) -> None:
        if len(kwargs) > 0:
            raise ValueError(f"Invalid args {tuple(kwargs.keys())}")

    def parse_args(self, *args: str):
        """Parse additional args defining the behavior of the Measure."""
        _pargs = self._parsable_args()
        if not (0 <= len(args) < len(_pargs)):
            raise ValueError(f"The measure {self.name} can have up to {len(_pargs)} "
                             f"arguments, but parsing {len(args)}: {args}.")

        def kwerror(i, args, msg) -> RuntimeError:
            return RuntimeError(f"Error parsing arg {i} in {args}: {msg}")
        _kwargs = {}
        _kwmode = False
        for i, (arg, default_key) in enumerate(zip(args, _pargs)):
            if (hit := self.__PATTERN.match(arg)) is None:
                # non-keyword mode
                if _kwmode:
                    raise kwerror(i, args, f"non-keyword after keyword")
                _kwargs[default_key] = arg
            else:
                # keyword mode
                _kwmode = True
                k = hit.group(1)
                if k in _kwargs:
                    raise kwerror(i, args, f"keyword '{k}' already assigned")
                if k not in _pargs:
                    raise kwerror(i, args, f"keyword '{k}' not in {_pargs}")
                _kwargs[k] = hit.group(2)

    def help(self) -> str:
        return f"{self.name}="

    @abc.abstractmethod
    def __str__(self) -> str:
        ...


class NullMeasure(AbstractMeasure):

    def _parsable_args(self) -> list[str]:
        return []

    def __call__(self) -> int | float:
        return 0 if self.unit == "unitless" else 0.0

    def help(self) -> str:
        return super().help() + "NULL"

    def __str__(self) -> str:
        return "NullMeasure()"


class Measure(AbstractMeasure, Generic[T_BufferType], metaclass=abc.ABCMeta):
    """Class to buffer computed values, buffers computed values. Implements a value
    buffering interface for computed measure values and implement the read_subject
    pattern."""

    __buffer: float | int | None = None
    __token: str = ""
    __PATTERN = re.compile("^([^\s=]*file)\s*=\s*(\S.*)$")

    def __call__(self) -> int | float:
        token = str(self._subject_dir)
        if self.__buffer is None or self.__token != token:
            self.__token = token
            self.__buffer = self._compute()
        return self.__buffer

    @abc.abstractmethod
    def _compute(self) -> int | float:
        ...

    def __init__(self, file: Path, name: str, description: str, unit: str,
                 read_hook: ReadFileHook[T_BufferType]):
        self._file = file
        self._callback = read_hook
        self._data: Optional[T_BufferType] = None
        super().__init__(name, description, unit)

    def _load_error(self, name: str = "data") -> RuntimeError:
        return RuntimeError(f"The '{name}' is not available yet for {self.name} "
                            f"({self.__class__.__name__}), has the subject been loaded "
                            f"or the cache invalidated, but not a new subject loaded.")
    def _filename(self) -> Path:
        return self._subject_dir / self._file

    def read_subject(self, subject_dir: Path) -> bool:
        """
        Perform IO required to compute/fill the Measure. Delegates file reading to
        read_hook (set in __init__).

        Parameters
        ----------
        subject_dir : Path
            path to the directory of the subject_dir (often subject_dir/subject_id)

        Returns
        -------
        bool
            whether there was an update
        """
        if super().read_subject(subject_dir):
            self._data = self._callback(self._filename())
            return True
        return False

    def _parsable_args(self) -> list[str]:
        return ["file"]

    def set_args(self, file: str | None = None, **kwargs: str) -> None:
        if file is not None:
            self._file = Path(file)
        return super().set_args(**kwargs)

    def __str__(self) -> str:
        return f"{type(self).__name__}(file={self._file})"


class ImportedMeasure(Measure[dict[str, MeasureTuple]]):

    read_file = staticmethod(read_measure_file)

    def __init__(self, key: str, measurefile: Path, name: str = "N/A",
                 description: str = "N/A", unit: str = "unitless",
                 read_file: Optional[ReadFileHook[Dict[str, MeasureTuple]]] = None):
        self._key: str = key
        super().__init__(measurefile, name, description, unit,
                         self.read_file if read_file is None else read_file)

    def _compute(self) -> int | float:
        """
        Will also update the name, description and unit from the strings in the file.

        Returns
        -------
        value : int | float
            value of the measure (as read from the file)
        """
        self._name, self._description, out, self._unit = self._data[self._key]
        return out

    def _parsable_args(self) -> list[str]:
        return ["key", "measurefile"]

    def set_args(self, key: str | None = None,
                 measurefile: str | None = None, **kwargs: str) -> None:
        if measurefile is not None:
            kwargs["file"] = measurefile
        if key is not None:
            self._key = key
        return super().set_args(**kwargs)

    def help(self) -> str:
        return super().help() + f"imported from {self._file}"

    def __str__(self) -> str:
        return f"ImportedMeasure(key={self._key}, measurefile={self._file})"


class SurfaceMeasure(Measure["lapy.TriaMesh"], metaclass=abc.ABCMeta):
    """Class to implement default Surface io."""

    read_file = staticmethod(read_mesh_file)

    def __init__(self, surface_file: Path, name: str, description: str, unit: str,
                 read_mesh: Optional[ReadFileHook["lapy.TriaMesh"]] = None):
        super().__init__(surface_file, name, description, unit,
                         self.read_file if read_mesh is None else read_mesh)

    def __str__(self) -> str:
        return f"{type(self).__name__}(surface_file={self._file})"

    def _parsable_args(self) -> list[str]:
        return ["surface_file"]

    def set_args(self, surface_file: str | None = None, **kwargs: str) -> None:
        if surface_file is not None:
            kwargs["file"] = surface_file
        return super().set_args(**kwargs)


class SurfaceHoles(SurfaceMeasure):
    """Class to compute surfaces holes for surfaces."""

    def _compute(self) -> int:
        return int(1 - self._data.euler() / 2)

    def help(self) -> str:
        return super().help() + f"surface holes from {self._file}"


class SurfaceVolume(SurfaceMeasure):
    """Class to compute surface volume for surfaces."""

    def _compute(self) -> float:
        return self._data.volume()

    def help(self) -> str:
        return super().help() + f"volume from {self._file}"


class PVMeasure(AbstractMeasure):
    """Class to compute volume for segmentations (includes PV-correction)."""

    read_file = None

    def __init__(self, classes: ClassesType, name: str, description: str,
                 unit: Literal["mm^3"] = "mm^3"):
        if unit != "mm^3":
            raise ValueError("unit must be mm^3 for PVMeasure!")
        self._classes = classes
        super().__init__(name, description, unit)
        self._pv_value = None
        self._vox_vol = 0.

    @property
    def vox_vol(self) -> float:
        return self._vox_vol

    @vox_vol.setter
    def vox_vol(self, v: float):
        self._vox_vol = v

    def labels(self) -> List[int]:
        return list(self._classes)

    def update_data(self, value: "pd.Series"):
        self._pv_value = value

    def __call__(self) -> float:
        if self._pv_value is None:
            raise RuntimeError(f"The partial volume of {self._name} has not been "
                               f"updated in the PVMeasure object yet!")
        if self.unit == "unitless":
            vox_vol, col = 1, "NVoxels"
        else:
            vox_vol, col = self._vox_vol, "Volume_mm3"
        out = self._pv_value[col].item() * vox_vol
        # TODO: release table, no buffering yet
        # self._pv_value = None
        return out

    def _parsable_args(self) -> list[str]:
        return ["classes"]

    def set_args(self, classes: str | None = None, **kwargs: str) -> None:
        if classes is not None:
            self._classes = classes
        return super().set_args(**kwargs)

    def __str__(self) -> str:
        return f"PVMeasure(classes={list(self._classes)})"

    def help(self) -> str:
        return super().help() + (f"partial volume of {format_classes(self._classes)} "
                                 f"in seg file")


def format_classes(_classes: Iterable[int]) -> str:
    """format an iterable of classes."""
    if not isinstance(_classes, Iterable):
        return str(_classes)
    sorted_list = list(sorted(_classes))
    if len(sorted_list) == 0:
        return "()"
    prev = ""
    out = str(sorted_list[0])

    from itertools import pairwise
    for a, b in pairwise(sorted_list):
        if a != b - 1:
            out += f"{prev},{b}"
            prev = ""
        else:
            prev = f"-{b}"

    return out + prev


class VolumeMeasure(Measure[ImageTuple]):
    """Counts Voxels belonging to a class or condition."""

    read_file = staticmethod(read_volume_file)

    def __init__(self, segfile: Path, classes_or_cond: ClassesOrCondType, name: str,
                 description: str, unit: Literal["unitless", "mm^3"] = "unitless",
                 read_file: Optional[ReadFileHook[ImageTuple]] = None):
        if callable(classes_or_cond):
            self._classes: Optional[ClassesType] = None
            self._cond: _ToBoolCallback = classes_or_cond
        else:
            if len(classes_or_cond) == 0:
                raise ValueError(f"No operation passed to {type(self).__name__}.")
            self._classes = classes_or_cond
            from functools import partial
            self._cond = partial(mask_in_array, items=self._classes)
        if unit not in ["unitless", "mm^3"]:
            raise ValueError("unit must be either 'mm^3' or 'unitless' for " +
                             type(self).__name__)
        super().__init__(segfile, name, description, unit,
                         self.read_file if read_file is None else read_file)

    def _get_vox_vol(self) -> float:
        return np.prod(self._data[0].header.get_zooms()).item()

    def _compute(self) -> int | float:
        if not isinstance(self._data, tuple) or len(self._data) != 2:
            raise self._load_error("data")
        vox_vol = 1 if self._unit == "unitless" else self._get_vox_vol()
        return np.sum(self._cond(self._data[1]), dtype=int).item() * vox_vol

    def _parsable_args(self) -> list[str]:
        return ["segfile", "classes"]

    def set_args(self, segfile: str | None = None,
                 classes: str | None = None, **kwargs: str) -> None:
        if segfile is not None:
            kwargs["file"] = segfile
        if classes is not None:
            _classes = re.split("\s+", classes.lstrip("[ ").rstrip("] "))
            self._classes = list(map(int, _classes))
            from functools import partial
            self._cond = partial(mask_in_array, items=self._classes)
        return super().set_args(**kwargs)

    def __str__(self) -> str:
        return f"{type(self).__name__}(segfile={self._file}, {self._param_string()})"

    def help(self) -> str:
        return f"{self._name}={self._param_help()} in {self._file}"

    def _param_help(self, prefix: str = ""):
        cond = getattr(self, prefix + "_cond")
        classes = getattr(self, prefix + "_classes")
        return prefix + (f"cond={cond}" if classes is None else format_classes(classes))

    def _param_string(self, prefix: str = ""):
        cond = getattr(self, prefix + "_cond")
        classes = getattr(self, prefix + "_classes")
        return prefix + (f"cond={cond}" if classes is None else f"classes={classes}")


class MaskMeasure(VolumeMeasure):

    def __init__(self, maskfile: Path, name: str, description: str,
                 unit: Literal["unitless", "mm^3"] = "unitless",
                 threshold: float = 0.5,
                 # sign: MaskSign = "abs", frame: int = 0,
                 # erode: int = 0, invert: bool = False,
                 read_file: Optional[ReadFileHook[ImageTuple]] = None):
        self._threshold: float = threshold
        # self._sign: MaskSign = sign
        # self._invert: bool = invert
        # self._frame: int = frame
        # self._erode: int = erode
        super().__init__(maskfile, self.mask, name, description, unit, read_file)

    def mask(self, data: "npt.NDArray[int]") -> "npt.NDArray[bool]":
        """Generates a mask from data similar to mri_binarize + erosion."""
        # if self._sign == "abs":
        #     data = np.abs(data)
        # elif self._sign == "neg":
        #     data = -data
        out = np.greater(data, self._threshold)
        # if self._invert:
        #     out = np.logical_not(out)
        # if self._erode != 0:
        #     from scipy.ndimage import binary_erosion
        #     binary_erosion(out, iterations=self._erode, output=out)
        return out

    def set_args(self, maskfile: Path | None = None,
                 threshold: float | None = None,
                 # invert: bool | None = None, sign: MaskSign | None = None,
                 # erode: int | None = None, frame: int | None = None,
                 **kwargs: str) -> None:
        # if sign is not None and sign not in get_args(MaskSign):
        #     raise ValueError(f"{sign} is not a valid sign from {get_args(MaskSign)}.")
        # if sign is not None:
        #     self._sign = sign
        if threshold is not None:
            self._threshold = float(threshold)
        # if invert is not None:
        #     self._invert = bool(invert)
        # if erode is not None:
        #     self._erode = int(erode)
        # if frame is not None:
        #     self._frame = int(frame)
        #     if self._frame != 0:
        #         raise NotImplementedError("Frames not equal to 0 are not supported")
        return super().set_args(**kwargs)

    def _parsable_args(self) -> list[str]:
        return ["maskfile", "threshold",
                # "sign", "invert", "erode", "frame"
                ]

    def __str__(self) -> str:
        return (f"{type(self).__name__}(maskfile={self._file}, "
                f"threshold={self._threshold}"
                # f", sign={self._sign}, invert={self._invert}, erode={self._erode}"
                f")")

    def _param_help(self, prefix: str = ""):
        # sign = {"pos": "%f", "neg": "- %f", "abs": "abs(%f)"}[self._sign]
        # invert = "not " if self._invert else ""
        # erosion_text = ""
        # if self._erode > 0:
        #     erosion_text = f" eroded {self._erode} times"
        # return f"{invert}voxel > {sign % self._threshold}{erosion_text}"
        return f"voxel > {self._threshold}"


class MultiVolumeMeasure(VolumeMeasure):

    def __init__(self, segfile: Path, other_file: Path,
                 classes_or_cond: ClassesOrCondType, name: str, description: str,
                 unit: Literal["unitless", "mm^3"] = "unitless",
                 read_file: Optional[ReadFileHook[ImageTuple]] = None,
                 other_classes_or_cond: ClassesOrCondType = (0,)):
        self._other_file = other_file
        self._other_data = None
        super().__init__(segfile, classes_or_cond, name, description, unit, read_file)
        if callable(other_classes_or_cond):
            self._other_classes: Optional[ClassesType] = None
            self._other_cond: Callable[["npt.NDArray[int]"],
            "npt.NDArray[bool]"] = other_classes_or_cond
        else:
            if len(other_classes_or_cond) == 0:
                raise ValueError(f"No other_classes passed to {type(self).__name__}.")
            self._other_classes = other_classes_or_cond
            from functools import partial
            self._other_cond = partial(mask_in_array, items=self._other_classes)

    def read_subject(self, subject_dir: Path) -> bool:
        """
        Perform IO required to compute/fill the Measure. Delegates file reading to
        read_hook (set in __init__).

        Parameters
        ----------
        subject_dir : Path
            path to the directory of the subject_dir (often subject_dir/subject_id)

        Returns
        -------
        bool
            whether there was an update
        """
        if super().read_subject(subject_dir):
            self._other_data = self._callback(self._other_filename())
            return True
        return False

    def _other_filename(self) -> Path:
        return self._subject_dir / self._other_file

    def _compute(self) -> int | float:
        if not isinstance(self._data, tuple) or len(self._data) != 2:
            raise self._load_error("data")

        if not isinstance(self._other_data, tuple) or len(self._other_data) != 2:
            raise self._load_error("other data")

        if not np.all(np.isclose(self._data[0].affine, self._other_data[0].affine)):
            raise RuntimeError(f"The two images {self._filename()} and "
                               f"{self._other_filename()} do not share the same "
                               f"affines.")

        # duplicate the seg-data
        self._data = self._data[0], self._data[1].copy()
        # only count regions that are not in the background for the ribbon
        # freesurfer/utils/cma.cpp#504
        mask = self._other_cond(self._other_data[1])
        self._data[1][mask] = self._fill_data(mask)
        self._other_data = None
        # compute the volume of operation in volume
        return self._compute()

    def _fill_data(self, mask: "npt.NDArray[bool]") -> "npt.ArrayLike":
        return 0

    def _parsable_args(self) -> list[str]:
        return ["segfile", "other_file", "operation", "other_classes"]

    def set_args(self, other_file: str | None = None,
                 other_classes: str | None = None, **kwargs: str) -> None:
        if other_file is not None:
            self._other_file = other_file
        if other_classes is not None:
            _classes = re.split("\s+", other_classes.lstrip("[ ").rstrip("] "))
            self._other_classes = list(map(int, _classes))
            from functools import partial
            self._other_cond = partial(mask_in_array, items=self._classes)
        return super().set_args(**kwargs)

    def help(self) -> str:
        return super().help() + (f"and {self._param_help('_other')} in "
                                 f"{self._other_file}")

    def __str__(self) -> str:
        return (f"{type(self).__name__}(segfile={self._file}, other_file="
                f"{self._other_file}, {self._param_string()}, "
                f"{self._param_string('_other')})")


AnyParentsTuple = Tuple[float, AnyMeasure]
ParentsTuple = Tuple[float, AnyMeasure]


class TransformMeasure(Measure, metaclass=abc.ABCMeta):

    read_file = staticmethod(read_transform_file)

    def __init__(self, lta_file: Path, name: str, description: str, unit: str,
                 read_lta: Optional[ReadFileHook["npt.NDArray[float]"]] = None):
        super().__init__(lta_file, name, description, unit,
                         self.read_file if read_lta is None else read_lta)

    def _parsable_args(self) -> list[str]:
        return ["lta_file"]

    def set_args(self, lta_file: str | None = None, **kwargs: str) -> None:
        if lta_file is not None:
            kwargs["file"] = lta_file
        return super().set_args(**kwargs)

    def __str__(self) -> str:
        return f"{type(self).__name__}(lta_file={self._file})"


class ETIVMeasure(TransformMeasure):
    """
    Compute the eTIV based on the freesurfer talairach registration and lta.

    Notes
    -----
    Reimplemneted from freesurfer/mri_sclimbic_seg
    https://github.com/freesurfer/freesurfer/blob/
    3296e52f8dcffa740df65168722b6586adecf8cc/mri_sclimbic_seg/mri_sclimbic_seg#L627
    """

    def __init__(self, lta_file: Path, name: str, description: str, unit: str,
                 read_lta: Optional[ReadFileHook["LTADict"]] = None,
                 etiv_scale_factor: float | None = None):
        if etiv_scale_factor is None:
            self._etiv_scale_factor = 1948106.  # 1948.106 cm^3 * 1e3 mm^3/cm^3
        else:
            self._etiv_scale_factor = etiv_scale_factor
        super().__init__(lta_file, name, description, unit, read_lta)

    def _parsable_args(self) -> list[str]:
        return super()._parsable_args() + ["etiv_scale_factor"]

    def set_args(self, etiv_scale_factor: str | None = None, **kwargs: str) -> None:
        if etiv_scale_factor is not None:
            self._etiv_scale_factor = float(etiv_scale_factor)
        return super().set_args(**kwargs)

    def _compute(self) -> float:
        # this scale factor is a fixed number derived by freesurfer
        return self._etiv_scale_factor / np.linalg.det(self._data).item()

    def help(self) -> str:
        return super().help() + f"eTIV from {self._file}"

    def __str__(self) -> str:
        return (f"{type(self).__name__}(lta_file={self._file}, etiv_scale_factor="
                f"{self._etiv_scale_factor})")


class DerivedMeasure(AbstractMeasure):

    def __init__(self,
                 parents: Iterable[Tuple[float, AnyMeasure] | AnyMeasure],
                 name: str, description: str, unit: str = "from parents",
                 operation: DerivedAggOperation = "sum",
                 measure_host: Optional[dict[str, AbstractMeasure]] = None):
        """
        Create the Measure, which depends on other measures, called parent measures.

        Parameters
        ----------
        parents : Iterable[tuple[float, AbstractMeasure] | AbstractMeasure]
            Iterable of either the measures (or a tuple of a float and a measure), the
            float is the factor by which the value of the respective measure gets
            weighted
            and defaults to 1.
        name : str
            Name of the Measure
        description : str
            Description text of the measure
        unit : str, optional
            Unit of the measure, typically 'mm^3' or 'unitless', autogenerated from
            parents' unit.
        operation : "sum" or "ratio", optional
            how to aggregate multiple `parents`, default = 'sum'
            'ratio' only supports exactly 2 parents.
        measure_host : dict[str, AbstractMeasure], optional
            a dict-like to provide AbstractMeasure objects for strings
        """

        def to_tuple(value: Tuple[float, AnyMeasure] | AnyMeasure) \
                -> Tuple[float, AnyMeasure]:
            if isinstance(value, Sequence) and not isinstance(value, str):
                if len(value) != 2:
                    raise ValueError("A tuple was not length 2.")
                factor, measure = value
            else:
                factor, measure = 1., value

            if not isinstance(measure, (str, AbstractMeasure)):
                raise ValueError(f"Expected a str or AbstractMeasure, not "
                                 f"{type(measure).__name__}!")
            if not isinstance(factor, float):
                factor = float(factor)
            return factor, measure

        self._parents: list[AnyParentsTuple] = [to_tuple(p) for p in parents]
        if len(self._parents) == 0:
            raise ValueError("No parents defined in DerivedMeasure.")
        self._measure_host = measure_host
        if operation in ("sum", "ratio"):
            self._operation: DerivedAggOperation = operation
        else:
            raise ValueError("operation must be 'sum' or 'ratio'.")
        super().__init__(name, description, unit)

    @property
    def unit(self):
        """Property to access the unit attribute, also implements auto-generation of
        unit, if the stored unit is 'from parents'."""
        if self._unit == "from parents":
            units = list(map(lambda x: x[1].unit, self.parents))
            if self._operation == "sum":
                if len(units) == 0:
                    raise ValueError("DerivedMeasure has no parent measures.")
                elif len(units) == 1 or all(units[0] == u for u in units[1:]):
                    return units[0]
            elif self._operation == "ratio":
                if len(units) != 2:
                    raise self.invalid_len_ratio()
                elif units[0] == units[1]:
                    return "unitless"
                elif units[1] == "unitless":
                    return units[0]
            raise RuntimeError(
                f"unit is set to auto-generate from parents, but the "
                f"parents' units are not consistent: {units}!"
            )
        else:
            return super().unit

    def invalid_len_ratio(self) -> ValueError:
        return ValueError(f"Invalid number of parents ({len(self._parents)}) for "
                          f"operation 'ratio'.")

    @property
    def parents(self) -> Iterable[AbstractMeasure]:
        """Iterable of the measures this measure depends on."""
        return (p for _, p in self.parents_items())

    def parents_items(self) -> Iterable[Tuple[float, AbstractMeasure]]:
        """Iterable of the measures this measure depends on."""
        return ((f, self._measure_host[p] if isinstance(p, str) else p)
                for f, p in self._parents)

    def __read_subject(self, subject_dir: Path) -> bool:
        """Default implementation for the read_subject_on_parents function hook."""
        return any(m.read_subject(subject_dir) for m in self.parents)

    @property
    def read_subject_on_parents(self) -> Callable[[Path], bool]:
        """read_subject_on_parents function hook property"""
        if (self._measure_host is not None and
                hasattr(self._measure_host, "read_subject_parents")):
            from functools import partial
            return partial(self._measure_host.read_subject_parents, self.parents)
        else:
            return self.__read_subject

    def read_subject(self, subject_dir: Path) -> bool:
        """
        Perform IO required to compute/fill the Measure. Will trigger the
        read_subject_on_parents function hook to populate the values of parent measures.

        Parameters
        ----------
        subject_dir : Path
            path to the directory of the subject_dir (often subject_dir/subject_id)

        Returns
        -------
        bool
            whether there was an update

        Notes
        -----
        Might trigger a race condition if the function hook `read_subject_on_parents`
        depends on this method finishing first, e.g. because of thread availability.
        """
        if super().read_subject(subject_dir):
            return self.read_subject_on_parents(self._subject_dir)
        return False

    def __call__(self) -> int | float:
        values = [s * m() for s, m in self.parents_items()]
        if self._operation == "sum":
            return np.sum(values)
        else:  # operation == "ratio"
            if len(self._parents) != 2:
                raise self.invalid_len_ratio()
            return values[0] / values[1]

    def _parsable_args(self) -> list[str]:
        return ["parents", "operation"]

    def set_args(self, parents: str | None = None, operation: str | None = None, **kwargs: str) -> None:
        if parents is not None:
            pat = re.compile("^(\d+\.?\d*\s+)?(\s.*)")
            stripped = parents.lstrip("[ ").rstrip("] ")

            def parse(p: str) -> Tuple[float, str]:
                hit = pat.match(p)
                if hit is None:
                    return 1., p
                return 1. if hit.group(1).strip() else float(hit.group(1)), hit.group(2)
            self._parents = list(map(parse, re.split("\s+", stripped)))
        if operation is not None:
            from typing import get_args as args
            if operation in args(DerivedAggOperation):
                self._operation = operation
            else:
                raise ValueError(f"operation can only be {args(DerivedAggOperation)}")
        return super().set_args(**kwargs)

    def __str__(self) -> str:
        return f"DerivedMeasure(parents={self._parents}, operation={self._operation})"

    def help(self) -> str:
        sign = {True: "+", False: "-"}

        def format_factor(f: float) -> str:
            return f"{sign[f >= 0]} " + ((str(abs(f)) + " ") if abs(f) != 1. else '')

        def format_parent(measure: str | AnyMeasure) -> str:
            if isinstance(measure, str):
                measure = self._measure_host[measure]
            return measure if isinstance(measure, str) else measure.help()
        if self._operation == "sum":
            par = "".join(f" {format_factor(f)}({format_parent(p)})"
                          for f, p in self._parents)
            return par.lstrip(' +')
        elif self._operation == "ratio":
            f = self._parents[0][0] * self._parents[1][0]
            return (f" {sign[f >= 0]} {format_factor(f)} (" +
                    " / ".join(format_parent(p[1]) for p in self._parents) + ")")


class MeasurePipeline(dict[str, AbstractMeasure]):
    _PATTERN_NO_ARGS = re.compile("^\s*([^(]+?)\s*$")
    _PATTERN_ARGS = re.compile("^\s*([^(]+)\(\s*([^)]*)\s*\)\s*$")
    _PATTERN_DELIM = re.compile("\s*,\s*")

    def __init__(self,
                 computed_measures: Iterable[str] = (),
                 imported_measures: Iterable[str] = (),
                 measurefile: Optional[Path] = None,
                 on_missing: Literal["fail", "skip", "fill"] = "fail",
                 executor: Optional[Executor] = None,
                 legacy: bool = False):
        """

        Parameters
        ----------
        imported_measures : Iterable[str], optional
            an iterable listing the measures to import
        measurefile : Path, optional
            path to the file to import measures from (other stats file, absolute or
            relative to subject_dir).
        on_missing : Literal["fail", "skip", "fill"], optional
            behavior to follow if a requested measure does not exist in path.
        executor : concurrent.futures.Executor, optional
            thread pool to parallelize io
        legacy : bool, optional
            use legacy freesurfer algorithms and statistics (default: off)
        """

        super().__init__()
        from concurrent.futures import ThreadPoolExecutor, Future
        if executor is None:
            self._executor = ThreadPoolExecutor(8)
        elif isinstance(executor, ThreadPoolExecutor):
            self._executor = executor
        else:
            raise TypeError("executor must be a futures.concurrent.ThreadPoolExecutor "
                            "to ensure proper multitask behavior.")
        self._io_futures: list[Future] = []
        self.__update_context: list[AbstractMeasure] = []
        self._on_missing = on_missing
        self._import_all_measures: list[Path] = []
        self._subject_all_imported: list[Path] = []
        self._exported_measures: list[str] = []
        self._buffer: Dict[Path, Future[AnyBufferType]] = {}
        self._pvmode: PVMode = "vox"
        self._freesurfer_legacy: bool = legacy

        _imported_measures = list(imported_measures)
        if len(_imported_measures) != 0:
            if measurefile is None:
                raise ValueError("Measures defined to import, but no measurefile "
                                 "specified. A default must always be defined.")

            _measurefile = Path(measurefile)
            _read_measurefile = self.make_read_hook(read_measure_file)
            _read_measurefile(_measurefile, blocking=False)
            for measure_string in _imported_measures:
                self.add_imported_measure(
                    measure_string,
                    measurefile=_measurefile, read_file=_read_measurefile
                )

        _computed_measures = list(computed_measures)
        for measure_string in _computed_measures:
            self.add_computed_measure(measure_string)
        self.instantiate_measures(self.values())

    def instantiate_measures(self, measures: Iterable[AbstractMeasure]) -> None:
        """Make sure all measures that dependent on `measures` are instantiated."""
        for measure in list(measures):
            if isinstance(measure, DerivedMeasure):
                self.instantiate_measures(measure.parents)

    def add_imported_measure(self, measure_string: str, **kwargs) -> None:
        """Add an imported measure from the measure_string definition and default
        measurefile.

        Parameters
        ----------
        measure_string : str
            definition of the measure

        Other Parameters
        ----------------
        measurefile : Path
            Path to the default measurefile to import from (argument to ImportedMeasure)
        read_file : ReadFileHook[dict[str, MeasureTuple]]
            function handle to read and parse the file (argument to ImportedMeasure)

        Raises
        ------
        RuntimeError
            If trying to replace a computed Measure of the same key.

        """
        # currently also extracts args, this maybe should be removed for simpler code
        key, args = self.extract_key_args(measure_string)
        if key == "all":
            _mfile = kwargs["measurefile"] if len(args) == 0 else Path(args[0])
            self._import_all_measures.append(_mfile)
        else:
            # get default name, description and unit of key
            default = self.default(key)
            kws = () if default is None else ("name", "description", "unit")
            kwargs.update({k: getattr(default, k) for k in kws})
            if key not in self.keys() or isinstance(self[key], ImportedMeasure):
                self[key] = ImportedMeasure(key, **kwargs)
                # parse the arguments (inplace)
                self[key].parse_args(*args)
                self._exported_measures.append(key)
            else:
                raise RuntimeError(
                    "Illegal operation: Trying to replace the computed measure at "
                    f"{key} ({self[key]}) with an imported measure.")

    def add_computed_measure(self, measure_string: str) -> None:
        """Add a computed measure from the measure_string definition."""
        # currently also extracts args, this maybe should be removed for simpler code
        key, args = self.extract_key_args(measure_string)
        # also overwrite prior definition
        if key in self._exported_measures:
            self[key] = self.default(key)
        else:
            self._exported_measures.append(key)
        # load the default config of the measure and copy, overwriting other measures
        # with the same key (only keep computed versions or the last) parse the
        # arguments (inplace)
        self[key].parse_args(*args)

    def __getitem__(self, key: str) -> AbstractMeasure:
        """Get the value of the key"""
        if "(" in key:
            key, args = key.split("(", 1)
            args = args.rstrip(") ")
        else:
            args = ""
        try:
            out = super().__getitem__(key)
        except KeyError as e:
            out = self.default(key)
            if out is not None:
                self[key] = out
            else:
                raise e
        if args != "":
            out.parse_args(args)
        return out

    def start_read_subject(self, subject_dir: Path) -> None:
        """Start the threads to read the subject in subject_dir, pairs with
        `wait_read_subject()`."""
        if len(self._io_futures) != 0:
            raise RuntimeError("Did not process/wait on finishing the processing for "
                               "the previous start_read_subject run. Needs call to "
                               "`wait_read_subject`.")
        self.__update_context = []
        self._subject_all_imported = []
        read_file = self.make_read_hook(read_measure_file)
        for file in self._import_all_measures:
            path = file if file.is_absolute() else subject_dir / file
            read_file(path, blocking=False)
            self._subject_all_imported.append(path)
        self.read_subject_parents(self.values(), subject_dir, False)

    @contextmanager
    def read_subject(self, subject_dir: Path) -> None:
        """Contextmanager for the `start_read_subject()` and the `wait_read_subject()`
        pair."""
        yield self.start_read_subject(subject_dir)
        return self.wait_read_subject()

    def wait_read_subject(self) -> None:
        """Wait for all threads to finish reading the 'current' subject."""
        for f in self._io_futures:
            exception = f.exception()
            if exception is not None:
                raise exception
        self._io_futures.clear()

    def read_subject_parents(self, measures: Iterable[AbstractMeasure],
                             subject_dir: Path, blocking: bool = False) -> True:
        """
        Multi-threaded iteration through measures and application of read_subject, also
        implementation for the read_subject_on_parents function hook. Guaranteed to
        return
        independent of state and thread availability to avoid a race condition.

        Parameters
        ----------
        measures : Iterable[AbstractMeasure]
            iterable of Measures to read
        subject_dir : Path
            path to the subject directory (often subjects_dir/subject_id)
        blocking : bool, optional
            whether the

        Returns
        -------
        True
        """

        def _read(measure: AbstractMeasure) -> bool:
            return measure.read_subject(subject_dir)

        _update_context = set(
            filter(lambda m: m not in self.__update_context, measures))
        self.__update_context.extend(_update_context)
        for x in _update_context:
            if isinstance(x, DerivedMeasure):
                x.read_subject(subject_dir)
            else:
                self._io_futures.append(self._executor.submit(_read, x))
        return True

    def extract_key_args(self, measure: str) -> Tuple[str, List[str]]:
        """
        Extract the name and options from a string like '<name>' or
        '<name>(<option_list>)'.

        Parameters
        ----------
        measure : str
            The measure string

        Returns
        -------
        key : str
            the name of the measure
        args : List[str]
            a list of options

        """
        hits_no_args = self._PATTERN_NO_ARGS.match(measure)
        hits_args = self._PATTERN_ARGS.match(measure)
        if hits_no_args is not None:
            key = hits_no_args.group(1)
            args = []
        elif hits_args is not None:
            key = hits_args.group(1)
            args = self._PATTERN_DELIM.split(hits_args.group(2))
        else:
            raise ValueError(f"Invalid Format of Measure {measure}!")
        return key, args

    def make_read_hook(self, read_func: Callable[[Path], T_BufferType]) \
            -> ReadFileHook[T_BufferType]:
        """
        Wraps an io function to buffer results, multi-thread calls, etc.

        Parameters
        ----------
        read_func : Callable[[Path], T_BufferType]
            Function to read Measure entries/ images/ surfaces from a file.

        Returns
        -------
        wrapped_func : ReadFileHook[T_BufferType]
            The returned function takes a path and whether to wait for the io to finish.
            file : Path
                the path to the read from (path can be used for buffering)
            blocking : bool, optional
                do not return the data, do not wait for the io to finish, just preload
                (default: False)
            The function returns None or the output of the wrapped function.
        """

        def read_wrapper(file: Path, blocking: bool = True) -> Optional[T_BufferType]:
            if file not in self._buffer:
                self._buffer[file] = self._executor.submit(read_func, file)
            if not blocking:
                return
            else:
                return self._buffer[file].result()

        return read_wrapper

    def clear(self):
        """Clear the file buffers."""
        self._buffer = {}

    def update_measures(self) -> dict[str, float | int]:
        """Get the values to alll measures (including imported via 'all')."""
        m = {key: v[2] for key, v in self.get_imported_all_measures().items()}
        m.update({key: self[key]() for key in self._exported_measures})
        return m

    def print_measures(self, file: Optional[TextIO] = None) -> None:
        """Print the measures to stdout or file."""
        kwargs = {} if file is None else {"file": file}
        for line in self.format_measures():
            print(line, **kwargs)

    def get_imported_all_measures(self) -> dict[str, MeasureTuple]:
        """Get the measures imported through the 'all' keyword."""
        if len(self._subject_all_imported) == 0:
            return {}
        measures = {}
        read_file = self.make_read_hook(ImportedMeasure.read_file)
        for path in self._subject_all_imported:
            measures.update(read_file(path))
        return measures

    def format_measures(self) -> Iterable[str]:
        """Formats all measures as strings and returns them as an iterable of str."""
        def fmt(key: str, data: MeasureTuple) -> str:
            return f"# Measure {key}, {data[0]}, {data[1]}, {data[2]}, {data[3]}"

        measures = self.get_imported_all_measures()
        measures.update({key: self[key].as_tuple() for key in self._exported_measures})

        # order the measures, so they are in default order, appends "new" keys in the
        # order they were in exported measures
        ordered_keys = tuple(self.default_measures())
        ordered = {k: measures[k] for k in ordered_keys if k in measures}
        ordered.update(filter(lambda i: i[0] not in ordered_keys, measures.items()))
        return map(fmt, ordered.keys(), ordered.values())

    def default_measures(self) -> Iterable[str]:
        """Iterable over measures typically included stats files in correct order."""
        return ("BrainSeg", "BrainSegNotVent", "VentricleChoroidVol", "lhCortex",
                "rhCortex", "Cortex", "lhCerebralWhiteMatter", "rhCerebralWhiteMatter",
                "CerebralWhiteMatter", "SubCortGray", "TotalGray", "SupraTentorial",
                "SupraTentorialNotVent", "Mask", "BrainSegVol-to-eTIV",
                "MaskVol-to-eTIV", "lhSurfaceHoles", "rhSurfaceHoles", "SurfaceHoles",
                "EstimatedTotalIntraCranialVol")

    def default(self, key: str) -> AbstractMeasure:
        """Returns the default Measure object for the measure with key `key`."""

        read_volume = self.make_read_hook(VolumeMeasure.read_file)
        if self._freesurfer_legacy:
            def voxel_class(
                    classes: ClassesType, name: str, description: str,
                    unit: Literal["unitless", "mm^3"] = "mm^3") -> AbstractMeasure:
                return VolumeMeasure(
                    Path("mri/aseg.presurf.mgz"), classes, name, description, unit,
                    read_file=read_volume
                )

            def supratentorial_class(
                    classes: ClassesType, name: str, description: str,
                    unit: Literal["unitless", "mm^3"] = "mm^3") -> AbstractMeasure:
                return MultiVolumeMeasure(
                    Path("mri/aseg.presurf.mgz"), Path("mri/ribbon.mgz"), classes,
                    name, description, unit,
                    other_classes_or_cond=lambda x: x > 0, read_file=read_volume
                )

            def ribboncorrected_class(
                    classes: ClassesType, name: str, description: str,
                    unit: Literal["unitless", "mm^3"] = "mm^3",
                    other: ClassesOrCondType = (0, 2, 3, 41, 42)) -> AbstractMeasure:
                return MultiVolumeMeasure(
                    Path("mri/aseg.presurf.mgz"), Path("mri/ribbon.mgz"), classes,
                    name, description, unit,
                    other_classes_or_cond=other, read_file=read_volume
                )
        elif self._pvmode == "vox":

            def voxel_class(
                    classes_or_cond: ClassesOrCondType, name: str, description: str,
                    unit: Literal["mm^3"] = "mm^3") -> AbstractMeasure:
                return PVMeasure(classes_or_cond, name, description, unit)

            def supratentorial_class(
                    classes_or_cond: ClassesOrCondType, name: str, description: str,
                    unit: Literal["mm^3"] = "mm^3") -> AbstractMeasure:
                return PVMeasure(classes_or_cond, name, description, unit)

            def ribboncorrected_class(
                    classes_or_cond: ClassesOrCondType, name: str, description: str,
                    unit: Literal["mm^3"] = "mm^3",
                    other: ClassesOrCondType = (0, 2, 3, 41, 42)) -> AbstractMeasure:
                return PVMeasure(classes_or_cond, name, description, unit)
        else:
            def voxel_class(
                    classes_or_cond: ClassesOrCondType, name: str, description: str,
                    unit: Literal["unitless", "mm^3"] = "mm^3") -> AbstractMeasure:
                return VolumeMeasure(
                    Path("mri/aseg.mgz"), classes_or_cond, name, description,
                    unit, read_file=read_volume
                )

            def supratentorial_class(
                    classes_or_cond: ClassesOrCondType, name: str, description: str,
                    unit: Literal["unitless", "mm^3"] = "mm^3") -> AbstractMeasure:
                return NullMeasure(name, description, unit)

            def ribboncorrected_class(
                    classes_or_cond: ClassesOrCondType, name: str, description: str,
                    unit: Literal["unitless", "mm^3"] = "mm^3",
                    other: ClassesOrCondType = (0, 2, 3, 41, 42)) -> AbstractMeasure:
                return NullMeasure(name, description, unit)

        hemi = key[:2]
        side = "Left" if hemi != "rh" else "Right"
        cc_classes = (251, 252, 253, 253, 255)
        if key in ("lhSurfaceHoles", "rhSurfaceHoles"):
            # l/rSurfaceHoles: (1-lheno/2) -- Euler number of /surf/l/rh.orig.nofix
            return SurfaceHoles(
                Path(f"surf/{hemi}.orig.nofix"), f"{hemi}SurfaceHoles",
                f"Number of defect holes in {hemi} surfaces prior to fixing",
                "unitless"
            )
        elif key == "SurfaceHoles":
            # sum of holes in left and right surfaces
            return DerivedMeasure(
                ["rhSurfaceHoles", "lhSurfaceHoles"], "SurfaceHoles",
                "Total number of defect holes in surfaces prior to fixing",
                measure_host=self
            )
        elif key in ("lhPialTotal", "rhPialTotal"):
            return SurfaceVolume(
                Path(f"surf/{hemi}.pial"), f"{hemi}PialTotalVol",
                f"{side} hemisphere total pial volume", "mm^3"
            )
        elif key in ("lhWhiteMatterTotal", "rhWhiteMatterTotal"):
            return SurfaceVolume(
                Path(f"surf/{hemi}.white"), f"{hemi}PialVol",
                f"{side} hemisphere total white matter volume", "mm^3"
            )
        elif key in ("lhCortexRibbon", "rhCortexRibbon"):
            # l/rhCtxGMCor: 3/42 in ribbon, but not (0, 2, 3)/(0, 41, 42) in aseg
            classes = {"lh": (3,), "rh": (42,)}
            ribbon_classes = {"lh": (0, 2, 3), "rh": (0, 41, 42)}
            from functools import partial
            return ribboncorrected_class(
                classes[hemi], key,
                f"{side} hemisphere cortical gray matter volume correction", "mm^3",
                other=partial(mask_not_in_array, items=ribbon_classes[hemi])
            )
        elif key in ("lhCortex", "rhCortex"):
            # 5/6 => l/rhCtxGM: (l/rhpialvolTot - l/hwhitevolTot - l/rhCtxGMCor)
            return DerivedMeasure(
                [f"{hemi}PialTotal", (-1, f"{hemi}WhiteMatterTotal"),
                 (-1, f"{hemi}CortexRibbon")],
                f"{hemi}CortexVol", f"{side} hemisphere cortical gray matter volume",
                measure_host=self
            )
        elif key == "Cortex":
            # 7 => lhCtxGM + rhCtxGM: sum of left and right cerebral GM
            return DerivedMeasure(
                ["lhCortex", "rhCortex"],
                "CortexVol", f"Total cortical gray matter volume",
                measure_host=self
            )
        elif key == "CorpusCallosumVol":
            # CCVol:
            # CC_Posterior CC_Mid_Posterior CC_Central CC_Mid_Anterior CC_Anterior
            return voxel_class(
                cc_classes, "CorpusCallosumVol", "Volume of the Corpus Callosum", "mm^3"
            )
        elif key in ("lhWhiteMatterRibbon", "rhWhiteMatterRibbon"):
            # l/rhCtxWMCor:
            # 2/41 in ribbon, but not (2/41, 77, Corpus Callosum) in aseg
            classes = {"lh": (3,), "rh": (42,)}
            ribbon_classes = ({"lh": 2, "rh": 41}[hemi], 77) + cc_classes
            from functools import partial
            return ribboncorrected_class(
                classes[hemi], key,
                f"{side} hemisphere cortical gray matter volume correction", "mm^3",
                other=partial(mask_not_in_array, items=ribbon_classes)
            )
        elif key in ("lhCerebralWhiteMatter", "rhCerebralWhiteMatter"):
            # 9/10 => l/rCtxWM: l/rWhiteMatter
            return DerivedMeasure(
                [f"{hemi}WhiteMatterTotal", (-1, f"{hemi}WhiteMatterRibbon")],
                f"{hemi}CerebralWhiteMatterVol",
                f"{side} hemisphere cerebral white matter volume",
                measure_host=self
            )
        elif key == "CerebralWhiteMatter":
            # 11 => lhCtxWM + rhCtxWM: sum of left and right cerebral WM
            return DerivedMeasure(
                ["rhCerebralWhiteMatter", "lhCerebralWhiteMatter"],
                "CerebralWhiteMatterVol", "Total cerebral white matter volume",
                measure_host=self
            )
        elif key == "CerebellarGM":
            #
            # Left-Cerebellum-Cortex Right-Cerebellum-Cortex Cbm_Left_I_IV
            # Cbm_Right_I_IV Cbm_Left_V Cbm_Right_V Cbm_Left_VI Cbm_Vermis_VI
            # Cbm_Right_VI Cbm_Left_CrusI Cbm_Vermis_CrusI Cbm_Right_CrusI
            # Cbm_Left_CrusII Cbm_Vermis_CrusII Cbm_Right_CrusII Cbm_Left_VIIb
            # Cbm_Vermis_VIIb Cbm_Right_VIIb Cbm_Left_VIIIa Cbm_Vermis_VIIIa
            # Cbm_Right_VIIIa Cbm_Left_VIIIb Cbm_Vermis_VIIIb Cbm_Right_VIIIb
            # Cbm_Left_IX Cbm_Vermis_IX Cbm_Right_IX Cbm_Left_X Cbm_Vermis_X Cbm_Right_X
            # Cbm_Vermis_VII Cbm_Vermis_VIII Cbm_Vermis
            cerebellum_classes = (8, 47, 601, 602, 603, 604, 605, 606, 607, 608, 609,
                                  610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620,
                                  621, 622, 623, 624, 625, 626, 627, 628, 630, 631, 632)
            return voxel_class(
                cerebellum_classes, "CerebellarGMVol",
                "Cerebellar gray matter volume", "mm^3",
            )
        elif key == "SubCortGray":
            # 4 => SubCortGray
            # Left-Thalamus Right-Thalamus Left-Caudate Right-Caudate Left-Putamen
            # Right-Putamen Left-Pallidum Right-Pallidum Left-Hippocampus
            # Right-Hippocampus Left-Amygdala Right-Amygdala Left-Accumbens-area
            # Right-Accumbens-area Left-VentralDC Right-VentralDC Left-Substantia-Nigra
            # Right-Substantia-Nigra
            subcortgray_classes = (10, 11, 12, 13, 17, 18, 26, 27, 28,
                                   49, 50, 51, 52, 53, 54, 58, 59, 60)
            return voxel_class(
                subcortgray_classes, "SubCortGrayVol",
                "Subcortical gray matter volume", "mm^3"
            )
        elif key == "TotalGray":
            # 8 => TotalGMVol: sum of SubCortGray., Cortex and Cerebellar GM
            return DerivedMeasure(
                ["SubCortGray", "Cortex", "CerebellarGM"],
                "TotalGrayVol", "Total gray matter volume",
                measure_host=self
            )
        elif key == "TFFC":
            # 3rd-Ventricle 4th-Ventricle 5th-Ventricle CSF
            tffc_classes = (14, 15, 72, 24)
            return voxel_class(
                tffc_classes, "Third-Fourth-Fifth-CSF",
                "volume of 3rd, 4th, 5th ventricle and CSF", "mm^3"
            )
        elif key == "VentricleChoroidVol":
            # 15 => VentChorVol:
            # Left-Choroid-Plexus Right-Choroid-Plexus Left-Lateral-Ventricle
            # Right-Lateral-Ventricle Left-Inf-Lat-Vent Right-Inf-Lat-Vent
            ventchor_classes = (31, 63, 4, 43, 5, 44)
            return voxel_class(
                ventchor_classes, "VentricleChoroidVol",
                "Volume of ventricles and choroid plexus", "mm^3"
            )
        elif key == "BrainSeg":
            # 0 => BrainSegVol: all labels but background and brainstem
            return ribboncorrected_class(
                list(i for i in range(1, 256) if i != 16),  # not 0, not brainstem
                "BrainSegVol", "Brain Segmentation Volume", "mm^3"
            )
        elif key == "BrainSegNotVent":
            # 1 => BrainSegNotVent: BrainSegVolNotVent (BrainSegVol-VentChorVol-TFFC)
            return DerivedMeasure(
                ["BrainSeg", (-1, "VentricleChoroidVol"), (-1, "TFFC")],
                "BrainSegVolNotVent", "Brain Segmentation Volume Without Ventricles",
                measure_host=self
            )
        elif key == "SupraTentorialRibbon":
            # SupraTentCor:
            # Left-Thalamus Right-Thalamus Left-Caudate Right-Caudate Left-Putamen
            # Right-Putamen Left-Pallidum Right-Pallidum Left-Hippocampus
            # Right-Hippocampus Left-Amygdala Right-Amygdala Left-Accumbens-area
            # Right-Accumbens-area Left-VentralDC Right-VentralDC
            # Left-Lateral-Ventricle Right-Lateral-Ventricle Left-Inf-Lat-Vent
            # Right-Inf-Lat-Vent Left-choroid-plexus Right-choroid-plexus
            # WM-hypointensities Left-WM-hypointensities Right-WM-hypointensities
            # CC_Posterior CC_Mid_Posterior CC_Central CC_Mid_Anterior CC_Anterior
            supratentorial_classes = (4, 5, 10, 11, 12, 13, 17, 18, 26, 28, 31, 78,
                                      43, 44, 49, 50, 51, 52, 53, 54, 58, 60, 63, 79,
                                      77, 251, 252, 253, 254, 255)
            return supratentorial_class(
                supratentorial_classes,
                "Ribbon-Corrected-Supratentorial",
                "Volume of supratentorial operation, but not ribbon", "mm^3"
            )
        elif key == "SupraTentorial":
            # 2 => SupraTentVol: (lhpialvolTot + rhpialvolTot + SupraTentVolCor)
            return DerivedMeasure(
                ["lhPialTotal", "rhPialTotal", (-1, "SupraTentorialRibbon")],
                "SupraTentorialVol", "Supratentorial volume",
                measure_host=self
            )
        elif key == "SupraTentorialNotVent":
            # 3 => SupraTentVolNotVent: SupraTentorial w/o Ventricles & Choroid Plexus
            return DerivedMeasure(
                ["SupraTentorial", (-1, "VentricleChoroidVol")],
                "SupraTentorialVolNotVent", "Supratentorial volume",
                measure_host=self
            )
        elif key == "Mask":
            # 12 => MaskVol: Any voxel in mask > 0
            from functools import partial
            return MaskMeasure(
                Path("mri/brainmask.mgz"),
                "MaskVol", "Mask Volume", "mm^3"
            )
        elif key == "EstimatedTotalIntraCranialVol":
            # atlas_icv: eTIV from talairach transform determinate
            return ETIVMeasure(
                Path("mri/transforms/talairach.xfm"),
                "eTIV", "Estimated Total Intracranial Volume", "mm^3"
            )
        elif key == "BrainSegVol-to-eTIV":
            # 0/atlas_icv: ratio BrainSegVol to eTIV
            return DerivedMeasure(
                ["BrainSegVol", "EstimatedTotalIntraCranialVol"],
                "BrainSegVol-to-eTIV", "Ratio of BrainSegVol to eTIV",
                measure_host=self, operation="ratio"
            )
        elif key == "MaskVol-to-eTIV":
            # 12/atlas_icv: ratio Mask to eTIV
            return DerivedMeasure(
                ["Mask", "EstimatedTotalIntraCranialVol"],
                "MaskVol-to-eTIV", "Ratio of MaskVol to eTIV",
                measure_host=self, operation="ratio"
            )

    def __iter__(self) -> List[AbstractMeasure]:
        """Iterate through all measures that are exported directly or indirectly."""

        out = [self[name] for name in self._exported_measures]
        i = 0
        while i < len(out):
            this = out[i]
            if isinstance(this, DerivedMeasure):
                out.extend(filter(lambda x: x not in out, this.parents_items()))
            i += 1
        return out

    def compute_non_derived_pv(self, compute_threads: Executor) -> "list[Future]":
        """Trigger computation of all non-derived, non-pv measures that are required."""
        valid_measure_types = (DerivedMeasure, PVMeasure)
        return [compute_threads.submit(this)
                for this in self.values() if not isinstance(this, valid_measure_types)]

    def get_virtual_labels(self, label_pool: Iterable[int]) -> dict[int, List[int]]:
        """Get the virtual substitute labels that are required."""
        lbls = (this.labels() for this in self.values() if isinstance(this, PVMeasure))
        no_duplicate_dict = {self.__to_lookup(labs): labs for labs in lbls}
        return dict(zip(label_pool, no_duplicate_dict.values()))

    @staticmethod
    def __to_lookup(labels: Sequence[int]) -> str:
        return str(set(sorted(map(int, labels))))

    def update_pv_from_table(self,
                             dataframe: "pd.DataFrame",
                             merged_labels: dict[int, list[int]]) -> "pd.DataFrame":
        """Update pv measures from dataframe and remove """
        _lookup = {self.__to_lookup(ml): vl for vl, ml in merged_labels.items()}
        filtered_df = dataframe
        # go through the pv measures and find a measure that has the same list
        for this in self.values():
            if isinstance(this, PVMeasure):
                virtual_label = _lookup.get(self.__to_lookup(this.labels()), None)
                if virtual_label is None:
                    raise RuntimeError(f"Could not find the virtual label for {this}")
                row = dataframe[dataframe["SegId"] == virtual_label]
                if row.shape[0] != 1:
                    raise RuntimeError(f"The search results in the dataframe for "
                                       f"{this} failed: shape {row.shape}")
                this.update_data(row)
                filtered_df = filtered_df[filtered_df["SegId"] != virtual_label]

        return filtered_df
