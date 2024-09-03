import abc
import logging
import re
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import Executor, Future
from contextlib import contextmanager
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Generic,
    Literal,
    Protocol,
    TextIO,
    TypeVar,
    Union,
    cast,
    overload,
)

import numpy as np

if TYPE_CHECKING:
    import lapy
    import nibabel as nib
    import pandas as pd
    from numpy import typing as npt

    from CerebNet.datasets.utils import LTADict

MeasureTuple = tuple[str, str, int | float, str]
ImageTuple = tuple["nib.analyze.SpatialImage", "np.ndarray"]
UnitString = Literal["unitless", "mm^3"]
MeasureString = Union[str, "Measure"]
AnyBufferType = Union[
    dict[str, MeasureTuple],
    ImageTuple,
    "lapy.TriaMesh",
    "npt.NDArray[float]",
    "pd.DataFrame",
]
T_BufferType = TypeVar(
    "T_BufferType",
    bound=Union[
        ImageTuple,
        dict[str, MeasureTuple],
        "lapy.TriaMesh",
        "np.ndarray",
        "pd.DataFrame",
    ])
DerivedAggOperation = Literal["sum", "ratio", "by_vox_vol"]
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

    def __call__(self, file: Path, b: bool = True) -> T_BufferType | None: ...


class _DefaultFloat(float):
    pass


def read_measure_file(path: Path) -> dict[str, MeasureTuple]:
    """
    Read '# Measure <key> <name> <description> <value> <unit>'-entries from stats files.

    Parameters
    ----------
    path : Path
        The path to the file to read from.

    Returns
    -------
    A dictionary of Measure keys to tuple of descriptors like
    {'<key>': ('<name>', '<description>', <value>, '<unit>')}.
    """
    if not path.exists():
        raise OSError(f"Measures could not be imported from {path}, "
                      f"the file does not exist.")
    with open(path) as fp:
        lines = list(fp.readlines())
    vox_line = list(filter(lambda ln: ln.startswith("# VoxelVolume_mm3 "), lines))
    lines = filter(lambda ln: ln.startswith("# Measure "), lines)

    def to_measure(line: str) -> tuple[str, MeasureTuple]:
        data_tup = line.removeprefix("# Measure ").strip()
        import re
        key, name, desc, sval, unit = re.split("\\s*,\\s*", data_tup)
        value = float(sval) if "." in sval else int(sval)
        return key, (name, desc, value, unit)

    data = dict(map(to_measure, lines))
    if len(vox_line) > 0:
        vox_vol = float(vox_line[-1].split(" ")[2].strip())
        data["vox_vol"] = ("Voxel volume", "The volume of a voxel", vox_vol, "mm^3")

    return data


def read_volume_file(path: Path) -> ImageTuple:
    """
    Read a volume from disk.

    Parameters
    ----------
    path : Path
        The path to the file to read from.

    Returns
    -------
    A tuple of nibabel image object and the data.
    """
    try:
        import nibabel as nib
        img = cast(nib.analyze.SpatialImage, nib.load(path))
        if not isinstance(img, nib.analyze.SpatialImage):
            raise RuntimeError(
                f"Loading the file '{path}' for Measure was invalid, no SpatialImage."
            )
    except (OSError, FileNotFoundError) as e:
        args = e.args[0]
        raise OSError(f"Failed loading the file '{path}' with error: {args}") from e
    data = np.asarray(img.dataobj)
    return img, data


def read_mesh_file(path: Path) -> "lapy.TriaMesh":
    """
    Read a mesh from disk.

    Parameters
    ----------
    path : Path
        The path to the file.

    Returns
    -------
    lapy.TriaMesh
        The mesh object read from the file.
    """
    try:
        import lapy
        mesh = lapy.TriaMesh.read_fssurf(str(path))
    except (OSError, FileNotFoundError) as e:
        args = e.args[0]
        raise OSError(
            f"Failed loading the file '{path}' with error: {args}") from e
    return mesh


def read_lta_transform_file(path: Path) -> "npt.NDArray[float]":
    """
    Read and extract the first lta transform from an LTA file.

    Parameters
    ----------
    path : Path
        The path of the LTA file.

    Returns
    -------
    matrix : npt.NDArray[float]
        Matrix of shape (4, 4).
    """
    from CerebNet.datasets.utils import read_lta
    return read_lta(path)["lta"][0, 0]


def read_xfm_transform_file(path: Path) -> "npt.NDArray[float]":
    """
    Read XFM talairach transform.

    Parameters
    ----------
    path : str | Path
        The filename/path of the transform file.

    Returns
    -------
    tal
        The talairach transform matrix.

    Raises
    ------
    ValueError
        If the file is of an invalid format.
    """
    with open(path) as f:
        lines = f.readlines()

    try:
        transf_start = [ln.lower().startswith("linear_") for ln in lines].index(True) + 1
        tal_str = [ln.replace(";", " ") for ln in lines[transf_start:transf_start + 3]]
        tal = np.genfromtxt(tal_str)
        tal = np.vstack([tal, [0, 0, 0, 1]])

        return tal
    except Exception as e:
        err = ValueError(f"Could not find taiairach transform in {path}.")
        raise err from e


def read_transform_file(path: Path) -> "npt.NDArray[float]":
    """
    Read xfm or lta transform file.

    Parameters
    ----------
    path : Path
        The path to the file.

    Returns
    -------
    tal
        The talairach transform matrix.
    """
    if path.suffix == ".lta":
        return read_lta_transform_file(path)
    elif path.suffix == ".xfm":
        return read_xfm_transform_file(path)
    else:
        raise NotImplementedError(
            f"The extension {path.suffix} is not '.xfm' or '.lta' and not recognized.")


def mask_in_array(arr: "npt.NDArray", items: "npt.ArrayLike") -> "npt.NDArray[bool]":
    """
    Efficient function to generate a mask of elements in `arr`, which are also in items.

    Parameters
    ----------
    arr : npt.NDArray
        An array with data, most likely int.
    items : npt.ArrayLike
        Which elements of `arr` in arr should yield True.

    Returns
    -------
    mask : npt.NDArray[bool]
        A binary array, true, where elements in `arr` are in `items`.

    See Also
    --------
    mask_not_in_array
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


def mask_not_in_array(
        arr: "npt.NDArray",
        items: "npt.ArrayLike",
) -> "npt.NDArray[bool]":
    """
    Inverse of mask_in_array.

    Parameters
    ----------
    arr : npt.NDArray
        An array with data, most likely int.
    items : npt.ArrayLike
        Which elements of `arr` in arr should yield False.

    Returns
    -------
    mask : npt.NDArray[bool]
        A binary array, true, where elements in `arr` are not in `items`.

    See Also
    --------
    mask_in_array
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
    """
    The base class of all measures, which implements the name, description, and unit
    attributes as well as the methods as_tuple(), __call__(), read_subject(),
    set_args(), parse_args(), help(), and __str__().
    """

    __PATTERN = re.compile("^([^\\s=]+)\\s*=\\s*(\\S.*)$")

    def __init__(self, name: str, description: str, unit: str):
        self._name: str = name
        self._description: str = description
        self._unit: str = unit
        self._subject_dir: Path | None = None

    def as_tuple(self) -> MeasureTuple:
        return self._name, self._description, self(), self.unit

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
            Path to the directory of the subject_dir (often subject_dir/subject_id).

        Returns
        -------
        bool
            Whether there was an update.
        """
        updated = subject_dir != self.subject_dir
        if updated:
            self._subject_dir = subject_dir
        return updated

    @abc.abstractmethod
    def _parsable_args(self) -> list[str]:
        ...

    def set_args(self, **kwargs: str) -> None:
        """
        Set the arguments of the Measure.

        Raises
        ------
        ValueError
            If there are unrecognized keyword arguments.
        """
        if len(kwargs) > 0:
            raise ValueError(f"Invalid args {tuple(kwargs.keys())}")

    def parse_args(self, *args: str) -> None:
        """
        Parse additional args defining the behavior of the Measure.

        Parameters
        ----------
        *args : str
            Each args can be a string of '<value>' (arg-style) and '<keyword>=<value>'
            (keyword-arg-style), arg-style cannot follow keyword-arg-style args.

        Raises
        ------
        ValueError
            If there are more arguments than registered argument names.
        RuntimeError
            If an arg-style follows a keyword-arg-style argument, or if a keyword value
            is redefined, or a keyword is not valid.
        """

        def kwerror(i, args, msg) -> RuntimeError:
            return RuntimeError(f"Error parsing arg {i} in {args}: {msg}")

        _pargs = self._parsable_args()
        if len(args) > len(_pargs):
            raise ValueError(
                f"The measure {self.name} can have up to {len(_pargs)} arguments, but "
                f"parsing {len(args)}: {args}."
            )
        _kwargs = {}
        _kwmode = False
        for i, (arg, default_key) in enumerate(zip(args, _pargs, strict=False)):
            if (hit := self.__PATTERN.match(arg)) is None:
                # non-keyword mode
                if _kwmode:
                    raise kwerror(i, args, "non-keyword after keyword")
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
        self.set_args(**_kwargs)

    def help(self) -> str:
        """
        Compiles a help message for the measure describing the measure's settings.

        Returns
        -------
        A help string describing the Measure settings.
        """
        return f"{self.name}="

    @abc.abstractmethod
    def __str__(self) -> str:
        ...


class NullMeasure(AbstractMeasure):
    """
    A Measure that supports no operations, always returns a value of zero.
    """

    def _parsable_args(self) -> list[str]:
        return []

    def __call__(self) -> int | float:
        return 0 if self.unit == "unitless" else 0.0

    def help(self) -> str:
        return super().help() + "NULL"

    def __str__(self) -> str:
        return "NullMeasure()"


class Measure(AbstractMeasure, Generic[T_BufferType], metaclass=abc.ABCMeta):
    """
    Class to buffer computed values, buffers computed values. Implements a value
    buffering interface for computed measure values and implement the read_subject
    pattern.
    """

    __buffer: float | int | None
    __token: str = ""
    __PATTERN = re.compile("^([^\\s=]*file)\\s*=\\s*(\\S.*)$")

    def __call__(self) -> int | float:
        token = str(self._subject_dir)
        if self.__buffer is None or self.__token != token:
            self.__token = token
            self.__buffer = self._compute()
        return self.__buffer

    @abc.abstractmethod
    def _compute(self) -> int | float:
        ...

    def __init__(
            self,
            file: Path,
            name: str,
            description: str,
            unit: str,
            read_hook: ReadFileHook[T_BufferType],
    ):
        self._file = file
        self._callback = read_hook
        self._data: T_BufferType | None = None
        self.__buffer = None
        super().__init__(name, description, unit)

    def _load_error(self, name: str = "data") -> RuntimeError:
        return RuntimeError(
            f"The '{name}' is not available for {self.name} ({type(self).__name__}), "
            f"maybe the subject has not been loaded or the cache been invalidated."
        )

    def _filename(self) -> Path:
        return self._subject_dir / self._file

    def read_subject(self, subject_dir: Path) -> bool:
        """
        Perform IO required to compute/fill the Measure. Delegates file reading to
        read_hook (set in __init__).

        Parameters
        ----------
        subject_dir : Path
            Path to the directory of the subject_dir (often subject_dir/subject_id).

        Returns
        -------
        bool
            Whether there was an update to the data.
        """
        if super().read_subject(subject_dir):
            try:
                self._data = self._callback(self._filename())
            except Exception as e:
                e.args = f"{e.args[0]} ... during reading for measure {self}.",
                raise e
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
    """
    A Measure that implements reading measure values from a statsfile.
    """

    PREFIX = "__IMPORTEDMEASURE-prefix__"
    read_file = staticmethod(read_measure_file)

    def __init__(
            self,
            key: str,
            measurefile: Path,
            name: str = "N/A",
            description: str = "N/A",
            unit: UnitString = "unitless",
            read_file: ReadFileHook[dict[str, MeasureTuple]] | None = None,
            vox_vol: float | None = None,
    ):
        self._key: str = key
        super().__init__(
            measurefile,
            name,
            description,
            unit,
            self.read_file if read_file is None else read_file,
        )
        self._vox_vol: float | None = vox_vol

    def _compute(self) -> int | float:
        """
        Will also update the name, description and unit from the strings in the file.

        Returns
        -------
        value : int | float
            value of the measure (as read from the file)
        """
        try:
            self._name, self._description, out, self._unit = self._data[self._key]
        except KeyError as e:
            raise KeyError(f"Could not find {self._key} in {self._file}.") from e
        return out

    def _parsable_args(self) -> list[str]:
        return ["key", "measurefile"]

    def set_args(
            self,
            key: str | None = None,
            measurefile: str | None = None,
            **kwargs: str,
    ) -> None:
        if measurefile is not None:
            kwargs["file"] = measurefile
        if key is not None:
            self._key = key
        return super().set_args(**kwargs)

    def help(self) -> str:
        return super().help() + f" imported from {self._file}"

    def __str__(self) -> str:
        return f"ImportedMeasure(key={self._key}, measurefile={self._file})"

    def assert_measurefile_absolute(self):
        """
        Assert that the Measure can be imported without a subject and subject_dir.

        Raises
        ------
        AssertionError
        """
        if not self._file.is_absolute() or not self._file.exists():
            raise AssertionError(
                f"The ImportedMeasures {self.name} is defined for import, but the "
                f"associated measure file {self._file} is not an absolute path or "
                f"does not exist and no subjects dir or subject id are defined."
            )

    def get_vox_vol(self) -> float:
        """
        Returns the voxel volume.

        Returns
        -------
        float
            The voxel volume associated with the imported measure.

        Raises
        ------
        RuntimeError
            If the voxel volume was not defined.
        """
        if self._vox_vol is None:
            raise RuntimeError(f"The voxel volume of {self} has never been specified.")
        return self._vox_vol

    def set_vox_vol(self, value: float):
        self._vox_vol = value

    def read_subject(self, subject_dir: Path) -> bool:
        if super().read_subject(subject_dir):
            vox_vol_tup = self._data.get("vox_vol", None)
            if isinstance(vox_vol_tup, tuple) and len(vox_vol_tup) > 2:
                self._vox_vol = vox_vol_tup[2]
            return True
        return False


class SurfaceMeasure(Measure["lapy.TriaMesh"], metaclass=abc.ABCMeta):
    """
    Class to implement default Surface io.
    """

    read_file = staticmethod(read_mesh_file)

    def __init__(
            self,
            surface_file: Path,
            name: str,
            description: str,
            unit: UnitString,
            read_mesh: ReadFileHook["lapy.TriaMesh"] | None = None,
    ):
        super().__init__(
            surface_file,
            name,
            description,
            unit,
            self.read_file if read_mesh is None else read_mesh,
        )

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

    def __init__(
            self,
            classes: ClassesType,
            name: str,
            description: str,
            unit: Literal["mm^3"] = "mm^3",
    ):
        if unit != "mm^3":
            raise ValueError("unit must be mm^3 for PVMeasure!")
        self._classes = classes
        super().__init__(name, description, unit)
        self._pv_value = None

    @property
    def vox_vol(self) -> float:
        return self._vox_vol

    @vox_vol.setter
    def vox_vol(self, v: float):
        self._vox_vol = v

    def labels(self) -> list[int]:
        return list(self._classes)

    def update_data(self, value: "pd.Series"):
        self._pv_value = value

    def __call__(self) -> float:
        if self._pv_value is None:
            raise RuntimeError(
                f"The partial volume of {self._name} has not been updated in the "
                f"PVMeasure object yet!"
            )
        col = "NVoxels" if self.unit == "unitless" else "Volume_mm3"
        return self._pv_value[col].item()

    def _parsable_args(self) -> list[str]:
        return ["classes"]

    def set_args(self, classes: str | None = None, **kwargs: str) -> None:
        if classes is not None:
            self._classes = classes
        return super().set_args(**kwargs)

    def __str__(self) -> str:
        return f"PVMeasure(classes={list(self._classes)})"

    def help(self) -> str:
        help_str = f"partial volume of {format_classes(self._classes)} in seg file"
        return super().help() + help_str


def format_classes(_classes: Iterable[int]) -> str:
    """
    Formats an iterable of classes. This compresses consecutive integers into ranges.
    >>> format_classes([1, 2, 3, 6])  # '1-3,6'

    Parameters
    ----------
    _classes : Iterable[int]
        An iterable of integers.

    Returns
    -------
    A string of sorted integers and integer ranges, '()' if iterable is empty, or just
    the string conversion of _classes, if _classes is not an iterable.

    Notes
    -----
    This function will likely be moved to a different file.
    """
    # TODO move this function to a more appropriate module
    if not isinstance(_classes, Iterable):
        return str(_classes)
    from itertools import pairwise
    sorted_list = list(sorted(_classes))
    if len(sorted_list) == 0:
        return "()"
    prev = ""
    out = str(sorted_list[0])

    for a, b in pairwise(sorted_list):
        if a != b - 1:
            out += f"{prev},{b}"
            prev = ""
        else:
            prev = f"-{b}"
    return out + prev


class VolumeMeasure(Measure[ImageTuple]):
    """
    Counts Voxels belonging to a class or condition.
    """

    read_file = staticmethod(read_volume_file)

    def __init__(
            self,
            segfile: Path,
            classes_or_cond: ClassesOrCondType,
            name: str,
            description: str,
            unit: UnitString = "unitless",
            read_file: ReadFileHook[ImageTuple] | None = None,
    ):
        if callable(classes_or_cond):
            self._classes: ClassesType | None = None
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

    def get_vox_vol(self) -> float:
        return np.prod(self._data[0].header.get_zooms()).item()

    def _compute(self) -> int | float:
        if not isinstance(self._data, tuple) or len(self._data) != 2:
            raise self._load_error("data")
        vox_vol = 1 if self._unit == "unitless" else self.get_vox_vol()
        return np.sum(self._cond(self._data[1]), dtype=int).item() * vox_vol

    def _parsable_args(self) -> list[str]:
        return ["segfile", "classes"]

    def _set_classes(self, classes: str | None, attr_name: str, cond_name: str) -> None:
        """Helper method for set_args."""
        if classes is not None:
            from functools import partial
            _classes = re.split("\\s+", classes.lstrip("[ ").rstrip("] "))
            items = list(map(int, _classes))
            setattr(self, attr_name, items)
            setattr(self, cond_name, partial(mask_in_array, items=items))

    def set_args(
            self,
            segfile: str | None = None,
            classes: str | None = None,
            **kwargs: str,
    ) -> None:
        if segfile is not None:
            kwargs["file"] = segfile
        self._set_classes(classes, "_classes", "_cond")
        return super().set_args(**kwargs)

    def __str__(self) -> str:
        return f"{type(self).__name__}(segfile={self._file}, {self._param_string()})"

    def help(self) -> str:
        return f"{self._name}={self._param_help()} in {self._file}"

    def _param_help(self, prefix: str = ""):
        """Helper method for format classes and cond."""
        cond = getattr(self, prefix + "_cond")
        classes = getattr(self, prefix + "_classes")
        return prefix + (f"cond={cond}" if classes is None else format_classes(classes))

    def _param_string(self, prefix: str = ""):
        """Helper method to convert classes and cond to string."""
        cond = getattr(self, prefix + "_cond")
        classes = getattr(self, prefix + "_classes")
        return prefix + (f"cond={cond}" if classes is None else f"classes={classes}")


class MaskMeasure(VolumeMeasure):

    def __init__(
            self,
            maskfile: Path,
            name: str,
            description: str,
            unit: UnitString = "unitless",
            threshold: float = 0.5,
            # sign: MaskSign = "abs", frame: int = 0,
            # erode: int = 0, invert: bool = False,
            read_file: ReadFileHook[ImageTuple] | None = None,
    ):
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

    def set_args(
            self,
            maskfile: Path | None = None,
            threshold: float | None = None,
            **kwargs: str,
    ) -> None:
        if threshold is not None:
            self._threshold = float(threshold)
        if maskfile is not None:
            kwargs["file"] = maskfile
        return super().set_args(**kwargs)

    def _parsable_args(self) -> list[str]:
        return ["maskfile", "threshold"]

    def __str__(self) -> str:
        return (
            f"{type(self).__name__}(maskfile={self._file}, threshold={self._threshold})"
        )

    def _param_help(self, prefix: str = ""):
        return f"voxel > {self._threshold}"


AnyParentsTuple = tuple[float, AnyMeasure]
ParentsTuple = tuple[float, AnyMeasure]


class TransformMeasure(Measure, metaclass=abc.ABCMeta):
    read_file = staticmethod(read_transform_file)

    def __init__(
            self,
            lta_file: Path,
            name: str,
            description: str,
            unit: str,
            read_lta: ReadFileHook["npt.NDArray[float]"] | None = None,
    ):
        super().__init__(
            lta_file,
            name,
            description,
            unit,
            self.read_file if read_lta is None else read_lta,
        )

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

    def __init__(
            self,
            lta_file: Path,
            name: str,
            description: str,
            unit: str,
            read_lta: ReadFileHook["LTADict"] | None = None,
            etiv_scale_factor: float | None = None,
    ):
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
        return f"{super().__str__()[:-1]}, etiv_scale_factor={self._etiv_scale_factor})"


class DerivedMeasure(AbstractMeasure):

    def __init__(
            self,
            parents: Iterable[tuple[float, AnyMeasure] | AnyMeasure],
            name: str,
            description: str,
            unit: str = "from parents",
            operation: DerivedAggOperation = "sum",
            measure_host: dict[str, AbstractMeasure] | None = None,
    ):
        """
        Create the Measure, which depends on other measures, called parent measures.

        Parameters
        ----------
        parents : Iterable[tuple[float, AbstractMeasure] | AbstractMeasure]
            Iterable of either the measures (or a tuple of a float and a measure), the
            float is the factor by which the value of the respective measure gets
            weighted and defaults to 1.
        name : str
            Name of the Measure.
        description : str
            Description text of the measure
        unit : str, optional
            Unit of the measure, typically 'mm^3' or 'unitless', autogenerated from
            parents' unit.
        operation : "sum", "ratio", "by_vox_vol", optional
            How to aggregate multiple `parents`, default = 'sum'
            'ratio' only supports exactly 2 parents.
            'by_vox_vol' only supports exactly one parent.
        measure_host : dict[str, AbstractMeasure], optional
            A dict-like to provide AbstractMeasure objects for strings.
        """

        def to_tuple(
                value: tuple[float, AnyMeasure] | AnyMeasure,
        ) -> tuple[float, AnyMeasure]:
            if isinstance(value, Sequence) and not isinstance(value, str):
                if len(value) != 2:
                    raise ValueError("A tuple was not length 2.")
                factor, measure = value
            else:
                factor, measure = 1., value

            if not isinstance(measure, str | AbstractMeasure):
                raise ValueError(f"Expected a str or AbstractMeasure, not "
                                 f"{type(measure).__name__}!")
            if not isinstance(factor, float):
                factor = float(factor)
            return factor, measure

        self._parents: list[AnyParentsTuple] = [to_tuple(p) for p in parents]
        if len(self._parents) == 0:
            raise ValueError("No parents defined in DerivedMeasure.")
        self._measure_host = measure_host
        if operation in ("sum", "ratio", "by_vox_vol"):
            self._operation: DerivedAggOperation = operation
        else:
            raise ValueError("operation must be 'sum', 'ratio' or 'by_vox_vol'.")
        super().__init__(name, description, unit)

    @property
    def unit(self) -> str:
        """
        Property to access the unit attribute, also implements auto-generation of unit,
        if the stored unit is 'from parents'.

        Returns
        -------
        str
            A string that identifies the unit of the Measure.

        Raises
        ------
        RuntimeError
            If unit is 'from parents' and some parent measures are inconsistent with
            each other.
        """
        if self._unit == "from parents":
            units = list(map(lambda x: x.unit, self.parents))
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
            elif self._operation == "by_vox_vol":
                if len(units) != 1:
                    raise self.invalid_len_vox_vol()
                elif units[0] == "mm^3":
                    return "unitless"
                else:
                    raise RuntimeError("Invalid value of parent, must be mm^3, but "
                                       f"was {units[0]}.")
            raise RuntimeError(
                f"unit is set to auto-generate from parents, but the parents' units "
                f"are not consistent: {units}!"
            )
        else:
            return super().unit

    def invalid_len_ratio(self) -> RuntimeError:
        return RuntimeError(f"Invalid number of parents ({len(self._parents)}) for "
                            f"operation 'ratio'.")

    def invalid_len_vox_vol(self) -> RuntimeError:
        return RuntimeError(f"Invalid number of parents ({len(self._parents)}) for "
                            f"operation 'by_vox_vol'.")

    @property
    def parents(self) -> Iterable[AbstractMeasure]:
        """Iterable of the measures this measure depends on."""
        return (p for _, p in self.parents_items())

    def parents_items(self) -> Iterable[tuple[float, AbstractMeasure]]:
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
            Path to the directory of the subject_dir (often subject_dir/subject_id).

        Returns
        -------
        bool
            Whether there was an update.

        Notes
        -----
        Might trigger a race condition if the function hook `read_subject_on_parents`
        depends on this method finishing first, e.g. because of thread availability.
        """
        if super().read_subject(subject_dir):
            return self.read_subject_on_parents(self._subject_dir)
        return False

    def __call__(self) -> int | float:
        """
        Compute dependent measures and accumulate them according to the operation.
        """
        factor_value = [(s, m()) for s, m in self.parents_items()]
        isint = all(isinstance(v, int) for _, v in factor_value)
        isint &= all(np.isclose(s, np.round(s)) for s, _ in factor_value)
        values = [s * v for s, v in factor_value]
        if self._operation == "sum":
            # sum should be an int, if all contributors are int
            # and all factors are integers (but not necessarily int)
            out = np.sum(values)
            target_type = int if isint else float
            return target_type(out)
        elif self._operation == "by_vox_vol":
            if len(self._parents) != 1:
                raise self.invalid_len_vox_vol()
            vox_vol = self.get_vox_vol()
            if isinstance(vox_vol, _DefaultFloat):
                logging.getLogger(__name__).warning(
                    f"The vox_vol in {self} was unexpectedly not initialized; using "
                    f"{vox_vol}!"
                )
            # ratio should always be float / could be partial voxels
            return float(values[0]) / vox_vol
        else:  # operation == "ratio"
            if len(self._parents) != 2:
                raise self.invalid_len_ratio()
            # ratio should always be float
            return float(values[0]) / float(values[1])

    def get_vox_vol(self) -> float | None:
        """
        Return the voxel volume of the first parent measure.

        Returns
        -------
        float, None
            voxel volume of the first parent
        """
        _types = (VolumeMeasure, DerivedMeasure)
        _type = ImportedMeasure
        fallback = None
        for p in self.parents:
            if isinstance(p, _types) and (_vvol := p.get_vox_vol()) is not None:
                return _vvol
            if isinstance(p, _type) and (_vvol := p.get_vox_vol()) is not None:
                if isinstance(_vvol, _DefaultFloat):
                    fallback = _vvol
                else:
                    return _vvol
        return fallback

    def _parsable_args(self) -> list[str]:
        return ["parents", "operation"]

    def set_args(
            self,
            parents: str | None = None,
            operation: str | None = None,
            **kwargs: str,
    ) -> None:
        if parents is not None:
            pat = re.compile("^(\\d+\\.?\\d*\\s+)?(\\s.*)")
            stripped = parents.lstrip("[ ").rstrip("] ")

            def parse(p: str) -> tuple[float, str]:
                hit = pat.match(p)
                if hit is None:
                    return 1., p
                return 1. if hit.group(1).strip() else float(hit.group(1)), hit.group(2)

            self._parents = list(map(parse, re.split("\\s+", stripped)))
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
        elif self._operation == "by_vox_vol":
            f, measure = self._parents[0]
            return f"{sign[f >= 0]} {format_factor(f)} [{format_parent(measure)}]"
        elif self._operation == "ratio":
            f = self._parents[0][0] / self._parents[1][0]
            return (f" {sign[f >= 0]} {format_factor(f)} (" +
                    ") / (".join(format_parent(p[1]) for p in self._parents) + ")")
        else:
            return f"invalid operation {self._operation}"


class VoxelClassGenerator(Protocol):
    """
    Generator for voxel-based metric Measures.
    """

    def __call__(
            self,
            classes: Sequence[int],
            name: str,
            description: str,
            unit: str,
    ) -> PVMeasure | VolumeMeasure:
        ...


def format_measure(key: str, data: MeasureTuple) -> str:
    value = data[2] if isinstance(data[2], int) else f"{data[2]:.6f}"
    return f"# Measure {key}, {data[0]}, {data[1]}, {value}, {data[3]}"


class Manager(dict[str, AbstractMeasure]):
    _PATTERN_NO_ARGS = re.compile("^\\s*([^(]+?)\\s*$")
    _PATTERN_ARGS = re.compile("^\\s*([^(]+)\\(\\s*([^)]*)\\s*\\)\\s*$")
    _PATTERN_DELIM = re.compile("\\s*,\\s*")

    _compute_futures: list[Future]
    __DEFAULT_MEASURES = (
        "BrainSeg",
        "BrainSegNotVent",
        "VentricleChoroidVol",
        "lhCortex",
        "rhCortex",
        "Cortex",
        "lhCerebralWhiteMatter",
        "rhCerebralWhiteMatter",
        "CerebralWhiteMatter",
        "SubCortGray",
        "TotalGray",
        "SupraTentorial",
        "SupraTentorialNotVent",
        "Mask",
        "BrainSegVol-to-eTIV",
        "MaskVol-to-eTIV",
        "lhSurfaceHoles",
        "rhSurfaceHoles",
        "SurfaceHoles",
        "EstimatedTotalIntraCranialVol",
    )

    def __init__(
            self,
            measures: Sequence[tuple[bool, str]],
            measurefile: Path | None = None,
            segfile: Path | None = None,
            on_missing: Literal["fail", "skip", "fill"] = "fail",
            executor: Executor | None = None,
            legacy_freesurfer: bool = False,
            aseg_replace: Path | None = None,
    ):
        """

        Parameters
        ----------
        measures : Sequence[tuple[bool, str]]
            The measures to be included as whether it is computed and name/measure str.
        measurefile : Path, optional
            The path to the file to import measures from (other stats file, absolute or
            relative to subject_dir).
        segfile : Path, optional
            The path to the file to use for segmentation (other stats file, absolute or
            relative to subject_dir).
        on_missing : Literal["fail", "skip", "fill"], optional
            behavior to follow if a requested measure does not exist in path.
        executor : concurrent.futures.Executor, optional
            thread pool to parallelize io
        legacy_freesurfer : bool, default=False
            FreeSurfer compatibility mode.
        """
        from concurrent.futures import Future, ThreadPoolExecutor
        from copy import deepcopy

        def _check_measures(x):
            return not (isinstance(x, tuple) and len(x) == 2 or
                        isinstance(x[0], bool) or isinstance(x[1], str))
        super().__init__()
        self._default_measures = deepcopy(self.__DEFAULT_MEASURES)
        if not isinstance(measures, Sequence) or any(map(_check_measures, measures)):
            raise ValueError("measures must be sequences of str.")
        if executor is None:
            self._executor = ThreadPoolExecutor(8)
        elif isinstance(executor, ThreadPoolExecutor):
            self._executor = executor
        else:
            raise TypeError(
                "executor must be a futures.concurrent.ThreadPoolExecutor to ensure "
                "proper multitask behavior."
            )
        self._io_futures: list[Future] = []
        self.__update_context: list[AbstractMeasure] = []
        self._on_missing = on_missing
        self._import_all_measures: list[Path] = []
        self._subject_all_imported: list[Path] = []
        self._exported_measures: list[str] = []
        self._cache: dict[Path, Future[AnyBufferType] | AnyBufferType] = {}
        # self._lut: Optional[pd.DataFrame] = None
        self._fs_compat: bool = legacy_freesurfer
        self._seg_from_file = Path("mri/aseg.mgz")
        if aseg_replace:
            # explicitly defined a file to reduce the aseg for segmentation mask with
            logging.getLogger(__name__).info(
                f"Replacing segmentation volume to compute volume measures from with "
                f"the explicitly defined {aseg_replace}."
            )
            self._seg_from_file = Path(aseg_replace)
        elif not self._fs_compat and segfile and Path(segfile) != self._seg_from_file:
            # not in freesurfer compatibility mode, so implicitly use segfile
            logging.getLogger(__name__).info(
                f"Replacing segmentation volume to compute volume measures from with "
                f"the segmentation file {segfile}."
            )
            self._seg_from_file = Path(segfile)

        import_kwargs = {"vox_vol": _DefaultFloat(1.0)}
        if any(filter(lambda x: x[0], measures)):
            if measurefile is None:
                raise ValueError(
                    "Measures defined to import, but no measurefile specified. "
                    "A default must always be defined."
                )
            import_kwargs["measurefile"] = Path(measurefile)
            import_kwargs["read_file"] = self.make_read_hook(read_measure_file)
            import_kwargs["read_file"](Path(measurefile), blocking=False)
        for is_imported, measure_string in measures:
            if is_imported:
                self.add_imported_measure(measure_string, **import_kwargs)
            else:
                self.add_computed_measure(measure_string)
        self.instantiate_measures(self.values())

    @property
    def executor(self) -> Executor:
        return self._executor

    # @property
    # def lut(self) -> Optional["pd.DataFrame"]:
    #     return self._lut
    #
    # @lut.setter
    # def lut(self, lut: Optional["pd.DataFrame"]):
    #     self._lut = lut

    def assert_measure_need_subject(self) -> None:
        """
        Assert whether the measure expects a definition of the subject_dir.

        Raises
        ------
        AssertionError
        """
        any_computed = False
        for _key, measure in self.items():
            if isinstance(measure, DerivedMeasure):
                pass
            elif isinstance(measure, ImportedMeasure):
                measure.assert_measurefile_absolute()
            else:
                any_computed = True
        if any_computed:
            raise AssertionError(
                "Computed measures are defined, but no subjects dir or subject id."
            )

    def instantiate_measures(self, measures: Iterable[AbstractMeasure]) -> None:
        """
        Make sure all measures that dependent on `measures` are instantiated.
        """
        for measure in list(measures):
            if isinstance(measure, DerivedMeasure):
                self.instantiate_measures(measure.parents)

    def add_imported_measure(self, measure_string: str, **kwargs) -> None:
        """
        Add an imported measure from the measure_string definition and default
        measurefile.

        Parameters
        ----------
        measure_string : str
            Definition of the measure.

        Other Parameters
        ----------------
        measurefile : Path
            Path to the default measurefile to import from (ImportedMeasure argument).
        read_file : ReadFileHook[dict[str, MeasureTuple]]
            Function handle to read and parse the file (argument to ImportedMeasure).
        vox_vol: float, optional
            The voxel volume to associate the measure with.

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
        elif key not in self.keys() or isinstance(self[key], ImportedMeasure):
            # note: name, description and unit are always updated from the input file
            self[key] = ImportedMeasure(key, **kwargs)
            # parse the arguments (inplace)
            self[key].parse_args(*args)
            self._exported_measures.append(key)
        else:
            raise RuntimeError(
                "Illegal operation: Trying to replace the computed measure at "
                f"{key} ({self[key]}) with an imported measure."
            )

    def add_computed_measure(
            self,
            measure_string: str,
    ) -> None:
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
        """
        Get the value of the key.

        Parameters
        ----------
        key : str
            A string naming the Measure, may also include extra parameters as format
            '<name>(<parameter list>)', e.g. 'Mask(maskfile=/path/to/mask.mgz)'.

        Returns
        -------
        AbstractMeasure
            The measure associated with the '<name>'
        """
        if "(" in key:
            key, args = key.split("(", 1)
            args = list(map(str.strip, args.rstrip(") ").split(",")))
        else:
            args = []
        try:
            out = super().__getitem__(key)
        except KeyError:
            out = self.default(key)
            if out is not None:
                self[key] = out
            else:
                raise
        if len(args) > 0:
            out.parse_args(*args)
        return out

    def start_read_subject(self, subject_dir: Path) -> None:
        """
        Start the threads to read the subject in subject_dir, pairs with
        `wait_read_subject`.

        Parameters
        ----------
        subject_dir : Path
            The path to the directory of the subject (with folders 'mri', 'stats', ...).
        """
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
    def with_subject(self, subjects_dir: Path | None, subject_id: str | None) -> None:
        """
        Contextmanager for the `start_read_subject` and the `wait_read_subject` pair.

        If one value is None, it is assumed the subject_dir and subject_id are not
        needed, for example because all file names are given by absolute paths.

        Parameters
        ----------
        subjects_dir : Path, None
            The path to the directory of the subject (with folders 'mri', 'stats', ...).
        subject_id : str, None
            The subject_id identifying folder of the subjects_dir.

        Raises
        ------
        AssertionError
            If subjects_dir and or subject_id are needed.
        """
        if subjects_dir is None or subject_id is None:
            yield self.assert_measure_need_subject()
            # no reading the subject required, we have no measures to include
            return
        else:
            # the subject is defined, we read it.
            yield self.start_read_subject(subjects_dir / subject_id)
            return self.wait_read_subject()

    def wait_read_subject(self) -> None:
        """
        Wait for all threads to finish reading the 'current' subject.

        Raises
        ------
        Exception
            The first exception encountered during the read operation.
        """
        for f in self._io_futures:
            exception = f.exception()
            if exception is not None:
                raise exception
        self._io_futures.clear()
        vox_vol = None

        def check_needs_init(m: AbstractMeasure) -> bool:
            return isinstance(m, ImportedMeasure) and isinstance(m.get_vox_vol(),
                                                                 _DefaultFloat)

        # and an ImportedMeasure is present, but not initialized
        for m in filter(check_needs_init, self.values()):
            # lazily load a value for vox_vol
            if vox_vol is None:
                # if the _seg_from_file file is loaded into the cache (should be)
                if self._seg_from_file in self._cache:
                    read_func = self.make_read_hook(read_volume_file)
                    img, _ = read_func(self._seg_from_file, blocking=True)
                    vox_vol = np.prod(img.header.get_zooms())
            if vox_vol is not None:
                m.set_vox_vol(vox_vol)

    def read_subject_parents(
            self,
            measures: Iterable[AbstractMeasure],
            subject_dir: Path,
            blocking: bool = False,
    ) -> True:
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
            Path to the subject directory (often subjects_dir/subject_id).
        blocking : bool, optional
            whether the execution should be parallel or not (default: False/parallel).

        Returns
        -------
        True
        """

        def _read(measure: AbstractMeasure) -> bool:
            """Callback so files for measures are loaded in other threads."""
            return measure.read_subject(subject_dir)

        _update_context = set(
            filter(lambda m: m not in self.__update_context, measures)
        )
        # __update_context is the structure that holds measures that have read_subject
        # already called / submitted to the executor
        self.__update_context.extend(_update_context)
        for x in _update_context:
            # DerivedMeasure.read_subject calls Manager.read_subject_parents (this
            # method) to read the data from dependent measures (through the callback
            # DerivedMeasure.read_subject_on_parents, and DerivedMeasure.measure_host).
            if blocking or isinstance(x, DerivedMeasure):
                x.read_subject(subject_dir)
            else:
                # calls read_subject on all measures, redundant io operations are
                # handled/skipped through Manager.make_read_hook and the internal
                # caching of files within the _cache attribute of Manager.
                self._io_futures.append(self._executor.submit(_read, x))
        return True

    def extract_key_args(self, measure: str) -> tuple[str, list[str]]:
        """
        Extract the name and options from a string like '<name>(<options_list>)'.

        The '<option_list>' is optional and is similar to python parameters. It starts
        with numbered parameters, followed by key-value pairs.
        Examples are:
        - 'Mask(mri/aseg.mgz)'
          returns: ('BrainSeg', ['mri/aseg.mgz', 'classes=[2, 4]'])
        - 'TotalGray(mri/aseg.mgz, classes=[2, 4])'
          returns: ('BrainSeg', ['mri/aseg.mgz', 'classes=[2, 4]'])
        - 'BrainSeg(segfile=mri/aseg.mgz, classes=[2, 4])'
          returns: ('BrainSeg', ['segfile=mri/aseg.mgz', 'classes=[2, 4]'])

        Parameters
        ----------
        measure : str
            The measure string of the format '<name>' or '<name>(<list of parameters>)'.

        Returns
        -------
        key : str
            the name of the measure
        args : list[str]
            a list of options

        Raises
        ------
        ValueError
            If the string `measure` does not conform to the format requirements.
        """
        hits_no_args = self._PATTERN_NO_ARGS.match(measure)
        if hits_no_args is not None:
            key = hits_no_args.group(1)
            args = []
        elif (hits_args := self._PATTERN_ARGS.match(measure)) is not None:
            key = hits_args.group(1)
            args = self._PATTERN_DELIM.split(hits_args.group(2))
        else:
            extra = ""
            if any(q in measure for q in "\"'"):
                extra = ", watch out for quotes"
            raise ValueError(f"Invalid Format of Measure \"{measure}\"{extra}!")
        return key, args

    def make_read_hook(
            self,
            read_func: Callable[[Path], T_BufferType],
    ) -> ReadFileHook[T_BufferType]:
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

        def read_wrapper(file: Path, blocking: bool = True) -> T_BufferType | None:
            out = self._cache.get(file, None)
            if out is None:
                # not already in cache
                if blocking:
                    out = read_func(file)
                else:
                    out = self._executor.submit(read_func, file)
                self._cache[file] = out
            if not blocking:
                return
            elif isinstance(out, Future):
                self._cache[file] = out = out.result()
            return out

        return read_wrapper

    def clear(self):
        """
        Clear the file buffers.
        """
        self._cache = {}

    def update_measures(self) -> dict[str, float | int]:
        """
        Get the values to all measures (including imported via 'all').

        Returns
        -------
        dict[str, Union[float, int]]
            A dictionary of '<key>' (the Measure key) and the associated value.
        """
        m = {key: v[2] for key, v in self.get_imported_all_measures().items()}
        m.update({key: self[key]() for key in self._exported_measures})
        return m

    def print_measures(self, file: TextIO | None = None) -> None:
        """
        Print the measures to stdout or file.

        Parameters
        ----------
        file: TextIO, optional
            The file object to write to. If None, writes to stdout.
        """
        kwargs = {} if file is None else {"file": file}
        for line in self.format_measures():
            print(line, **kwargs)

    def get_imported_all_measures(self) -> dict[str, MeasureTuple]:
        """
        Get the measures imported through the 'all' keyword.

        Returns
        -------
        dict[str, MeasureTuple]
            A dictionary of Measure keys and tuples of name, description, value, unit.
        """
        if len(self._subject_all_imported) == 0:
            return {}
        measures = {}
        read_file = self.make_read_hook(ImportedMeasure.read_file)
        for path in self._subject_all_imported:
            measures.update(read_file(path))
        return measures

    def format_measures(
            self, /,
            fmt_func: Callable[[str, MeasureTuple], str] = format_measure,
    ) -> Iterable[str]:
        """
        Formats all measures as strings and returns them as an iterable of str.

        In the output, measures are ordered in the order they are added to the Manager
        object. Finally, the "all"-imported Measures are appended.

        Parameters
        ----------
        fmt_func: callable, default=fmt_measure
            Function to format the key and a MeasureTuple object into a string.

        Returns
        -------
        Iterable[str]
            An iterable of the measure strings.
        """
        measures = {key: self[key].as_tuple() for key in self._exported_measures}
        for k, v in self.get_imported_all_measures().items():
            measures.setdefault(k, v)

        return map(lambda x: fmt_func(*x), measures.items())

    @property
    def default_measures(self) -> Iterable[str]:
        """
        Iterable over measures typically included stats files in correct order.

        Returns
        -------
        Iterable[str]
            An ordered iterable of the default Measure keys.
        """
        return self._default_measures

    @default_measures.setter
    def default_measures(self, values: Iterable[str]):
        """
        Sets the iterable over measure keys in correct order.

        Parameters
        ----------
        values : Iterable[str]
            An ordered iterable of the default Measure keys.
        """
        self._default_measures = values

    @property
    def voxel_class(self) -> VoxelClassGenerator:
        """
        A callable initializing a Volume-based Measure object with the legacy mode.

        Returns
        -------
        type[AbstractMeasure]
            A callable to create an object to perform a Volume-based Measure.
        """
        from functools import partial
        if self._fs_compat:
            return partial(
                VolumeMeasure,
                self._seg_from_file,
                read_file=self.make_read_hook(VolumeMeasure.read_file),
            )
        else:  # FastSurfer compat == None
            return partial(PVMeasure)

    def default(self, key: str) -> AbstractMeasure:
        """
        Returns the default Measure object for the measure with key `key`.

        Parameters
        ----------
        key : str
            The key name of the Measure.

        Returns
        -------
        AbstractMeasure
            The Measure object initialized with default values.

        Supported keys are:
        - `lhSurfaceHoles`, `rhSurfaceHoles`, and `SurfaceHoles`
           The number of holes in the surfaces.
        - `lhPialTotal`, and `rhPialTotal`
          The volume enclosed in the pial surfaces.
        - `lhWhiteMatterVol`, and `rhWhiteMatterVol`
          The Volume of the white matter in the segmentation (incl. lateralized
          WM-hypo).
        - `lhWhiteMatterTotal`, and `rhWhiteMatterTotal`
          The volume enclosed in the white matter surfaces.
        - `lhCortex`, `rhCortex`, and `Cortex`
          The volume between the pial and the white matter surfaces.
        - `CorpusCallosumVol`
          The volume of the corpus callosum in the segmentation.
        - `lhWM-hypointensities`, and `rhWM-hypointensities`
          The volume of unlateralized the white matter hypointensities in the
          segmentation, but lateralized by neigboring voxels
          (FreeSurfer uses talairach coordinates to re-lateralize).
        - `lhCerebralWhiteMatter`, `rhCerebralWhiteMatter`, and `CerebralWhiteMatter`
          The volume of the cerebral white matter in the segmentation (including corpus
          callosum split evenly into left and right and white matter and WM-hypo).
        - `CerebellarGM`
          The volume of the cerbellar gray matter in the segmentation.
        - `CerebellarWM`
          The volume of the cerbellar white matter in the segmentation.
        - `SubCortGray`
          The volume of the subcortical gray matter in the segmentation.
        - `TotalGray`
          The total gray matter volume in the segmentation.
        - `TFFC`
          The volume of the 3rd-5th ventricles and CSF in the segmentation.
        - `VentricleChoroidVol`
          The volume of the choroid plexus and inferiar and lateral ventricles and CSF.
        - `BrainSeg`
          The volume of all brains structres in the segmentation.
        - `BrainSegNotVent`, and `BrainSegNotVentSurf`
          The brain segmentation volume without ventricles.
        - `Cerebellum`
          The total cerebellar volume.
        - `SupraTentorial`, `SupraTentorialNotVent`, and `SupraTentorialNotVentVox`
          The supratentorial brain volume/voxel count (without centricles and CSF).
        - `Mask`
          The volume of the brain mask.
        - `EstimatedTotalIntraCranialVol`
          The eTIV estimate (via talairach registration).
        - `BrainSegVol-to-eTIV`, and `MaskVol-to-eTIV`
          The ratios of the brain segmentation volume and the mask volume with respect
          to the eTIV estimate.
        """

        hemi = key[:2]
        side = "Left" if hemi != "rh" else "Right"
        cc_classes = tuple(range(251, 256))
        if key in ("lhSurfaceHoles", "rhSurfaceHoles"):
            # FastSurfer and FS7 are same
            # l/rSurfaceHoles: (1-lheno/2) -- Euler number of /surf/l/rh.orig.nofix
            return SurfaceHoles(
                Path(f"surf/{hemi}.orig.nofix"),
                f"{hemi}SurfaceHoles",
                f"Number of defect holes in {hemi} surfaces prior to fixing",
                "unitless",
            )
        elif key == "SurfaceHoles":
            # sum of holes in left and right surfaces
            return DerivedMeasure(
                ["rhSurfaceHoles", "lhSurfaceHoles"],
                "SurfaceHoles",
                "Total number of defect holes in surfaces prior to fixing",
                measure_host=self,
            )
        elif key in ("lhPialTotal", "rhPialTotal"):
            # FastSurfer and FS7 are same
            return SurfaceVolume(
                Path(f"surf/{hemi}.pial"),
                f"{hemi}PialTotalVol",
                f"{side} hemisphere total pial volume",
                "mm^3",
            )
        elif key in ("lhWhiteMatterVol", "rhWhiteMatterVol"):
            # This is volume-based in FS7 (ComputeBrainVolumeStats2)
            if key[:1] == "l":
                classes = (2, 78)
            else:  # r
                classes = (41, 79)
            return self.voxel_class(
                classes,
                f"{hemi}WhiteMatterVol",
                f"{side} hemisphere total white matter volume",
                "mm^3",
            )
        elif key in ("lhWhiteMatterTotal", "rhWhiteMatterTotal"):
            return SurfaceVolume(
                Path(f"surf/{hemi}.white"),
                f"{hemi}WhiteMatterSurfVol",
                f"{side} hemisphere total white matter volume",
                "mm^3",
            )
        elif key in ("lhCortex", "rhCortex"):
            # From https://github.com/freesurfer/freesurfer/blob/
            # 3753f8a1af484ac2507809c0edf0bc224bb6ccc1/utils/cma.cpp#L1190C1-L1192C52
            # CtxGM = everything inside pial surface minus everything in white surface.
            parents = [f"{hemi}PialTotal", (-1, f"{hemi}WhiteMatterTotal")]
            # With version 7, don't need to do a correction because the pial surface is
            # pinned to the white surface in the medial wall
            return DerivedMeasure(
                parents,
                f"{hemi}CortexVol",
                f"{side} hemisphere cortical gray matter volume",
                measure_host=self,
            )
        elif key == "Cortex":
            # 7 => lhCtxGM + rhCtxGM: sum of left and right cerebral GM
            return DerivedMeasure(
                ["lhCortex", "rhCortex"],
                "CortexVol",
                "Total cortical gray matter volume",
                measure_host=self,
            )
        elif key == "CorpusCallosumVol":
            # FastSurfer and FS7 are same
            # CCVol:
            # CC_Posterior CC_Mid_Posterior CC_Central CC_Mid_Anterior CC_Anterior
            return self.voxel_class(
                cc_classes,
                "CorpusCallosumVol",
                "Volume of the Corpus Callosum",
                "mm^3",
            )
        elif key in ("lhWM-hypointensities", "rhWM-hypointensities"):
            # lateralized counting of class 77 WM hypo intensities
            def mask_77_lat(arr):
                """
                This function returns a lateralized mask of hypo-WM (class 77).

                This is achieved by looking at surrounding labels and associating them
                with left or right (this is not 100% robust when there is no clear
                classes with left aseg labels present, but it is cheap to perform.
                """
                mask = arr == 77
                left_aseg = (2, 4, 5, 7, 8, 10, 11, 12, 13, 17, 18, 26, 28, 30, 31)
                is_left = mask_in_array(arr, left_aseg)
                from scipy.ndimage import uniform_filter
                is_left = uniform_filter(is_left.astype(np.float32), size=7) > 0.2
                is_side = np.logical_not(is_left) if hemi == "rh" else is_left
                return np.logical_and(mask, is_side)

            return VolumeMeasure(
                self._seg_from_file,
                mask_77_lat,
                f"{side}WhiteMatterHypoIntensities",
                f"Volume of {side} White matter hypointensities",
                "mm^3"
            )
        elif key in ("lhCerebralWhiteMatter", "rhCerebralWhiteMatter"):
            # SurfaceVolume
            # 9/10 => l/rCerebralWM
            parents = [
                f"{hemi}WhiteMatterVol",
                f"{hemi}WM-hypointensities",
                (0.5, "CorpusCallosumVol"),
            ]
            return DerivedMeasure(
                parents,
                f"{hemi}CerebralWhiteMatterVol",
                f"{side} hemisphere cerebral white matter volume",
                measure_host=self,
            )
        elif key == "CerebralWhiteMatter":
            # 11 => lhCtxWM + rhCtxWM: sum of left and right cerebral WM
            return DerivedMeasure(
                ["rhCerebralWhiteMatter", "lhCerebralWhiteMatter"],
                "CerebralWhiteMatterVol",
                "Total cerebral white matter volume",
                measure_host=self,
            )
        elif key == "CerebellarGM":
            # Left-Cerebellum-Cortex Right-Cerebellum-Cortex Cbm_Left_I_IV
            # Cbm_Right_I_IV Cbm_Left_V Cbm_Right_V Cbm_Left_VI Cbm_Vermis_VI
            # Cbm_Right_VI Cbm_Left_CrusI Cbm_Vermis_CrusI Cbm_Right_CrusI
            # Cbm_Left_CrusII Cbm_Vermis_CrusII Cbm_Right_CrusII Cbm_Left_VIIb
            # Cbm_Vermis_VIIb Cbm_Right_VIIb Cbm_Left_VIIIa Cbm_Vermis_VIIIa
            # Cbm_Right_VIIIa Cbm_Left_VIIIb Cbm_Vermis_VIIIb Cbm_Right_VIIIb
            # Cbm_Left_IX Cbm_Vermis_IX Cbm_Right_IX Cbm_Left_X Cbm_Vermis_X Cbm_Right_X
            # Cbm_Vermis_VII Cbm_Vermis_VIII Cbm_Vermis
            cerebellum_classes = [8, 47]
            cerebellum_classes.extend(range(601, 629))
            cerebellum_classes.extend(range(630, 633))
            return self.voxel_class(
                cerebellum_classes,
                "CerebellarGMVol",
                "Cerebellar gray matter volume",
                "mm^3",
            )
        elif key == "CerebellarWM":
            # Left-Cerebellum-White-Matter Right-Cerebellum-White-Matter
            cerebellum_classes = [7, 46]
            return self.voxel_class(
                cerebellum_classes,
                "CerebellarWMVol",
                "Cerebellar white matter volume",
                "mm^3",
            )
        elif key == "SubCortGray":
            # 4 => SubCortGray
            # Left-Thalamus Right-Thalamus Left-Caudate Right-Caudate Left-Putamen
            # Right-Putamen Left-Pallidum Right-Pallidum Left-Hippocampus
            # Right-Hippocampus Left-Amygdala Right-Amygdala Left-Accumbens-area
            # Right-Accumbens-area Left-VentralDC Right-VentralDC Left-Substantia-Nigra
            # Right-Substantia-Nigra
            subcortgray_classes = [17, 18, 26, 27, 28, 58, 59, 60]
            subcortgray_classes.extend(range(10, 14))
            subcortgray_classes.extend(range(49, 55))
            return self.voxel_class(
                subcortgray_classes,
                "SubCortGrayVol",
                "Subcortical gray matter volume",
                "mm^3",
            )
        elif key == "TotalGray":
            # FastSurfer, FS6 and FS7 are same
            # 8 => TotalGMVol: sum of SubCortGray., Cortex and Cerebellar GM
            return DerivedMeasure(
                ["SubCortGray", "Cortex", "CerebellarGM"],
                "TotalGrayVol",
                "Total gray matter volume",
                measure_host=self,
            )
        elif key == "TFFC":
            # FastSurfer, FS6 and FS7 are same
            # TFFC:
            # 3rd-Ventricle 4th-Ventricle 5th-Ventricle CSF
            tffc_classes = (14, 15, 72, 24)
            return self.voxel_class(
                tffc_classes,
                "Third-Fourth-Fifth-CSF",
                "volume of 3rd, 4th, 5th ventricle and CSF",
                "mm^3",
            )
        elif key == "VentricleChoroidVol":
            # FastSurfer, FS6 and FS7 are same, except FS7 adds a KeepCSF flag, which
            # excludes CSF (but not by default)
            # 15 => VentChorVol:
            # Left-Choroid-Plexus Right-Choroid-Plexus Left-Lateral-Ventricle
            # Right-Lateral-Ventricle Left-Inf-Lat-Vent Right-Inf-Lat-Vent
            ventchor_classes = (4, 5, 31, 43, 44, 63)
            return self.voxel_class(
                ventchor_classes,
                "VentricleChoroidVol",
                "Volume of ventricles and choroid plexus",
                "mm^3",
            )
        elif key in "BrainSeg":
            # 0 => BrainSegVol:
            # FS7 (does mot use ribbon any more, just )
            #   not background, in aseg ctab, not Brain stem, not optic chiasm,
            #   aseg undefined in aseg ctab and not cortex or WM (L/R Cerebral
            #   Ctx/WM)
            # ComputeBrainStats2 also removes any regions that are not part of the
            # AsegStatsLUT.txt
            # background, brainstem, optic chiasm: 0, 16, 85
            brain_seg_classes = [2, 3, 4, 5, 7, 8]
            brain_seg_classes.extend(range(10, 16))
            brain_seg_classes.extend([17, 18, 24, 26, 28, 30, 31])
            brain_seg_classes.extend(range(41, 55))
            brain_seg_classes.remove(45)
            brain_seg_classes.remove(48)
            brain_seg_classes.extend([58, 60, 62, 63, 72])
            brain_seg_classes.extend(range(77, 83))
            brain_seg_classes.extend(cc_classes)
            if not self._fs_compat:
                # also add asegdkt regions 1002-1035, 2002-2035
                brain_seg_classes.extend(range(1002, 1032))
                brain_seg_classes.remove(1004)
                brain_seg_classes.extend((1034, 1035))
                brain_seg_classes.extend(range(2002, 2032))
                brain_seg_classes.remove(2004)
                brain_seg_classes.extend((2034, 2035))
            return self.voxel_class(
                brain_seg_classes,
                "BrainSegVol",
                "Brain Segmentation Volume",
                "mm^3",
            )
        elif key in ("BrainSegNotVent", "BrainSegNotVentSurf"):
            # FastSurfer, FS6 and FS7 are same
            # 1 => BrainSegNotVent: BrainSegVolNotVent (BrainSegVol-VentChorVol-TFFC)
            return DerivedMeasure(
                ["BrainSeg", (-1, "VentricleChoroidVol"), (-1, "TFFC")],
                key.replace("SegNot", "SegVolNot"),
                "Brain Segmentation Volume Without Ventricles",
                measure_host=self,
            )
        elif key == "Cerebellum":
            return DerivedMeasure(
                ("CerebellarGM", "CerebellarWM"),
                "CerebellumVol",
                "Cerebellar volume",
                measure_host=self,
            )
        elif key == "SupraTentorial":
            parents = ["BrainSeg", (-1.0, "Cerebellum")]
            return DerivedMeasure(
                parents,
                "SupraTentorialVol",
                "Supratentorial volume",
                measure_host=self,
            )
        elif key == "SupraTentorialNotVent":
            # 3 => SupraTentVolNotVent: SupraTentorial w/o Ventricles & Choroid Plexus
            parents = ["SupraTentorial", (-1, "VentricleChoroidVol"), (-1, "TFFC")]
            return DerivedMeasure(
                parents,
                "SupraTentorialVolNotVent",
                "Supratentorial volume",
                measure_host=self,
            )
        elif key == "SupraTentorialNotVentVox":
            # 3 => SupraTentVolNotVent: SupraTentorial w/o Ventricles & Choroid Plexus
            return DerivedMeasure(
                ["SupraTentorialNotVent"],
                "SupraTentorialVolNotVentVox",
                "Supratentorial volume voxel count",
                operation="by_vox_vol",
                measure_host=self,
            )
        elif key == "Mask":
            # 12 => MaskVol: Any voxel in mask > 0
            return MaskMeasure(
                Path("mri/brainmask.mgz"),
                "MaskVol",
                "Mask Volume",
                "mm^3",
            )
        elif key == "EstimatedTotalIntraCranialVol":
            # atlas_icv: eTIV from talairach transform determinate
            return ETIVMeasure(
                Path("mri/transforms/talairach.xfm"),
                "eTIV",
                "Estimated Total Intracranial Volume",
                "mm^3",
            )
        elif key == "BrainSegVol-to-eTIV":
            # 0/atlas_icv: ratio BrainSegVol to eTIV
            return DerivedMeasure(
                ["BrainSeg", "EstimatedTotalIntraCranialVol"],
                "BrainSegVol-to-eTIV",
                "Ratio of BrainSegVol to eTIV",
                measure_host=self,
                operation="ratio",
            )
        elif key == "MaskVol-to-eTIV":
            # 12/atlas_icv: ratio Mask to eTIV
            return DerivedMeasure(
                ["Mask", "EstimatedTotalIntraCranialVol"],
                "MaskVol-to-eTIV",
                "Ratio of MaskVol to eTIV",
                measure_host=self,
                operation="ratio",
            )

    def __iter__(self) -> list[AbstractMeasure]:
        """
        Iterate through all measures that are exported directly or indirectly.
        """

        out = [self[name] for name in self._exported_measures]
        i = 0
        while i < len(out):
            this = out[i]
            if isinstance(this, DerivedMeasure):
                out.extend(filter(lambda x: x not in out, this.parents_items()))
            i += 1
        return out

    def compute_non_derived_pv(
            self,
            compute_threads: Executor | None = None
    ) -> "list[Future[int | float]]":
        """
        Trigger computation of all non-derived, non-pv measures that are required.

        Parameters
        ----------
        compute_threads : concurrent.futures.Executor, optional
            An Executor object to perform the computation of measures, if an Executor
            object is passed, the computation of measures is submitted to the Executor
            object. If not, measures are computed in the main thread.

        Returns
        -------
        list[Future[int | float]]
            For each non-derived and non-PV measure, a future object that is associated
            with the call to the measure.
        """

        def run(f: Callable[[], int | float]) -> Future[int | float]:
            out = Future()
            out.set_result(f())
            return out

        if isinstance(compute_threads, Executor):
            run = compute_threads.submit

        invalid_types = (DerivedMeasure, PVMeasure)
        self._compute_futures = [
            run(this) for this in self.values() if not isinstance(this, invalid_types)
        ]
        return self._compute_futures

    def needs_pv_calculation(self) -> bool:
        """
        Returns whether the manager has PV-dependent measures.

        Returns
        -------
        bool
            Whether the manager has PVMeasure children.
        """
        return any(isinstance(this, PVMeasure) for this in self.values())

    def get_virtual_labels(self, label_pool: Iterable[int]) -> dict[int, list[int]]:
        """
        Get the virtual substitute labels that are required.

        Parameters
        ----------
        label_pool : Iterable[int]
            An iterable over available labels.

        Returns
        -------
        dict[int, list[int]]
            A dictionary of key-value pairs of new label and a list of labels this
            represents.
        """
        lbls = (this.labels() for this in self.values() if isinstance(this, PVMeasure))
        no_duplicate_dict = {self.__to_lookup(labs): labs for labs in lbls}
        return dict(zip(label_pool, no_duplicate_dict.values(), strict=False))

    @staticmethod
    def __to_lookup(labels: Sequence[int]) -> str:
        return str(list(sorted(set(map(int, labels)))))

    def update_pv_from_table(
            self,
            dataframe: "pd.DataFrame",
            merged_labels: dict[int, list[int]],
    ) -> "pd.DataFrame":
        """
        Update pv measures from dataframe and remove corresponding entries from the
        dataframe.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The dataframe object with the PV values.
        merged_labels : dict[int, list[int]]
            Mapping from PVMeasure proxy label to list of labels it merges.

        Returns
        -------
        pd.DataFrame
            A dataframe object, where label 'groups' used for updates and in
            `merged_labels` are removed, i.e. those labels added for PVMeasure objects.

        Raises
        ------
        RuntimeError
        """
        _lookup = {self.__to_lookup(ml): vl for vl, ml in merged_labels.items()}
        filtered_df = dataframe
        # go through the pv measures and find a measure that has the same list
        for this in self.values():
            if isinstance(this, PVMeasure):
                virtual_label = _lookup.get(self.__to_lookup(this.labels()), None)
                if virtual_label is None:
                    raise RuntimeError(f"Could not find the virtual label for {this}.")
                row = dataframe[dataframe["SegId"] == virtual_label]
                if row.shape[0] != 1:
                    raise RuntimeError(
                        f"The search results in the dataframe for {this} failed: "
                        f"shape {row.shape}"
                    )
                this.update_data(row)
                filtered_df = filtered_df[filtered_df["SegId"] != virtual_label]

        return filtered_df

    def wait_compute(self) -> Sequence[BaseException]:
        """
        Wait for all pending computation processes and return their errors.

        Also resets the internal compute futures.

        Returns
        -------
        Sequence[BaseException]
            The errors raised in the computations.
        """
        errors = [future.exception() for future in self._compute_futures]
        self._compute_futures = []
        return [error for error in errors if error is not None]

    def wait_write_brainvolstats(self, brainvol_statsfile: Path):
        """
        Wait for measure computation to finish and write results to brainvol_statsfile.

        Parameters
        ----------
        brainvol_statsfile: Path
            The file to write the measures to.

        Raises
        ------
        RuntimeError
            If errors occurred during measure computation.
        """
        errors = list(self.wait_compute())
        if len(errors) != 0:
            error_messages = ["Some errors occurred during measure computation:"]
            error_messages.extend(map(lambda e: str(e.args[0]), errors))
            raise RuntimeError("\n - ".join(error_messages))

        def fmt_measure(key: str, data: MeasureTuple) -> str:
            return f"# Measure {key}, {data[0]}, {data[1]}, {data[2]:.12f}, {data[3]}"

        lines = self.format_measures(fmt_func=fmt_measure)

        with open(brainvol_statsfile, "w") as file:
            for line in lines:
                print(line, file=file)
