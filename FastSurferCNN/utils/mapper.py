# Copyright 2023 Image Analysis Lab, German Center for Neurodegenerative Diseases(DZNE), Bonn
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
Original Author: David Kuegler.

Date: Aug-19-2022
"""

import json
import os.path
from functools import partial, partialmethod, reduce
from numbers import Integral, Number
from typing import (
    Any,
    Callable,
    Collection,
    Container,
    Dict,
    Generic,
    Hashable,
    Iterable,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Set,
    TextIO,
    Tuple,
    TypeVar,
    Union,
    cast,
    overload,
)

import numpy as np
import pandas
import torch
from matplotlib.pyplot import get_cmap
from matplotlib.colors import Colormap
from numpy import typing as npt

from FastSurferCNN.utils import logging

_SYMBOLS = [
    "ColorLookupTable",
    "JsonColorLookupTable",
    "TSVLookupTable",
    "Mapper",
    "ClassMapper",
]


KT = TypeVar("KT")
VT = TypeVar("VT")
T_OtherValue = TypeVar("T_OtherValue")
NT = TypeVar("NT", bound=Number)
AT = TypeVar("AT", npt.NDArray[Number], torch.Tensor)

ColorTuple = Tuple[float, float, float]
ColormapGenerator = Callable[[int], npt.NDArray[float]]

logger = logging.getLogger(__name__)

LabelImageType = TypeVar("LabelImageType", torch.Tensor, npt.NDArray[int])


def is_int(a_object) -> bool:
    """
    Check whether the array_or_tensor is an integer.

    Parameters
    ----------
    a_object : Any
        The object to check its type.

    Returns
    -------
    bool
        Whether the data type of the object is int or not.
    """
    from collections.abc import Collection

    if isinstance(a_object, np.ndarray):
        return np.issubdtype(a_object.dtype, np.integer)
    elif isinstance(a_object, torch.Tensor):
        return not a_object.dtype.is_floating()
    elif isinstance(a_object, Number):
        return isinstance(a_object, Integral)
    elif isinstance(a_object, Collection):
        return all(is_int(obj) for obj in a_object)
    else:
        return False


def to_same_type(data, type_hint: AT) -> AT:
    """
    Convert data to the same type as type_hint.

    Parameters
    ----------
    data : Any
        The data to convert.
    type_hint : AT
        Hint for the data type.

    Returns
    -------
    AT
        Data converted to the same type as specified by type_hint.
    """
    if torch.is_tensor(type_hint) and not torch.is_tensor(data):
        return torch.as_tensor(data, dtype=type_hint.dtype, device=type_hint.device)
    elif isinstance(type_hint, np.ndarray) and not isinstance(data, np.ndarray):
        if isinstance(data, torch.Tensor):
            return data.cpu().numpy().astype(type_hint.dtype)
        return np.asarray(data, dtype=type_hint.dtype)
    else:
        return data


class Mapper(Generic[KT, VT]):
    """
    Map from one label space to a generic 'label'-space.
    """

    _map_dict: Dict[KT, npt.NDArray[VT]]
    _label_shape: Tuple[int, ...]
    _map_np: npt.NDArray[VT]
    _map_torch: torch.Tensor
    _max_label: Optional[int]
    _name: str

    def __init__(
        self, mappings: Mapping[KT, Union[VT, npt.NDArray[VT]]], name: str = "undefined"
    ):
        """
        Construct `Mapper` object from a mappings dictionary.

        Parameters
        ----------
        mappings : Mapping[KT, Union[VT, npt.NDArray[VT]]]
            A dictionary of labels from -> to mappings.
        name : str
            Name for messages (default: "undefined").
        """
        if len(mappings) == 0:
            raise RuntimeError("The mappings object is empty.")

        self._name = name
        self._map_dict = dict(mappings)  # this also copies
        self._label_shape = np.asarray(list(self._map_dict.values())).shape[1:]

        keys = self.source_space  # this converts to set
        if all(isinstance(k, Number) for k in keys):
            _keys = np.asarray(list(keys))
            extreme_labels = _keys.min(initial=0), _keys.max(initial=0)
            if extreme_labels[0] < 0:
                raise NotImplementedError(
                    f"Negative classes are not supported (min: {extreme_labels[0]}, "
                    f"max: {extreme_labels[1]})"
                )
            self._max_label = max(0, extreme_labels[1])
        elif all(isinstance(k, str) for k in keys):
            self._max_label = None  # invalid
        else:
            raise TypeError(
                "Invalid type for keys of the mapping, must either be all numeric or all strings."
            )

    @property
    def name(self) -> str:
        """
        Return the name of the mapper.
        """
        return self._name

    @name.setter
    def name(self, name: str):
        """
        Set the name.
        """
        self._name = name

    @property
    def source_space(self) -> Set[KT]:
        """
        Return a set of labels the mapper accepts.
        """
        return set(self._map_dict.keys())

    @property
    def target_space(self) -> Collection[VT]:
        """
        Return the set of labels the mapper converts to as a set of python-natives (if possible), arrays expanded to tuples.
        """
        return self._map_dict.values()

    @property
    def max_label(self) -> int:
        """
        Return the max label.
        """
        if self._max_label is None:
            raise RuntimeError("max_label is only valid for integer keys.")
        return self._max_label

    def update(
        self, other: "Mapper[KT, VT]", overwrite: bool = True
    ) -> "Mapper[KT, VT]":
        """
        Merge another map into this mapper.

        Parameters
        ----------
        other : Mapper[KT, VT]
            The other Mapper object whose key-value pairs are to be added to this Mapper object.
        overwrite : bool, default=True
            Flag to overwrite value if key already exists in Mapper object (Default value = True).

        Returns
        -------
        Mapper[KT, VT]
            Mapper after merge.
        """
        for key, value in iter(other):
            if overwrite or key not in self._map_dict:
                self._map_dict[key] = value

        # reset the internal numpy/pytorch mapping constructs
        if hasattr(self, "_map_np"):
            delattr(self, "_map_np")
        if hasattr(self, "_map_torch"):
            delattr(self, "_map_torch")
        return self

    __iadd__ = partialmethod(update, overwrite=True)

    def map(self, image: AT, out: Optional[AT] = None) -> AT:
        """
        Forward map the labels from prediction to internal space.

        Parameters
        ----------
        image : AT
            Data to map to internal space.
        out : Optional[AT]
            Output array for performance.
            Returns an `numpy.ndarray` with mapped values. (Default value = None).

        Returns
        -------
        AT
            Data after being mapped to the internal space.
        """
        # torch sparse tensors can't index with images
        # self._map = _b.torch.sparse_coo_tensor(src_labels, labels, (self._max_label,) + self._label_shape)

        out_type = image if out is None else out

        if self._max_label is None:
            # if we do not have numbers, we need to map all entries individually
            return self._map_py(image, out)

        # the key is an integer
        map_shape = (self._max_label + 1,) + self._label_shape
        init_map = True

        if isinstance(out_type, np.ndarray):
            if not hasattr(self, "_map_np"):
                if self._max_label > 4096:
                    from scipy import sparse

                    lil = (
                        sparse.lil_array
                        if hasattr(sparse, "lil_array")
                        else sparse.lil_matrix
                    )
                    self._map_np = cast(npt.NDArray[VT], lil(map_shape))
                else:
                    self._map_np = np.zeros(map_shape, dtype=out_type.dtype)
            else:
                init_map = False
            _map = self._map_np
        elif torch.is_tensor(out_type):
            if not hasattr(self, "_map_torch"):
                self._map_torch = out_type.new_full(
                    (self._max_label + 1,) + self._label_shape, fill_value=0
                )
            else:
                init_map = False
            _map = self._map_torch
        else:
            raise TypeError("image or out are an invalid type.")

        if init_map:
            for k, v in self._map_dict.items():
                _map[k] = to_same_type(v, type_hint=out_type)

        try:
            mapped = _map[image]
        except IndexError as e:
            map_unique = (
                np.unique(_map) if isinstance(_map, np.ndarray) else _map.unique()
            )
            missing_keys = set(map_unique.tolist()) - set(self._map_dict.keys())
            raise KeyError(f"Could not find the mapping of keys {missing_keys}.") from e

        if torch.is_tensor(mapped) and (
            hasattr(mapped, "is_sparse") and mapped.is_sparse
        ):
            mapped.to_dense()
        try:
            # noinspection PyUnresolvedReferences
            from scipy import sparse

            if isinstance(mapped, sparse.spmatrix):
                mapped = mapped.toarray()
        except ImportError:
            pass

        if out is not None:
            out[:] = to_same_type(mapped, type_hint=out)
            return out
        return to_same_type(mapped, type_hint=image)

    def _map_py(self, image: AT, out: Optional[AT] = None) -> AT:
        """
        Map internally by python, for example for strings.

        Parameters
        ----------
        image : AT
            Image data.
        out : Optional[AT]
            Output data. Optional (Default value = None).

        Returns
        -------
        AT
            Image data after being mapped.
        """
        out_type = image if out is None else out
        if out is None:

            def _internal_map(img):
                return [
                    self._map_dict[v] if img.ndim == 1 else _internal_map(v)
                    for v in img
                ]

            return to_same_type(_internal_map(image), type_hint=out_type)
        else:

            def _internal_map(img, o):
                if img.ndim == 1:
                    o[:] = to_same_type(
                        [self._map_dict[v] for v in image], type_hint=out_type
                    )
                else:
                    for i in range(img.shape[0]):
                        _internal_map(img[i], out[i])

            _internal_map(image, out)
            return out

    def __call__(
        self, image: AT, label_image: Union[npt.NDArray[KT], torch.Tensor]
    ) -> Tuple[AT, Union[npt.NDArray, torch.Tensor]]:
        """
        Transform a dataset from prediction to internal space for sets of image and segmentation.

        Parameters
        ----------
        image : AT
            Image - will stay same.
        label_image : Union[npt.NDArray[KT], torch.Tensor]
            Data to map to internal space
            Returns two `numpy.ndarray`s with image and mapped values.

        Returns
        -------
        image : image
            Image.
        Union[npt.NDArray, torch.Tensor]
            Mapped values.
        """
        return image, self.map(label_image)

    def reversed_dict(self) -> Mapping[VT, KT]:
        """
        Map dictionary from the target space to the source space.
        """
        rev_mappings = {}
        for src in sorted(self.source_space):
            a = self._map_dict[src]
            if not isinstance(a, Hashable):
                a = tuple(
                    a.tolist() if isinstance(a, (np.ndarray, torch.Tensor)) else a
                )
            rev_mappings.setdefault(a, src)
        return rev_mappings

    def __reversed__(self) -> "Mapper[VT, KT]":
        """
        Reverse map the original transformation (with non-bijective mappings mapping to the lower key).
        """
        return Mapper(self.reversed_dict(), name="reverse-" + self.name)

    def is_bijective(self) -> bool:
        """
        Return, whether the Mapper is bijective.
        """
        return len(self.source_space) == len(self.target_space)

    def __getitem__(self, item: KT) -> VT:
        """
        Return the value of the item.
        """
        return self._map_dict[item]

    def __iter__(self) -> Iterator[Tuple[KT, VT]]:
        """
        Create an iterator for the Mapper object.
        """
        return iter(self._map_dict.items())

    def __contains__(self, item: KT) -> bool:
        """
        Check whether the mapping contains the item.
        """
        return self._map_dict.__contains__(item)

    def chain(
        self, other_mapper: "Mapper[VT, T_OtherValue]"
    ) -> "Mapper[KT, T_OtherValue]":
        """
        Chain the current mapper with the `other_mapper`.

        This effectively is an optimization to first applying this
        mapper and then applying the `other_mapper`.

        Parameters
        ----------
        other_mapper : "Mapper[VT, T_OtherValue]"
            Mapper mapping from the target-space of this mapper to a new space.

        Returns
        -------
        Mapper : "Mapper[KT, T_OtherValue]"
            A mapper mapping from the input space of this mapper to the target-space of the `other_mapper`.
        """
        target_space = list(self.target_space)
        is_target_set = [not isinstance(t, Hashable) for t in target_space]
        if any(is_target_set):
            index = is_target_set.index(True)
            raise ValueError(
                f"The target space must be hashable, but {is_target_set.count(True)} values are not "
                f"hashable, for example {index}: {target_space[index]}."
            )
        target_space = set(target_space)
        if not target_space <= other_mapper.source_space:
            # test whether every element in self.target_space is also in other_mapper.source_space
            raise ValueError(
                f"The first set ({self.name}) maps to the following keys, that the second mapper "
                f"({other_mapper.name}) does not map from:\n  "
                + ", ".join(f"'{v}'" for v in target_space - other_mapper.source_space)
            )
        return Mapper(
            dict(
                (in_key, other_mapper[out_key])
                for in_key, out_key in self._map_dict.items()
            ),
            name=f"{self.name} -> {other_mapper.name}",
        )

    @classmethod
    def make_classmapper(
        cls,
        mappings: Dict[int, int],
        keep_labels: Sequence[int] = tuple(),
        compress_out_space: bool = False,
        name: str = "undefined",
    ) -> "Mapper[int, int]":
        """
        Map from one label space (int) to another (also int) using a mappings function.

        Can also be used as a transform.

        Creates a :class:`Mapper` object from a mappings dictionary and a
        list of labels to keep.

        Parameters
        ----------
        mappings : Dict[int, int]
            A dictionary of labels from -> to mappings.
        keep_labels : Sequence[int]
            A list of classes to keep after mapping, where all not included classes are not changed
            (default: empty).
        compress_out_space : bool
            Whether to reassign labels to reduce the maximum label (default: False).
        name : str
            Mame for messages (default: "undefined").

        Returns
        -------
        "Mapper[int, int]"
        A Mapper object that provides a mapping from one label space to another.

        Raises
        ------
        ValueError
            If keep_labels contains an entry > 65535.
        """
        if any(v not in keep_labels for v in mappings.values()):
            mappings.update(dict((k, k) for k in keep_labels))

        if compress_out_space:
            target_labels = dict(
                (v, i) for i, v in enumerate(sorted(set(mappings.values())))
            )

            def relabel(old_label_in: int, old_label_out: int) -> Tuple[int, int]:
                return old_label_in, target_labels[old_label_out]

            mappings = dict(map(relabel, mappings.items()))

        return Mapper(mappings, name=name)

    def _map_logits(
        self,
        logits: AT,
        axis: int = -1,
        reverse: bool = False,
        out: Optional[AT] = None,
        mode: Literal["logit", "prob"] = "logit",
    ) -> AT:
        """
        Map logits or probabilities with the Mapper.
        """
        if not is_int(self.source_space) or not is_int(self.target_space):
            raise ValueError("map_logits/map_probs requires a mapping from int to int.")

        if mode == "logit":
            reduce_func = (
                np.multiply.reduce
                if isinstance(logits, np.ndarray)
                else partial(reduce, torch.mul)
            )
            spread_func = (
                np.float_power if isinstance(logits, np.ndarray) else torch.pow
            )
        elif mode == "prob":
            reduce_func = (
                np.add.reduce
                if isinstance(logits, np.ndarray)
                else partial(reduce, torch.add)
            )
            spread_func = np.multiply if isinstance(logits, np.ndarray) else torch.mul
        else:
            raise ValueError(
                f"Unknown mode, should be in 'logit', 'prob', but was '{mode}'."
            )

        from collections import defaultdict

        data = defaultdict(list)
        mappings = self._map_dict.items()
        if reverse:
            unique_target_classes = np.unique(
                list(self._map_dict.values()), return_counts=True
            )
            cls_cts = {cls: cts for cls, cts in zip(*unique_target_classes) if cts > 1}
            mappings = ((v, k) for k, v in mappings)  # swap source and target mappings
        else:
            cls_cts = {}

        shape = list(logits.shape)
        ii = [slice(None) for _ in shape]
        for source_dim, target_dim in mappings:
            ii[axis] = source_dim
            this_logit = logits[tuple(ii)]
            cts = cls_cts.get(source_dim, 1)
            if cts > 1:
                this_logit = spread_func(this_logit, 1.0 / cts)
            data[target_dim].append(this_logit)

        stack = np.stack if isinstance(logits, np.ndarray) else torch.stack
        labels = np.unique(np.asarray(list(data.keys())))
        if not np.all(labels == np.arange(labels.size)):
            shape[axis] = 1
            alloc = (
                partial(np.zeros, like=logits)
                if isinstance(logits, np.ndarray)
                else partial(logits.new_full, fill_value=0)
            )
            zeros = alloc(shape)
        else:
            zeros = None

        data = {k: v[0] if len(v) == 1 else reduce_func(v) for k, v in data.items()}
        return stack(
            [
                data[i] if i in labels else zeros
                for i in range(labels.max(initial=-1) + 1)
            ],
            axis,
            out=out,
        )

    map_logits = partialmethod(_map_logits, mode="logit")
    map_probs = partialmethod(_map_logits, mode="prob")


class ColorLookupTable(Generic[KT]):
    """
    This class provides utility in creating color palettes from colormaps.
    """

    _color_palette: Optional[npt.NDArray[float]]
    _colormap: Union[str, Colormap, ColormapGenerator]
    _classes: Optional[List[KT]]
    _name: str

    def __init__(
        self,
        classes: Optional[Iterable[KT]] = None,
        color_palette: Union[Dict[KT, npt.ArrayLike], npt.ArrayLike, None] = None,
        colormap: Union[str, Colormap, ColormapGenerator] = "gist_ncar",
        name: Optional[str] = None,
    ):
        """
        Construct a LookupTable object.

        Parameters
        ----------
        classes : Optional[Iterable[KT]]
            Iterable of the classes. (Default value = None).
        color_palette : Union[Dict[KT, npt.ArrayLike], npt.ArrayLike], Optional
            colors associated with each class, either indexed by a dictionary (class -> Color) or by the
            order of classes in classes (default: None). (Default value = None).
        colormap : Union[str, Colormap, ColormapGenerator]
            Alternative to color_palette, uses a colormap to generate a color_palette automatically. Colormap
            can be string, matplotlib.Colormap or a function (num_classes -> NDArray of shape (num_classes, 3 or 4))
            (default: 'gist_ncar').
        name : Optional[str]
            Name for messages (default: "unnamed lookup table").
        """
        self._name = "unnamed lookup table" if name is None else name

        if (
            isinstance(colormap, Colormap)
            or callable(colormap)
            or isinstance(colormap, str)
        ):
            self._colormap = colormap
        else:
            raise TypeError("Invalid type for the colormap.")
        self.classes = classes
        if color_palette is not None:
            self.color_palette = color_palette

    @property
    def name(self) -> str:
        """
        Return the name of the mapper.
        """
        return self._name

    @name.setter
    def name(self, name: str):
        """
        Set the name.
        """
        self._name = name

    @property
    def classes(self) -> Optional[List[KT]]:
        """
        Return the classes.
        """
        return self._classes

    @classes.setter
    def classes(self, classes: Optional[Iterable[KT]]):
        """
        Set the classes and generates a color palette for the given classes.

        Will override a manually set color_palette.

        Parameters
        ----------
        classes : Optional[Iterable[KT]]
            Iterable of the classes.
        """
        if classes is None:
            # resetting the classes
            self._classes = None
            self._color_palette = None
        else:
            self._classes = list(classes)
            num = len(self._classes)
            self._color_palette = get_cmap(self._colormap, num)(list(range(num)))

    @property
    def color_palette(self) -> Optional[npt.NDArray[float]]:
        """
        Return the color palette if it exists.
        """
        return self._color_palette

    @color_palette.setter
    def color_palette(
        self, color_palette: Union[Dict[KT, npt.ArrayLike], npt.ArrayLike, None]
    ):
        """
        Set (or reset) the color palette of the LookupTable.
        """
        if color_palette is None:
            self._color_palette = None
        else:
            if isinstance(color_palette, dict):
                if self._classes is None:
                    raise RuntimeError(
                        "No classes defined, but setting a color_palette via dict."
                    )
                color_palette = np.asarray([color_palette[c] for c in self._classes])
            elif not isinstance(color_palette, np.ndarray):
                color_palette = [
                    np.asarray(plt, dtype=float) / 255 if is_int(plt) else plt
                    for plt in color_palette
                ]
                color_palette = np.asarray(color_palette)
            elif is_int(color_palette):
                color_palette = np.asarray(color_palette, dtype=float) / 255

            self._color_palette = color_palette

    def __getitem__(self, key: KT) -> Tuple[int, KT, Tuple[int, int, int, int], Any]:
        """
        Return index, key, colors and additional values for the key.

        Parameters
        ----------
        key : KT
        The key for which the information is to be retrieved.

        Raises
        -------
        ValueError
            If key is not in _classes.
        """
        index = self._classes.index(key)
        return self.getitem_by_index(index)

    def getitem_by_index(
        self, index: int
    ) -> Tuple[int, KT, Tuple[int, int, int, int], Any]:
        """
        Return index, key, colors and additional values for the key.
        """
        color = self.get_color_by_index(index, 255)
        return index, self._classes[index], color, None

    def get_color_by_index(self, index: int, base: NT = 1.0) -> Tuple[NT, NT, NT, NT]:
        """
        Return the color (r, g, b, a) tuple associated with the index in the passed base.
        """
        if self._color_palette is None:
            raise RuntimeError("No color_palette set")
        base_type = type(base)
        if not isinstance(base, Number):
            raise TypeError(f"base must be a number, but was {base_type.__name__}")
        _color = tuple(base_type(k.item() * base) for k in self._color_palette[index])
        if len(_color) > 3:
            color = _color[:4]
        elif len(_color) == 3:
            color = _color + (255,)
        else:
            raise RuntimeError(
                f"Invalid shape of the color_palette, only {len(_color)} elements."
            )
        return color

    def colormap(self) -> Mapper[KT, ColorTuple]:
        """
        Generate a Mapper object that maps classes to their corresponding colors.
        """
        if self._color_palette is None:
            raise RuntimeError("No color_palette set")
        return Mapper(
            dict(zip(self.classes, self.color_palette)), name="color-" + self.name
        )

    def labelname2index(self) -> Mapper[KT, int]:
        """
        Return a mapping between the key and the (consecutive) index it is associated with.

        This is the inverse of ColorLookupTable.classes.
        """
        return Mapper(
            dict(zip(self._classes, range(len(self._classes)))),
            name="index-" + self.name,
        )

    def labelname2id(self) -> Mapper[KT, Any]:
        """
        Return a mapping between the key and the value it is associated with.

        Mapper[KT, Any]
            Not implemented in the base class.

        Raises
        ------
        RuntimeError
            If no value is associated.
        """
        raise RuntimeError("The base class keeps no ids (only indexes).")


class JsonColorLookupTable(ColorLookupTable[KT]):
    """
    This class extends the ColorLookupTable to handle JSON data.
    """

    _data: Any

    def __init__(
        self,
        file_or_buffer,
        color_palette: Union[Dict[KT, npt.ArrayLike], npt.ArrayLike, None] = None,
        colormap: Union[str, Colormap, ColormapGenerator] = "gist_ncar",
        name: Optional[str] = None,
    ) -> None:
        """
        Construct a JsonLookupTable object from `file_or_buffer` passed.

        Parameters
        ----------
        file_or_buffer :
            A json object to read from.
        color_palette : Union[Dict[KT, npt.ArrayLike], npt.ArrayLike], Optional
            colors associated with each class, either indexed by an dictionary (class -> Color) or by the
            order of classes in classes (default: None).
        colormap : Union[str, Colormap, ColormapGenerator]
            Alternative to color_palette, uses a colormap to generate a color_palette automatically. Colormap
            can be string, matplotlib.Colormap or a function (num_classes -> NDArray of shape (num_classes, 3 or 4))
            (default: 'gist_ncar').
        name : Optional[str]
            Name for messages (default: fallback to file_or_buffer, if possible).
        """
        if isinstance(file_or_buffer, str) and file_or_buffer.lstrip().startswith("{"):
            self._data = json.loads(file_or_buffer)
            if name is None:
                name = "unnamed json buffer string"
        elif isinstance(file_or_buffer, (str, os.PathLike)):
            if os.path.exists(file_or_buffer):
                with open(file_or_buffer, "r") as file:
                    self._data = json.load(file)
            else:
                raise ValueError(f"The file {file_or_buffer} does not exist")
            if name is None:
                name = os.path.basename(file_or_buffer)
        elif isinstance(file_or_buffer, TextIO):
            self._data = json.load(file_or_buffer)
            if name is None:
                name = "unnamed json stream"
        else:
            raise TypeError(
                f"file_or_buffer should be either a file pointer or a string, but was a "
                f"{type(file_or_buffer).__name__}."
            )

        labels = self._get_labels()
        classes = list(labels.keys() if isinstance(labels, dict) else labels)
        if isinstance(self._data, dict) and color_palette is None:
            color_palette = self._data.get("colors", None)
            if isinstance(color_palette, dict):
                color_palette = [color_palette[cls] for cls in classes]
        if all(cls.isdigit() for cls in classes):
            classes = list(map(int, classes))
        if len(set(classes)) != len(classes):
            unique_classes, counts = np.unique(np.asarray(classes), return_counts=True)
            raise KeyError(
                f"Duplicate classes in source_space: {unique_classes[counts>1]}"
            )
        super(JsonColorLookupTable, self).__init__(
            classes=classes, color_palette=color_palette, colormap=colormap, name=name
        )

    def _get_labels(self) -> Union[Dict[KT, Any], Iterable[KT]]:
        """
        Return labels.
        """
        return (
            self._data["labels"]
            if isinstance(self._data, dict) and "labels" in self._data
            else self._data
        )

    def dataframe(self) -> pandas.DataFrame:
        """
        Converts the labels from the internal data dictionary to a pandas DataFrame.
        """
        if isinstance(self._data, dict) and "labels" in self._data:
            return pandas.DataFrame.from_dict(self._data["labels"])

    def __getitem__(self, key: KT) -> Tuple[int, KT, Tuple[int, int, int, int], Any]:
        """
        Index by the index position, unless either key or value are int.
        """
        labels = self._get_labels()
        index, key, color, _other = super(JsonColorLookupTable, self).__getitem__(key)
        if isinstance(labels, dict):
            _other = labels[str(key)]
        return index, key, color, _other

    def labelname2id(self) -> Mapper[KT, Any]:
        """
        Return a mapping between the key and the value it is associated with.

        Returns
        -------
        Mapper[KT, Any]
            A Mapper object that provides a mapping between label names (keys) and their corresponding IDs (values).

        Raises
        ------
        RuntimeError
            If no value is associated.
        """
        labels = self._get_labels()
        if not isinstance(labels, dict):
            raise RuntimeError("The json file contained no values.")
        return Mapper(
            dict(zip(self._classes, labels.values())), name="value-" + self.name
        )


class TSVLookupTable(ColorLookupTable[str]):
    """
    This class extends the ColorLookupTable to handle TSV (Tab Separated Values) data.
    """

    _data: pandas.DataFrame

    def __init__(
        self,
        file_or_buffer,
        name: Optional[str] = None,
        header: bool = False,
        add_background: bool = True,
    ) -> None:
        """
        Create a CSVLookupTable object from `file_or_buffer` passed.

        Parameters
        ----------
        file_or_buffer :
            A `pandas`-compatible object to read from. Refer to :func:`pandas.read_csv` for additional
            documentation.
        name : str, Optional
            Name for messages (default: fallback to file_or_buffer, if possible).
        header : bool
            Whether the TSV file has a header line (default: False).
        add_background : bool
            Whether to add a label for background (default: True).
        """
        if name is None:
            if isinstance(file_or_buffer, str):
                if not os.path.exists(file_or_buffer):
                    name = "unnamed buffer string"
                else:
                    name = os.path.basename(file_or_buffer)
            else:
                name = "unnamed stream"

        names = {
            "ID": "int",
            "Label name": "str",
            "Red": "int",
            "Green": "int",
            "Blue": "int",
            "Alpha": "int",
        }

        self._data = pandas.read_csv(
            file_or_buffer,
            delim_whitespace=True,
            index_col=0,
            skip_blank_lines=True,
            comment="#",
            header=int(header),
            names=names.keys(),
            dtype=names,
        )
        if not (self._data.index == 0).any():
            df = pandas.DataFrame.from_dict(
                {k: [v] for k, v in zip(names.keys(), [0, "Unknown", 0, 0, 0, 0])}
            )
            self._data = pandas.concat([df, self._data])
        classes = self._data["Label name"].tolist()
        channels = ["Red", "Green", "Blue", "Alpha"]
        color_palette = np.asarray(
            [tuple(int(row[k].item()) for k in channels) for row in self._data.iloc]
        )
        super(TSVLookupTable, self).__init__(
            classes=classes, color_palette=color_palette, name=name
        )

    def getitem_by_index(
        self, index: int
    ) -> Tuple[int, str, Tuple[int, int, int, int], int]:
        """
        Find the Entry associated by a No.

        Parameters
        ----------
        index : int
            The index
            Returns a tuple of the index, the label, and a tuple of the RGBA color label.

        Returns
        -------
        index : int
            The index of the entry.
        key : str
            The label name associated with the entry.
        color : Tuple[int, int, int, int]
            The RGBA color label associated with the entry.
        int
            The data index associated with the entry.
        """
        index, key, color, _ = super(TSVLookupTable, self).getitem_by_index(index)
        return index, key, color, self._data.iloc[index].name

    def dataframe(self) -> pandas.DataFrame:
        """
        Return the raw panda data object.
        """
        return self._data

    def labelname2id(self) -> Mapper[KT, Any]:
        """
        Return a Mapper between the key and the value it is associated with.

        Returns
        -------
        Mapper[KT, Any]
            A Mapper object that links keys to their corresponding values based on the class and data index.

        Raises
        ------
        RuntimeError
            If no value is associated.
        """
        return Mapper(
            dict(zip(self._classes, self._data.index)), name="value-" + self.name
        )
