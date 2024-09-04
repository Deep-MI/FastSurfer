from collections.abc import Callable, Mapping
from dataclasses import (
    KW_ONLY,
    MISSING,
    Field,
    FrozenInstanceError,
    InitVar,
    asdict,
    astuple,
    dataclass,
    fields,
    is_dataclass,
    make_dataclass,
    replace,
)
from dataclasses import (
    field as _field,
)
from typing import Any, TypeVar, overload

__all__ = [
    "field",
    "asdict",
    "astuple",
    "dataclass",
    "fields",
    "Field",
    "FrozenInstanceError",
    "get_field",
    "is_dataclass",
    "InitVar",
    "make_dataclass",
    "MISSING",
    "KW_ONLY",
    "replace",
]

_T = TypeVar("_T")


@overload
def field(
        *,
        default: _T,
        help: str = "",
        flags: tuple[str] = (),
        init: bool = True,
        repr: bool = True,
        hash: bool | None = None,
        compare: bool = True,
        metadata: Mapping[Any, Any] | None = None,
        kw_only: bool = ...,
) -> _T: ...


@overload
def field(
        *,
        default_factory: Callable[[], _T],
        help: str = "",
        flags: tuple[str] = (),
        init: bool = True,
        repr: bool = True,
        hash: bool | None = None,
        compare: bool = True,
        metadata: Mapping[Any, Any] | None = None,
        kw_only: bool = ...,
) -> _T: ...


@overload
def field(
        *,
        help: str = "",
        flags: tuple[str] = (),
        init: bool = True,
        repr: bool = True,
        hash: bool | None = None,
        compare: bool = True,
        metadata: Mapping[Any, Any] | None = None,
        kw_only: bool = ...,
) -> Any: ...


def field(
        *,
        default: _T = MISSING,
        default_factory: Callable[[], _T] = MISSING,
        help: str = "",
        flags: tuple[str] = (),
        init: bool = True,
        repr: bool = True,
        hash: bool | None = None,
        compare: bool = True,
        metadata: Mapping[Any, Any] | None = None,
        kw_only: bool = False,
) -> _T:
    """
    Extends :py:`dataclasses.field` to adds `help` and `flags` to the metadata.

    Parameters
    ----------
    help : str, default=""
        A help string to be used in argparse description of parameters.
    flags : tuple of str, default=()
        A list of default flags to add for this attribute.

    Returns
    -------
    When used in dataclasses, returns .

    See Also
    --------
    :py:func:`dataclasses.field`
    """
    if isinstance(metadata, Mapping):
        metadata = dict(metadata)
    elif metadata is None:
        metadata = {}
    else:
        raise TypeError("Invalid type of metadata, must be a Mapping!")
    if help:
        if not isinstance(help, str):
            raise TypeError("help must be a str!")
        metadata["help"] = help
    if flags:
        if not isinstance(flags, tuple):
            raise TypeError("flags must be a tuple!")
        metadata["flags"] = flags

    kwargs = dict(init=init, repr=repr, hash=hash, compare=compare, kw_only=kw_only)
    if default is not MISSING:
        kwargs["default"] = default
    if default_factory is not MISSING:
        kwargs["default_factory"] = default_factory
    return _field(**kwargs, metadata=metadata)


def get_field(dc, fieldname: str) -> Field | None:
    """
    Return a specific Field object associated with a dataclass class or object.

    Parameters
    ----------
    dc : dataclass, type[dataclass]
        The dataclass containing the field.
    fieldname : str
        The name of the field.

    Returns
    -------
    Field, None
        The Field object associated with `fieldname` or None if the field does not exist.

    See Also
    --------
    :py:`dataclasses.fields`
    """
    for field in fields(dc):
        if field.name == fieldname:
            return field
    return None
