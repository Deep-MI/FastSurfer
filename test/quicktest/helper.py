import logging
from collections.abc import Iterable

import numpy as np
import pytest

logger = logging.getLogger(__name__)


def equal_within_tolerance(reference, test, rtol: float, atol: float) -> bool:
    """Whether two header values are equal as floats, so only their representation differs.

    Anything that is not a float field of the same shape, such as dims or the data type, is left to
    the exact comparison.
    """
    reference, test = np.asanyarray(reference), np.asanyarray(test)
    if reference.dtype.kind != "f" or test.dtype.kind != "f" or reference.shape != test.shape:
        return False
    return bool(np.allclose(reference, test, rtol=rtol, atol=atol, equal_nan=True))


def assert_same_headers(test_header, reference_header, rtol: float = 1e-6, atol: float = 1e-6):
    __tracebackhide__ = True

    from nibabel.cmdline.diff import get_headers_diff

    # the order given here is the order of the two values reported per field, so it is named in the
    # failure message: working it out from the code costs more than saying it
    header_diff = get_headers_diff([reference_header, test_header])
    # get_headers_diff compares exactly. Below 1mm the direction cosines are computed rather than
    # snapped, so every Mdc carries float dust that no rerun reproduces bit for bit.
    header_diff = {
        field: values
        for field, values in header_diff.items()
        if not equal_within_tolerance(*values, rtol=rtol, atol=atol)
    }
    if header_diff:
        differences = "\n".join(f"  '{k}': {v}" for k, v in header_diff.items())
        pytest.fail(
            f"The headers differ in the following fields, as [reference, test], with floats "
            f"compared at rtol={rtol}, atol={atol}:\n{differences}"
        )


class Approx:

    # Tell numpy to use our `__eq__` operator instead of its.
    # The reason is that ndarray.__eq__ will iterate over the array and create recursions
    # ndarray_1 == Approx(ndarray_2, ...) will call Approx.__eq__(ndarray_1.flat[i]) for i in ndindex...
    __array_ufunc__ = None
    __array_priority__ = 100


    def __init__(self, expected, *args, __indent = "", **kwargs):

        self.expected = expected
        self.args = args
        self.kwargs = kwargs
        self._indent = __indent

    def __eq__(self, actual):
        indent = self._indent
        if np.equal(self.expected, actual).all():
            logger.debug(f"{indent}'{str(self.expected):20s}' and '{str(actual):20s}' are identical!")
            return True
        if isinstance(actual, np.ndarray) or isinstance(self.expected, np.ndarray):
            within_bounds = np.allclose(actual, self.expected, **self._numpy_kwarg_tol())
        else:
            within_bounds = actual == pytest.approx(self.expected, *self.args, **self.kwargs)
        if within_bounds and logger.isEnabledFor(logging.DEBUG):
            def relative(actual: float, expected: float) -> float:
                return abs(expected - actual) / max((abs(expected), abs(actual), 1e-8))

            def fmt(value) -> str:
                return f"'{strval}'" if len(strval := str(value)) <= 20 else f"'{strval[:17]}...'"

            try:
                import math
                delta = math.nan
                rel = math.nan
                type_recognized = True
                if isinstance(self.expected, float | int):
                    delta = self.expected - actual
                    rel = relative(actual, self.expected)
                elif isinstance(self.expected, dict):
                    delta = max(abs(exp_value - actual[exp_key]) for exp_key, exp_value in self.expected.items())
                    rel = max(relative(actual[exp_key], exp_value) for exp_key, exp_value in self.expected.items())
                elif isinstance(self.expected, str):
                    logger.debug(f"{indent}Comparing strings {fmt(self.expected)} and {fmt(actual)}")
                    type_recognized = False
                elif isinstance(actual, np.ndarray) or isinstance(self.expected, np.ndarray):
                    logger.debug("\n".join(self._compare_numpy(actual)))
                elif isinstance(self.expected, Iterable):
                    # len is already same
                    for i, (a, b) in enumerate(zip(self.expected, actual, strict=True)):
                        within_bounds &= b == Approx(a, *self.args, **self.kwargs, __indent=f"{i}:{indent}")
                    type_recognized = False
                else:
                    expected_typ = type(self.expected).__name__
                    actual_typ = type(actual).__name__
                    logger.debug(f"{indent}Unrecognized data type: {expected_typ} and/or {actual_typ}!")
                    type_recognized = False
                if type_recognized:
                    logger.debug(f"{indent}{fmt(self.expected)} and {fmt(actual)} are within abs={delta}, rel={rel}!")
            except BaseException as e:
                logger.exception(e)
        return within_bounds

    def _numpy_kwarg_tol(self):
        return {"rtol": self.kwargs.get("rel", 0.), "atol": self.kwargs.get("abs", 0.)}

    def __ne__(self, actual):
        return not (self == actual)

    def __repr__(self):
        return repr(self.expected)

    def _repr_compare(self, other_side) -> list[str]:
        if isinstance(other_side, np.ndarray) or isinstance(self.expected, np.ndarray):
            return self._compare_numpy(other_side)
        return pytest.approx(self.expected, *self.args, **self.kwargs)._repr_compare(other_side)

    def _compare_numpy(self, actual: np.ndarray) -> list[str]:
        expected = np.asarray(self.expected)
        actual = np.asarray(actual)
        msg = [
            f"Comparison returned differences >{self.kwargs}",
            f"shape: {actual.shape} vs. {expected.shape}",
        ]
        try:
            delta = expected - actual
            msg.extend((f"mean delta: {np.abs(delta).mean()}", f"max delta: {np.abs(delta).mean()}"))
            delta_where = np.isclose(expected, actual, **self._numpy_kwarg_tol())
            if not delta_where.all():
                from skimage.measure import regionprops

                msg.append(f"voxels exceeding threshold: {np.sum(np.logical_not(delta_where))}")
                msg.append("Connected components exceeding threshold:")
                connected_components = regionprops(delta_where)
                msg.extend(f"  {i}: {cc.area} voxels @{cc.centroid}" for i, cc in enumerate(connected_components[:5]))
        except BaseException as e:
            logger.exception(e)
            msg.append(f"Exception while comparing arrays: {e}")
        return msg

