# Copyright 2023 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
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

from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Executor, Future, ProcessPoolExecutor, ThreadPoolExecutor
from functools import lru_cache
from typing import Any, TypeVar

from FastSurferCNN.utils import logging

__all__ = [
    "iterate",
    "get_num_threads",
    "pipeline",
    "process_executor",
    "serial_executor",
    "set_num_threads",
    "SerialExecutor",
    "thread_executor",
]

_T = TypeVar("_T")
_Ti = TypeVar("_Ti")

LOGGER = logging.getLogger(__name__)

class ParallelConfig:
    _num_threads: int | None = None
    _process_pool: ProcessPoolExecutor | None = None
    _thread_pool: ThreadPoolExecutor | None = None

    @classmethod
    def process_executor(cls) -> ProcessPoolExecutor:
        if cls._process_pool is None:
            cls._process_pool = ProcessPoolExecutor(max_workers=cls.get_num_threads())
        return cls._process_pool

    @classmethod
    def thread_executor(cls) -> ThreadPoolExecutor:
        if cls._thread_pool is None:
            cls._thread_pool = ThreadPoolExecutor(max_workers=cls.get_num_threads() + 4)
        return cls._thread_pool

    @classmethod
    def get_num_threads(cls):
        """
        Determine the number of available threads.

        Tries to get the process's CPU affinity for usable thread count; defaults
        to total CPU count on failure.

        Returns
        -------
        int
            Number of threads available to the process or total CPU count.
        """
        if cls._num_threads is not None and cls._num_threads > 0:
            return cls._num_threads
        try:
            from os import sched_getaffinity as __getaffinity

            return len(__getaffinity(0))
        except ImportError:
            from os import cpu_count

            return cpu_count()

    @classmethod
    def set_num_threads(cls, threads: int | str, wait_finish: bool = False) -> int:
        """
        Sets the number of threads to use.

        Parameters
        ----------
        threads : int
            The number of threads to use, values < 1 reset the thread count to automatically determined.
        wait_finish : bool, default=False
            Wait for the executor to finish current processing (warning, this may have side-effects!).

        Returns
        -------
        int
            The value as int.
        """
        value = int(threads)
        # if has a pool object AND changing number of threads AND NOT changing between auto-values
        if cls._process_pool is not None and value != cls._num_threads and (value > 1 or cls._num_threads > 1):
            cls._process_pool.shutdown(wait=wait_finish)
            cls._process_pool = None
        cls._num_threads = value
        return cls._num_threads

    @classmethod
    def shutdown(cls, wait = True, *, cancel_futures = False):
        """
        This is a cleanup function and should only be called at the end of a script!

        This first shuts down the `process_executor` and then the `thread_executor`.

        Notes
        -----
        The executors are not shut down at the same time, but rather, if wait is True, we first wait for the
        `process_executor` to "finish" shutting down.
        """
        if cls._process_pool is not None:
            cls._process_pool.shutdown(wait=wait, cancel_futures=cancel_futures)
            cls._process_pool = None
        if cls._thread_pool is not None:
            cls._thread_pool.shutdown(wait=wait, cancel_futures=cancel_futures)
            cls._thread_pool = None


get_num_threads = ParallelConfig.get_num_threads
set_num_threads = ParallelConfig.set_num_threads


class SerialExecutor(Executor):
    """
    Represent a serial executor.
    """

    def map(
        self,
        fn: Callable[..., _T],
        *iterables: Iterable[Any],
        timeout: float | None = None,
        chunksize: int = -1,
    ) -> Iterator[_T]:
        """
        The map function.

        Parameters
        ----------
        fn : Callable[..., _T]
            A callable function to be applied to the items in the iterables.
        *iterables : Iterable[Any]
            One or more iterable objects.
        timeout : Optional[float]
            Maximum number of seconds to wait for a result. Default is None.
        chunksize : int
            The size of the chunks, default value is -1.

        Returns
        -------
        Iterator[_T]
            An iterator that yields the results of applying 'fn' to the items of
            'iterables'.
        """
        return map(fn, *iterables)

    def submit(self, __fn: Callable[..., _T], *args, **kwargs) -> "Future[_T]":
        """
        A callable function that returns a Future representing the result.

        Parameters
        ----------
        __fn : Callable[..., _T]
            A callable function to be executed.
        *args :
            Potential arguments to be passed to the callable function.
        **kwargs :
            Keyword arguments to be passed to the callable function.

        Returns
        -------
        "Future[_T]"
            A Future object representing the execution result of the callable function.
        """
        f = Future()
        try:
            f.set_result(__fn(*args, **kwargs))
        except Exception as e:
            f.set_exception(e)
        return f

    def shutdown(self, wait = True, *, cancel_futures = False):
        return


@lru_cache(maxsize=1)
def serial_executor() -> SerialExecutor:
    return SerialExecutor()

thread_executor = ParallelConfig.thread_executor
process_executor = ParallelConfig.process_executor


def pipeline(
    pool: Executor,
    func: Callable[[_Ti], _T],
    iterable: Iterable[_Ti],
    *,
    pipeline_size: int = 1,
) -> Iterator[tuple[_Ti, _T]]:
    """
    Pipeline a function to be executed in the pool.

    Analogous to iterate, but run func in a different
    thread for the next element while the current element is returned.

    Parameters
    ----------
    pool : Executor
        Thread pool executor for parallel execution.
    func : callable
        Function to use.
    iterable : Iterable
        Iterable containing input elements.
    pipeline_size : int, default=1
        Size of the processing pipeline.

    Yields
    ------
    element : _Ti
        Elements
    _T
        Results of func corresponding to element: func(element).
    """
    # do pipeline loading the next element
    from collections import deque

    futures_queue = deque()
    import itertools

    for i, element in zip(itertools.count(-pipeline_size), iterable):
        # pre-load next element/data
        futures_queue.append((element, pool.submit(func, element)))
        if i >= 0:
            element, future = futures_queue.popleft()
            yield element, future.result()
    while len(futures_queue) > 0:
        element, future = futures_queue.popleft()
        yield element, future.result()


def iterate(
    pool: Executor, func: Callable[[_Ti], _T], iterable: Iterable[_Ti],
) -> Iterator[tuple[_Ti, _T]]:
    """
    Iterate over iterable, yield pairs of elements and func(element).

    Parameters
    ----------
    pool : Executor
        The Executor object (dummy object to have a common API with pipeline).
    func : callable
        Function to use.
    iterable : Iterable
        Iterable to draw objects to process with func from.

    Yields
    ------
    element : _Ti
        Elements
    _T
        Results of func corresponding to element: func(element).
    """
    for element in iterable:
        yield element, func(element)

shutdown_executors = ParallelConfig.shutdown



