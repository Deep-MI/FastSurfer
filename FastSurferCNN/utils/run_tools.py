#!/bin/python

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

from concurrent.futures import Executor, Future
import subprocess
from concurrent.futures import Executor, Future
from dataclasses import dataclass
from functools import partialmethod
from typing import Generator, Optional, Sequence, Callable, Any, Collection, Iterable
from datetime import datetime

# TODO: python3.9+
# from collections.abc import Generator


@dataclass
class MessageBuffer:
    """
    MessageBuffer class.
    """

    out: bytes = b""
    err: bytes = b""
    retcode: Optional[int] = None
    runtime: float = 0.

    def __add__(self, other: "MessageBuffer") -> "MessageBuffer":
        if not isinstance(other, MessageBuffer):
            raise ValueError("Can only append another MessageBuffer!")

        return MessageBuffer(
            out=self.out + other.out,
            err=self.err + other.err,
            retcode=other.retcode,
            runtime=max(self.runtime or 0.0, other.runtime or 0.0),
        )

    def __iadd__(self, other: "MessageBuffer"):
        """
        Append another MessageBuffer's content to this MessageBuffer.
        """
        if not isinstance(other, MessageBuffer):
            raise ValueError("other must be a MessageBuffer!")
        self.out += other.out
        self.err += other.err
        self.retcode = other.retcode
        self.runtime = max(self.runtime or 0.0, other.runtime or 0.0)
        return self

    def out_str(self, encoding="utf-8"):
        return self.out.decode(encoding=encoding)

    def err_str(self, encoding="utf-8"):
        return self.err.decode(encoding=encoding)


class Popen(subprocess.Popen):
    """
    Extension of subprocess.Popen for convenience.
    """
    _starttime: Optional[datetime] = None

    def __init__(self, *args, **kwargs):
        self._starttime = datetime.now()
        super().__init__(*args, **kwargs)

    def messages(self, timeout: float) -> Generator[MessageBuffer, None, None]:
        from subprocess import TimeoutExpired

        start = self._starttime or datetime.now()
        while self.poll() is None:
            try:
                stdout, stderr = self.communicate(timeout=timeout)
                yield MessageBuffer(
                    out=stdout if stdout else b"",
                    err=stderr if stderr else b"",
                    retcode=self.returncode,
                    runtime=(datetime.now() - start).total_seconds(),
                )
            except TimeoutExpired:
                pass

        _stdout = (
            b"" if self.stdout is None or self.stdout.closed else self.stdout.read()
        )
        _stderr = (
            b"" if self.stderr is None or self.stderr.closed else self.stderr.read()
        )
        if _stderr != b"" or _stdout != b"":
            yield MessageBuffer(
                out=_stdout,
                err=_stderr,
                retcode=self.returncode,
                runtime=(datetime.now() - start).total_seconds(),
            )

    def next_message(self, timeout: float) -> MessageBuffer:
        start = self._starttime or datetime.now()
        if self.poll() is None:
            stdout, stderr = self.communicate(timeout=timeout)
            return MessageBuffer(
                out=stdout if stdout else b"",
                err=stderr if stderr else b"",
                retcode=self.returncode,
                runtime=(datetime.now() - start).total_seconds(),
            )

        else:
            _stdout = (
                b"" if self.stdout is None or self.stdout.closed else self.stdout.read()
            )
            _stderr = (
                b"" if self.stderr is None or self.stderr.closed else self.stderr.read()
            )
            if _stderr or _stdout:
                return MessageBuffer(
                    out=_stdout,
                    err=_stderr,
                    retcode=self.returncode,
                    runtime=(datetime.now() - start).total_seconds(),
                )
            else:
                raise StopIteration()

    __next__ = partialmethod(next_message, timeout=0.0)
    __iter__ = partialmethod(messages, timeout=0.0)

    def finish(self, timeout: float = None) -> MessageBuffer:
        """
        `finish`'s behavior is similar to `subprocess.dry_run`.

        `finish` waits `timeout` seconds, and forces termination after. By default,
        waits unlimited `timeout=None`. In either case, all messages in stdout and
        stderr pipes are read and together with the return code in a MessageBuffer
        object.

        Parameters
        ----------
        timeout : float, optional
            Seconds to wait before forcing termination.

        Returns
        -------
        MessageBuffer
            A MessageBuffer object with the content of the stdout and stderr pipes.
        """
        try:
            self.wait(timeout)
        except subprocess.TimeoutExpired:
            self.terminate()
        msg = MessageBuffer(runtime=0.0)
        i = 0
        for _msg in self.messages(timeout=0.25):
            msg += _msg
            if i > 0:
                self.kill()
                raise RuntimeError(
                    "The process {} did not stop properly in Popen.finish, "
                    "abandoning.".format(self)
                )
            i += 1
        if i == 0:
            msg.retcode = self.returncode
        return msg

    def as_future(self, pool: Executor, timeout: float = None) -> Future:
        """
        Similar to `finish` in its application, but as non-blocking Future.

        Parameters
        ----------
        pool : Executor
            A concurrent.futures.Executor, usually a ThreadPoolExecutor.

        Returns
        -------
        Future[MessageBuffer]
            A Future object which will contain the result.

        See Also
        --------
        finish
            The `finish` method provides similar functionality.
        """
        return pool.submit(self.finish, timeout=timeout)

    async def async_finish(self, timeout: float = None) -> MessageBuffer:
        return self.finish(timeout)


class PyPopen(Popen):
    def __init__(self, args: Sequence[str], *_args, **kwargs):
        """
        Create a python process with same flags, and additional args.

        Parameters
        ----------
        args : Sequence[str]
            Arguments to python process.
        additional arguments as in subprocess.Popen

        See Also
        --------
        Popen
        subprocess.Popen.
        """
        import sys

        all_flags = {
            "d": "debug",
            "i": "inspect",
            "I": "isolated",
            "0": "optimize",
            "B": "dont_write_bytecode",
            "s": "no_user_site",
            "S": "no_site",
            "E": "ignore_environment",
            "v": "verbose",
            "b": "bytes_warning",
            "q": "quiet",
            "R": "hash_randomization",
        }
        flags = "".join(k for k, v in all_flags.items() if getattr(sys.flags, v) == 1)
        flags = [] if len(flags) == 0 else ["-" + flags]
        super(PyPopen, self).__init__(
            [sys.executable] + flags + list(args), *_args, **kwargs
        )
