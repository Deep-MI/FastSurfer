# Copyright 2026 DeepMI Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
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
Startup hook that runs in every python interpreter of an environment it is installed into.

``getpass.getuser()`` (used in nibabel's ``write_geometry``, monai, ...) fails if none of the
environment variables ``LOGNAME``, ``USER``, ``LNAME`` and ``USERNAME`` names the user and the
uid has no passwd entry, e.g., ``docker run -u``.

How it runs
-----------
This module and the two files of the same name next to it are installed into site-packages, where
the site module processes them at every interpreter start:

- ``fastsurfer_startup.pth``: its ``import`` line runs on python up to 3.14. Python 3.12 runs it
  twice in a virtual environment (``site.venv()`` and ``site.main()`` both process the venv's
  site-packages), so :func:`main` must stay idempotent.
- ``fastsurfer_startup.start``: the entry point that replaces ``import`` lines from python 3.15
  (PEP 829). Its presence also disables the ``import`` line of the ``.pth`` file with the same
  name, so the hook runs once per start there.

Only ``python -S`` skips both; ``-s``, ``-E`` and ``-I`` do not.

The ``build_venv`` stage of ``tools/Docker/Dockerfile`` copies the three files into ``/venv``; a
wheel (PyPI) would have to ship them instead, see ``pyproject.toml``.

Keep this module silent (``pin_cpu_dispatch.py`` reads the stdout of python processes), cheap
(it runs on every interpreter start) and free of exceptions (a hook that raises prints a
traceback on every interpreter start).
"""

import os

# the variables getpass.getuser() reads, in its order
_NAME_VARIABLES = ("LOGNAME", "USER", "LNAME", "USERNAME")


def main() -> None:
    """
    Set ``USERNAME=UNKNOWN`` if ``getpass.getuser()`` cannot name the user, otherwise do nothing.

    This repeats the lookup of ``getpass.getuser()`` instead of calling it, because importing
    getpass (with contextlib and warnings) costs about 30 ms, importing pwd well under 1 ms, and
    this runs on every interpreter start.
    """
    if any(os.environ.get(name) for name in _NAME_VARIABLES):
        return
    try:
        # no pwd module means no POSIX system, where getuser() only reads the variables
        import pwd

        pwd.getpwuid(os.getuid())
    except (ImportError, KeyError):
        os.environ["USERNAME"] = "UNKNOWN"
