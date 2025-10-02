# Copyright 2025 AI in Medical Imaging, German Center for Neurodegenerative Diseases(DZNE), Bonn
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

import os
import sys


class HiddenPrints:
    """Context manager for suppressing stdout output.

    Temporarily redirects stdout to os.devnull to hide any print statements
    within the context.

    Examples
    --------
    >>> with HiddenPrints():
    ...     print("This will not be visible")
    >>> print("This will be visible")
    """

    def __enter__(self) -> None:
        """Enter the context manager.

        Returns
        -------
        None
        """
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type: type | None, exc_val: Exception | None, 
                 exc_tb: type | None) -> None:
        """Exit the context manager.

        Parameters
        ----------
        exc_type : type or None
            Type of the exception that occurred, if any
        exc_val : Exception or None
            Exception instance that occurred, if any
        exc_tb : type or None
            Traceback of the exception that occurred, if any

        Returns
        -------
        None
        """
        sys.stdout.close()
        sys.stdout = self._original_stdout