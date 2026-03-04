#!/bin/python

# Copyright 2026 Image Analysis Lab, German Center for Neurodegenerative Diseases(DZNE), Bonn
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

from contextlib import contextmanager


@contextmanager
def backend(backend: str):
    """
    Context manager to temporarily set the matplotlib backend.

    Parameters
    ----------
    backend : str
        The name of the matplotlib backend to use within the context.

    Yields
    ------
    None
        This function does not yield any value, it only sets the backend temporarily.
    """
    import matplotlib

    original_backend = matplotlib.get_backend()
    try:
        matplotlib.use(backend, force=True)
        yield
    finally:
        matplotlib.use(original_backend, force=True)
