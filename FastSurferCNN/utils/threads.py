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


def get_num_threads():
    """
    Determine the number of available threads.

    Tries to get the process's CPU affinity for usable thread count; defaults
    to total CPU count on failure.

    Returns
    -------
    int
        Number of threads available to the process or total CPU count.
    """
    try:
        from os import sched_getaffinity as __getaffinity

        return len(__getaffinity(0))
    except ImportError:
        from os import cpu_count

        return cpu_count()
