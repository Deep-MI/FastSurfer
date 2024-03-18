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

__all__ = [
    "checkpoint",
    "common",
    "load_config",
    "logging",
    "lr_scheduler",
    "mapper",
    "meters",
    "metrics",
    "misc",
    "parser_defaults",
    "threads",
    "Plane",
    "PlaneAxial",
    "PlaneCoronal",
    "PlaneSagittal",
    "PLANES",
]

from typing import Literal, get_args

PlaneAxial = Literal["axial"]
PlaneCoronal = Literal["coronal"]
PlaneSagittal = Literal["sagittal"]
Plane = PlaneAxial | PlaneCoronal | PlaneSagittal
PLANES: tuple[PlaneAxial, PlaneCoronal, PlaneSagittal] = ("axial", "coronal", "sagittal")
