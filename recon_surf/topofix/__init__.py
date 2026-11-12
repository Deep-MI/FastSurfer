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
Topology correction of white matter surfaces on the surface itself, without a spherical map.

The handles of the initial surface are found and removed with :mod:`lapy.topology`: each handle
comes with two short loops that cross once, and surgery along one of them either cuts a bridge or
fills a tunnel (:mod:`.classify`). Protrusions on which a spherical map folds are cut off
(:mod:`.folds`), because a later spherical registration cannot unfold them. :mod:`.segsurf`
extracts an initial surface from the segmentation instead.
"""
