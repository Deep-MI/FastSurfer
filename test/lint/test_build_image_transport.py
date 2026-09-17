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
Keep the built image travelling to the test jobs as one copy rather than three.

The build job hands the image to the test jobs as an artifact and never runs it itself, so buildkit
writes the tarball directly. Loading it into the job's docker and calling `docker save` on it again
produces the same bytes after two more passes over several GB. That costs only time, so it went
unnoticed for a long time and would again.

Everything here is read as text, so this runs under ``uv run --no-project --with pytest`` like the
rest of test/lint.
"""

import re
from pathlib import Path

FASTSURFER_HOME = Path(__file__).parent.parent.parent
BUILD_ACTION = FASTSURFER_HOME / ".github" / "actions" / "build-docker" / "action.yml"
BUILD_PY = FASTSURFER_HOME / "tools" / "Docker" / "build.py"
TARBALL = "/tmp/docker-image.tar"


def test_the_build_action_exports_the_tarball_itself():
    """Without this the image is serialised by buildkit and then again by docker save."""
    text = BUILD_ACTION.read_text()
    assert f"--save_image {TARBALL}" in text, (
        "the build action no longer asks build.py to write the artifact tarball, so the image "
        "goes through the docker daemon and is copied twice more"
    )


def test_the_build_action_does_not_save_the_image_again():
    """`docker save` here would mean the image had to be loaded first, undoing the point."""
    saves = re.findall(r"^\s*docker save\b.*$", BUILD_ACTION.read_text(), re.MULTILINE)
    assert saves == [], f"the build action serialises the image a second time: {saves}"


def test_an_exported_image_can_be_loaded_back():
    """
    The docker format, because load-docker reads the tarball with `docker load`.

    buildkit's oci layout holds the attestation manifests and `docker load` cannot read it, so
    picking it for the plain export would break the test jobs rather than the build.
    """
    text = BUILD_PY.read_text()
    export_branch = text[text.index('elif action == "export":') : text.index("elif attestation:")]
    assert 'image_type = f"docker{dest}"' in export_branch, (
        "an image exported without attestation has to be in the docker format, otherwise "
        "load-docker cannot read it back"
    )


def test_exporting_does_not_also_ask_docker_to_load():
    """`--load` alongside a destination file brings the round trip back."""
    text = BUILD_PY.read_text()
    assert 'if action in ("load", "push")' in text, (
        "build.py no longer restricts the --load/--push flag to those actions, so an export "
        "either emits an invalid --export flag or loads the image as well"
    )
