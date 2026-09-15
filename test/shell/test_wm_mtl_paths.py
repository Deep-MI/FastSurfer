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
Check the shim that decides whether mri_edit_wm_with_aseg runs its MTL path step.

The step only runs when the aseg handed to that binary is uchar, so the shim hands it an int
copy to skip it, matching FreeSurfer, and leaves aseg.presurf.mgz on disk alone because
mris_place_surface and mris_ca_label read the same file later.

FreeSurfer is not available in CI, so a fake FREESURFER_HOME stands in: stubs that record the
argv they were called with. That is enough to pin everything about the shim except the numerical
effect, which is a property of the FreeSurfer binary rather than of this code.
"""

import os
import subprocess
from pathlib import Path

import pytest

FASTSURFER_HOME = Path(__file__).parent.parent.parent
SHIM = FASTSURFER_HOME / "recon_surf" / "shims" / "mri_edit_wm_with_aseg"

# records its arguments, then writes the output file so the caller sees a success
CONVERT_STUB = """#!/bin/bash
echo "$@" > "$RECORD_DIR/convert.argv"
# mri_convert -odt int <in> <out>
out="${@: -1}"
cp "${@: -2:1}" "$out"
echo "int" > "$out.dtype"
"""

BINARY_STUB = """#!/bin/bash
echo "$@" > "$RECORD_DIR/binary.argv"
exit ${STUB_EXIT:-0}
"""


@pytest.fixture
def fake_fs(tmp_path):
    """A FREESURFER_HOME whose binaries only record how they were called."""
    fsbin = tmp_path / "fs" / "bin"
    fsbin.mkdir(parents=True)
    for name, body in (("mri_convert", CONVERT_STUB), ("mri_edit_wm_with_aseg", BINARY_STUB)):
        path = fsbin / name
        path.write_text(body)
        path.chmod(0o755)
    (tmp_path / "records").mkdir()
    for name in ("wm.seg.mgz", "brain.mgz", "aseg.presurf.mgz", "wm.mgz", "entowm.mgz",
                 "brain.finalsurfs.mgz"):
        (tmp_path / name).write_text(f"pretend {name}")
    return tmp_path


def run_shim(fake_fs, mode=None, args=None, **env):
    """Invoke the shim the way recon-all invokes the real binary."""
    if args is None:
        # the invocation FastSurfer's runs actually produce, from recon-surf.log
        args = ["-keep-in", "wm.seg.mgz", "brain.mgz", "aseg.presurf.mgz", "wm.asegedit.mgz"]
    environ = {
        **os.environ,
        "FREESURFER_HOME": str(fake_fs / "fs"),
        "RECORD_DIR": str(fake_fs / "records"),
        **env,
    }
    if mode is not None:
        environ["FASTSURFER_WM_MTL_PATHS"] = mode
    else:
        environ.pop("FASTSURFER_WM_MTL_PATHS", None)
    return subprocess.run([str(SHIM), *args], cwd=fake_fs, env=environ, capture_output=True, text=True)


def aseg_handed_over(fake_fs):
    """The aseg path the binary actually received."""
    return (fake_fs / "records" / "binary.argv").read_text().split()[3]


def test_the_shim_is_executable():
    """It is reached through PATH, so the bit matters as much as the contents."""
    assert SHIM.is_file(), f"{SHIM} is missing"
    assert os.access(SHIM, os.X_OK), f"{SHIM} is not executable"


def test_default_hands_over_a_different_file(fake_fs):
    """Default is skip, so the binary must not see aseg.presurf.mgz itself."""
    result = run_shim(fake_fs)
    assert result.returncode == 0, result.stderr
    assert aseg_handed_over(fake_fs) != "aseg.presurf.mgz"
    assert (fake_fs / "records" / "convert.argv").exists(), "the aseg was never converted"


def test_keep_hands_over_the_original(fake_fs):
    result = run_shim(fake_fs, mode="keep")
    assert result.returncode == 0, result.stderr
    assert aseg_handed_over(fake_fs) == "aseg.presurf.mgz"
    assert not (fake_fs / "records" / "convert.argv").exists(), "keep must not convert"


def test_the_aseg_on_disk_is_untouched(fake_fs):
    """
    mris_place_surface and mris_ca_label read the same file later.

    Converting it in place would change what those steps see, which is the reason this is a
    shim over one call rather than a dtype change on disk.
    """
    before = (fake_fs / "aseg.presurf.mgz").read_bytes()
    run_shim(fake_fs)
    assert (fake_fs / "aseg.presurf.mgz").read_bytes() == before


def test_the_other_arguments_are_passed_through(fake_fs):
    """Only the aseg may change; options and the other volumes must arrive untouched."""
    run_shim(fake_fs)
    argv = (fake_fs / "records" / "binary.argv").read_text().split()
    assert argv[0] == "-keep-in"
    assert argv[1] == "wm.seg.mgz"
    assert argv[2] == "brain.mgz"
    assert argv[4] == "wm.asegedit.mgz"


def test_the_temporary_copy_is_cleaned_up(fake_fs):
    run_shim(fake_fs)
    handed = Path(aseg_handed_over(fake_fs))
    assert not handed.exists(), f"{handed} was left behind"


def test_the_exit_code_is_propagated(fake_fs):
    """
    A failure here has to reach the caller.

    Swallowing it is the failure mode the surrounding work exists to remove, and this sits
    inside recon-all where a silent success is especially hard to notice.
    """
    assert run_shim(fake_fs, STUB_EXIT="3").returncode == 3


def test_a_failed_conversion_is_loud(fake_fs):
    """Falling through to the unconverted aseg would silently re-enable the step."""
    (fake_fs / "fs" / "bin" / "mri_convert").write_text("#!/bin/bash\necho boom >&2\nexit 1\n")
    (fake_fs / "fs" / "bin" / "mri_convert").chmod(0o755)
    result = run_shim(fake_fs)
    assert result.returncode != 0
    assert "boom" in result.stderr
    assert not (fake_fs / "records" / "binary.argv").exists(), "the binary ran despite the failure"


def test_a_missing_freesurfer_home_is_loud(fake_fs):
    result = run_shim(fake_fs, FREESURFER_HOME="")
    assert result.returncode != 0
    assert "FREESURFER_HOME" in result.stderr


def test_a_call_with_no_aseg_is_loud(fake_fs):
    """
    Passing through would run the MTL path step, which is what this shim exists to prevent.

    So a standard call whose aseg cannot be identified has to fail rather than continue.
    """
    result = run_shim(fake_fs, args=["-keep-in", "wm.seg.mgz", "brain.mgz",
                                     "not_an_aseg.mgz", "wm.asegedit.mgz"])
    assert result.returncode != 0
    assert "not_an_aseg.mgz" in result.stderr
    assert not (fake_fs / "records" / "binary.argv").exists()


def test_the_output_volume_is_not_mistaken_for_the_aseg(fake_fs):
    """wm.asegedit.mgz contains 'aseg', so a substring match would rewrite the output path."""
    (fake_fs / "wm.asegedit.mgz").write_text("a previous run left this behind")
    run_shim(fake_fs)
    argv = (fake_fs / "records" / "binary.argv").read_text().split()
    assert argv[4] == "wm.asegedit.mgz"


def test_the_aseg_is_found_regardless_of_position(fake_fs):
    """
    Options may be added ahead of the positionals, as -fix-ento-wm does.

    Locating the aseg by name rather than by offset keeps those calls working.
    """
    run_shim(fake_fs, args=["-keep-in", "-fix-ento-wm", "entowm.mgz", "3", "255", "255",
                            "wm.seg.mgz", "brain.mgz", "aseg.presurf.mgz", "wm.asegedit.mgz"])
    argv = (fake_fs / "records" / "binary.argv").read_text().split()
    assert argv[8] != "aseg.presurf.mgz", "the aseg was not substituted"
    assert argv[2] == "entowm.mgz", "an option value was rewritten"


def test_ambiguity_is_loud(fake_fs):
    """Two aseg-looking arguments means the rule no longer identifies one volume."""
    (fake_fs / "aseg.auto.mgz").write_text("pretend second aseg")
    result = run_shim(fake_fs, args=["-keep-in", "aseg.auto.mgz", "brain.mgz",
                                     "aseg.presurf.mgz", "wm.asegedit.mgz"])
    assert result.returncode != 0
    assert not (fake_fs / "records" / "binary.argv").exists()


@pytest.mark.parametrize("flag", ["-sa-fix-ento-wm", "-fix-scm-ha-only"])
def test_standalone_modes_pass_through_untouched(fake_fs, flag):
    """
    These return before the MTL path step and lay their arguments out differently.

    -sa-fix-ento-wm takes no aseg at all, so rewriting by position substituted an input volume;
    -fix-scm-ha-only takes one but runs a different function.
    """
    args = [flag, "entowm.mgz", "3", "255", "255", "wm.mgz", "wm.mgz"]
    result = run_shim(fake_fs, args=args)
    assert result.returncode == 0, result.stderr
    assert (fake_fs / "records" / "binary.argv").read_text().split() == args
    assert not (fake_fs / "records" / "convert.argv").exists(), "nothing may be converted here"


def test_short_invocations_pass_straight_through(fake_fs):
    """--help and usage errors are the binary's to answer."""
    result = run_shim(fake_fs, args=["--help"])
    assert result.returncode == 0
    assert (fake_fs / "records" / "binary.argv").read_text().strip() == "--help"


def test_an_unknown_mode_warns_and_passes_through(fake_fs):
    result = run_shim(fake_fs, mode="maybe")
    assert result.returncode == 0
    assert "maybe" in result.stderr
    assert aseg_handed_over(fake_fs) == "aseg.presurf.mgz"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
