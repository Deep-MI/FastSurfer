"""The FreeSurfer stub helper links into any installation path, and fails when it cannot."""

import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "build" / "link_fs.sh"


def _link_fs(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["/bin/bash", str(SCRIPT), *args], capture_output=True, text=True)


# a space, and a -h that is part of the path rather than the help flag
@pytest.mark.parametrize("name", ["FreeSurfer installation", "fs-hpc"])
def test_link_fs_directory(tmp_path: Path, name: str) -> None:
    """Create executable stub links under the installation directory."""
    freesurfer = tmp_path / name
    bindir = freesurfer / "bin"
    bindir.mkdir(parents=True)
    assert _link_fs(str(freesurfer)).returncode == 0

    target = bindir / "not-here.sh"
    assert target.is_file()
    assert os.access(target, os.X_OK)
    links = list(bindir.glob("*"))
    assert len(links) > 1
    for link in links:
        if link == target:
            continue
        assert link.is_symlink()
        assert link.resolve() == target.resolve()
    result = subprocess.run([str(bindir / "mri_and"), "-all-info"], check=True, capture_output=True, text=True)
    assert "not included" in result.stdout


def test_link_fs_fails_without_bin(tmp_path: Path) -> None:
    """A stub that cannot be written fails the script instead of leaving dangling links."""
    result = _link_fs(str(tmp_path))
    assert result.returncode != 0
    assert "Could not create" in result.stdout
