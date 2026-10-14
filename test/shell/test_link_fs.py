"""The FreeSurfer stub helper accepts installation paths containing spaces."""

import os
import subprocess
from pathlib import Path


def test_link_fs_spaced_directory(tmp_path: Path) -> None:
    """Create executable stub links under a spaced installation directory."""
    freesurfer = tmp_path / "FreeSurfer installation"
    bindir = freesurfer / "bin"
    bindir.mkdir(parents=True)
    script = Path(__file__).resolve().parents[2] / "tools" / "build" / "link_fs.sh"
    subprocess.run(["/bin/bash", str(script), str(freesurfer)], check=True, capture_output=True, text=True)

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
