from pathlib import Path


def create_expand_output_directory(
        subject_dir: Path,
        qc_snapshots: bool = False,
) -> None:
    """
    Create the output directories for HypVINN.

    Parameters
    ----------
    subject_dir : Path
        The path to the subject directory.
    qc_snapshots : bool, default=False
        Whether the qc_snapshots directory should be created.
    """
    paths = [
        subject_dir,
        subject_dir / "mri" / "transforms",
        subject_dir / "stats",
    ]
    if qc_snapshots:
        paths.append(subject_dir / "qc_snapshots")

    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
