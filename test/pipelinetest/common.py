import logging
from functools import lru_cache
from pathlib import Path
from typing import cast

import nibabel.filebasedimages
import numpy as np
import pytest
import yaml

from FastSurferCNN.segstats import PVStats, dataframe_to_table, read_statsfile
from FastSurferCNN.utils.brainvolstats import MeasureTuple
from FastSurferCNN.utils.mapper import TSVLookupTable


@lru_cache
def _read_image_cached(__file: Path) -> nibabel.analyze.SpatialImage:
    return cast(nibabel.analyze.SpatialImage, nibabel.load(__file))


@lru_cache
def _read_stats_cached(__file: Path) -> tuple[dict[str, MeasureTuple], list[PVStats]]:
    annotations, dataframe = read_statsfile(__file)
    return annotations, dataframe_to_table(dataframe)


@lru_cache
def _read_surface_cached(__file: Path) -> tuple[np.ndarray, np.ndarray]:
    from nibabel.freesurfer.io import read_geometry
    coords, faces = read_geometry(str(__file))
    return np.asarray(coords), np.asarray(faces)


@lru_cache
def read_chain() -> list[dict[str, str]]:
    """The pipeline outputs in the order they are produced, see data/chain.yaml."""
    with open(Path(__file__).parent / "data/chain.yaml") as fp:
        return yaml.safe_load(fp)["stages"]


@lru_cache
def chain_position(filename: str) -> int:
    """
    Where a file sits in the pipeline, for sorting comparisons so that the first failure is the
    first divergence. Files missing from the chain sort last, keeping them out of that reading.
    """
    for position, entry in enumerate(read_chain()):
        if entry["file"] == filename:
            return position
    return len(read_chain())


def chain_stage(filename: str) -> str:
    """The pipeline stage a file belongs to, or an empty string if it is not in the chain."""
    for entry in read_chain():
        if entry["file"] == filename:
            return entry["stage"]
    return ""


def chain_order(filename: str) -> tuple[int, str]:
    """
    Sort key placing a file at its pipeline position, breaking ties by name.

    The name matters: files the chain does not list all share the last position, and without a
    second key their order would follow set iteration and change between runs.
    """
    return chain_position(filename), filename


_DIFFERING = "_pipelinetest_differing"
_FAILED = "_pipelinetest_failed"


def _store(config: "pytest.Config", attribute: str) -> set[str]:
    store = getattr(config, attribute, None)
    if store is None:
        store = set()
        setattr(config, attribute, store)
    return store


def record_difference(config: "pytest.Config", filename: str) -> None:
    """
    Note that a file is not identical to the reference, whatever its tolerance then says.

    A tolerance answers "is this close enough", which is not the same question as "where did the
    change enter". One flipped voxel passes every tolerance downstream of it and still marks the
    stage that changed, so the two are reported separately.
    """
    _store(config, _DIFFERING).add(filename)


def record_failure(config: "pytest.Config", filename: str) -> None:
    """Note that a file failed its comparison, which is the tolerance being exceeded."""
    _store(config, _FAILED).add(filename)


def recorded_files(config: "pytest.Config", *, failed: bool) -> set[str]:
    return _store(config, _FAILED if failed else _DIFFERING)


def skip_if_missing(
        ref_subject: "SubjectDefinition",
        test_subject: "SubjectDefinition",
        filename: str,
        *,
        surface: bool = False,
) -> None:
    """
    Skip a comparison when either side lacks the file, naming the side that does.

    test_file_existence owns the existence assertion, so a comparison that cannot run reports the
    gap once rather than a second time as a confusing failure inside a dice or distance check.
    Both subjects carry the same name, so they are labelled by role here.
    """
    has = SubjectDefinition.has_surface if surface else SubjectDefinition.has_image
    absent = [
        label
        for label, subject in (("the reference", ref_subject), ("the test subject", test_subject))
        if not has(subject, filename)
    ]
    if absent:
        pytest.skip(f"{filename} is absent from {' and '.join(absent)}")


logger = logging.getLogger(__name__)

class SubjectDefinition:
    name: str
    path: Path

    def __init__(self, path: Path):
        self.name = path.name
        self.path = path
        self.cache = {}

        if not self.path.exists():
            pytest.fail(f"The subject {path} does not exist!")
        if not  self.path.is_dir():
            pytest.fail(f"The subject {path} is not a directory!")

    def with_subjects_dir(self, subjects_dir: Path):
        return SubjectDefinition(subjects_dir / self.name)

    def load_image(self, filename: str) -> tuple[Path, nibabel.analyze.SpatialImage]:
        image_path = self.path / "mri" / filename
        if not image_path.exists():
            pytest.fail(f"The image {self.name}/mri/{filename} does not exist!")

        return image_path, _read_image_cached(image_path)

    def has_image(self, filename: str) -> bool:
        return (self.path / "mri" / filename).exists()

    def load_surface(self, filename: str) -> tuple[Path, np.ndarray, np.ndarray]:
        """Vertex coordinates and faces of a FreeSurfer surface under surf/."""
        surface_path = self.path / "surf" / filename
        if not surface_path.exists():
            pytest.fail(f"The surface {self.name}/surf/{filename} does not exist!")

        coords, faces = _read_surface_cached(surface_path)
        return surface_path, coords, faces

    def has_surface(self, filename: str) -> bool:
        return (self.path / "surf" / filename).exists()

    def load_stats_file(self, filename: str) -> tuple[Path, dict[str, MeasureTuple], list[PVStats]]:
        stats_path = self.path / "stats" / filename
        if not stats_path.exists():
            pytest.fail(f"The stats file {self.name}/stats/{filename} does not exist!")

        annotations, table = _read_stats_cached(stats_path)
        return stats_path, annotations, table

    def __repr__(self):
        return f"Subject<{self.name}>"

class Tolerances:

    def __init__(self, config_file: Path):
        """
        Load the thresholds from the config_file.

        Parameters
        ----------
        config_file : Path
            The file with the thresholds to consider.
        """
        logger.debug(f"Reading {config_file}...")
        self.config_file = config_file
        with open(config_file) as fp:
            self.config = yaml.safe_load(fp)

        if "lut" in self.config:
            self.lut = self.config["lut"]
            # here we want a mapper from id to labelname
            self.mapper = TSVLookupTable(Path(__file__).parents[2] / self.lut).labelname2id().__reversed__()

        else:
            self.lut = None
            self.mapper = None
            # raise ValueError("lut not found in config file")

    def threshold(self, label_or_key: int | str) -> tuple[str, float]:
        """
        Return a threshold for a label or key.

        Parameters
        ----------
        label_or_key : int | str
            If label is an int, assume this is a segmentation id, so try the lut, else get the value under thresholds.

        Returns
        -------
        str
            Name of the label that the threshold belongs to.
        float
            The relevant threshold.
        """
        if isinstance(label_or_key, str):
            return label_or_key, self.config["thresholds"][label_or_key]
        elif isinstance(label_or_key, int | np.integer):
            labelname = ""
            try:
                labelname = self.mapper[label_or_key]
                return labelname, self.config["thresholds"][labelname]
            except KeyError:
                _labelname = str(label_or_key)
                if not bool(labelname):
                    labelname = _labelname
            try:
                return labelname, self.config["thresholds"][_labelname]
            except KeyError:
                return labelname, self.config["default_threshold"]
        else:
            raise ValueError("Invalid type of label argument!")

    def __repr__(self):
        config = str(tuple(self.config.keys()))
        return f"{self.__class__.__name__}<{config[1:-1]}>"


def write_table_file(
        table_file: Path | None,
        subject_id: str,
        file: str,
        scores: dict[int, int | float] | dict[str, int | float],
) -> None:
    """
    Logs the calculated statistics (difference between test and reference) to a table file.

    Parameters
    ----------
    table_file : Path, None
        The file to write to, skip if None.
    subject_id : str
        The subject id.
    file : file
        The file associated with the comparison.
    scores : dict[int | str, int | float]
        The pairs of data associated with the comparison, e.g. index and value.
    """
    if not bool(table_file):
        # no valid file passed, skip
        return

    for id, score in scores.items():
        fmt = f'''"{{subject_id}}","{{file}}",{"{id:d}" if isinstance(id, int) else f'"{id}"'},'''
        fmt += f"{{score:{'.6f' if isinstance(score, float) else 'd'}}}\n"
        data = {"subject_id": subject_id, "file": file, "id": id, "score": score}
        if not table_file.is_file():
            with open(table_file, "w") as f:
                f.write(",".join(data.keys()) + "\n")
        with open(table_file, "a") as f:
            f.write(fmt.format(**data))
