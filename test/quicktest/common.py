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
