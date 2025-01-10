import logging
import math
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import cast

import nibabel.filebasedimages
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

        assert self.path.exists(), f"The subject {path} does not exist!"
        assert self.path.is_dir(), f"The subject {path} is not a directory!"

    def with_subjects_dir(self, subjects_dir: Path):
        return SubjectDefinition(subjects_dir / self.name)

    def load_image(self, filename: str) -> tuple[Path, nibabel.analyze.SpatialImage]:
        image_path = self.path / "mri" / filename
        assert image_path.exists(), f"The image {image_path} does not exist!"

        return image_path, _read_image_cached(image_path)

    def load_stats_file(self, filename: str) -> tuple[Path, dict[str, MeasureTuple], list[PVStats]]:
        stats_path = self.path / "stats" / filename
        assert stats_path.exists(), f"The stats file {stats_path} does not exist!"

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

        if "lut" not in self.config:
            raise ValueError("lut not found in config file")

        self.lut = self.config["lut"]
        # here we want a mapper from id to labelname
        self.mapper = TSVLookupTable(Path(__file__).parents[2] / self.lut).labelname2id().__reversed__()

    def threshold(self, label: int) -> tuple[str, float]:
        labelname = ""
        try:
            labelname = self.mapper[label]
            return labelname, self.config["thresholds"][labelname]
        except KeyError:
            _labelname = str(label)
            if not bool(labelname):
                labelname = _labelname
        try:
            return labelname, self.config["thresholds"][_labelname]
        except KeyError:
            return labelname, self.config["default_threshold"]


    def __repr__(self):
        config = str(tuple(self.config.keys()))
        return f"{self.__class__.__name__}<{config[1:-1]}>"

class ApproxAndLog:

    def __init__(self, value, *args, **kwargs):
        self.value = value
        self.approx_obj = pytest.approx(value, *args, **kwargs)

    def __eq__(self, other):
        if self.value == other:
            logger.debug(f"'{str(self.value):20s}' and '{str(other):20s}' are identical!")
            return True
        within_bounds = self.approx_obj == other
        if within_bounds and logger.isEnabledFor(logging.DEBUG):
            try:
                type_recognized = True
                if isinstance(self.value, float | int):
                    delta = self.value - other
                    rel = abs(delta)/max((abs(self.value), abs(other)))
                elif isinstance(self.value, dict):
                    delta = max(abs(v - other[k]) for k, v in self.value.items())
                    rel = max(abs(v - other[k])/max((abs(v), abs(other[k]), 1e-8)) for k, v in self.value.items())
                elif isinstance(self.value, Iterable):
                    delta = max(abs(a - b) for a, b in zip(self.value, other, strict=True))
                    rel = max(abs(a - b)/max((abs(a), abs(b), 1e-8)) for a, b in zip(self.value, other, strict=True))
                else:
                    logger.debug(f"Unrecognized data type: {type(self.value).__name__} and/or {type(other).__name__}!")
                    delta = math.nan
                    rel = math.nan
                    type_recognized = False
                if type_recognized:
                    logger.debug(f"'{str(self.value):20s}' and '{str(other):20s}' are within abs={delta}, rel={rel}!")
            except BaseException as e:
                logger.exception(e)
        return within_bounds
