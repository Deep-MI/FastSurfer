# Copyright 2019 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
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
This is the FastSurfer/run_prediction.py script, the backbone for whole brain
segmentation.

Usage:

See Also
--------
:ref:`/scripts/fastsurfercnn.rst`
`run_prediction.py --help`
"""


# IMPORTS
import argparse
import copy
import sys
from concurrent.futures import Executor, ThreadPoolExecutor, Future
from pathlib import Path
from typing import Any, Iterator, Literal, Optional, Sequence

import nibabel as nib
import numpy as np
import torch
import yacs.config

import FastSurferCNN.reduce_to_aseg as rta
from FastSurferCNN.data_loader import conform as conf
from FastSurferCNN.data_loader import data_utils as du
from FastSurferCNN.inference import Inference
from FastSurferCNN.utils import logging, parser_defaults, Plane, PLANES
from FastSurferCNN.utils.arg_types import VoxSizeOption
from FastSurferCNN.utils.checkpoint import (
    get_checkpoints,
    load_checkpoint_config_defaults,
)
from FastSurferCNN.utils.load_config import load_config
from FastSurferCNN.utils.common import (
    SerialExecutor,
    find_device,
    assert_no_root,
    handle_cuda_memory_exception,
    SubjectList,
    SubjectDirectory,
    pipeline,
)
from FastSurferCNN.utils.parser_defaults import SubjectDirectoryConfig
from FastSurferCNN.quick_qc import check_volume

##
# Global Variables
##
from FastSurferCNN.utils.parser_defaults import FASTSURFER_ROOT
LOGGER = logging.getLogger(__name__)
CHECKPOINT_PATHS_FILE = FASTSURFER_ROOT / "FastSurferCNN/config/checkpoint_paths.yaml"


##
# Processing
##
def set_up_cfgs(
        cfg_file: str | Path,
        batch_size: int = 1,
) -> yacs.config.CfgNode:
    """
    Set up configuration.

    Sets up configurations with given arguments inside the yaml file.

    Parameters
    ----------
    cfg_file : Path, str
        Path to yaml file of configurations.
    batch_size : int, default=1
        The batch size to use.

    Returns
    -------
    yacs.config.CfgNode
        Node of configurations.
    """
    cfg = load_config(str(cfg_file))
    cfg.OUT_LOG_NAME = "fastsurfer"
    cfg.TEST.BATCH_SIZE = batch_size

    cfg.MODEL.OUT_TENSOR_WIDTH = cfg.DATA.PADDED_SIZE
    cfg.MODEL.OUT_TENSOR_HEIGHT = cfg.DATA.PADDED_SIZE
    return cfg


def args2cfg(
    cfg_ax: Optional[str] = None,
    cfg_cor: Optional[str] = None,
    cfg_sag: Optional[str] = None,
    batch_size: int = 1,
) -> tuple[
    yacs.config.CfgNode, yacs.config.CfgNode, yacs.config.CfgNode, yacs.config.CfgNode
]:
    """
    Extract the configuration objects from the arguments.

    Parameters
    ----------
    cfg_ax : str, optional
        The path to the axial network YAML config file.
    cfg_cor : str, optional
        The path to the coronal network YAML config file.
    cfg_sag : str, optional
        The path to the sagittal network YAML config file.
    batch_size : int, default=1
        The batch size for the network.

    Returns
    -------
     yacs.config.CfgNode
        Configurations for all planes.
    """
    if cfg_cor is not None:
        cfg_cor = set_up_cfgs(cfg_cor, batch_size)
    if cfg_sag is not None:
        cfg_sag = set_up_cfgs(cfg_sag, batch_size)
    if cfg_ax is not None:
        cfg_ax = set_up_cfgs(cfg_ax, batch_size)
    cfgs = (cfg_cor, cfg_sag, cfg_ax)
    # returns the first non-None cfg
    try:
        cfg_fin = next(filter(None, cfgs))
    except StopIteration:
        raise RuntimeError("No valid configuration passed!")
    return (cfg_fin,) + cfgs


##
# Input array preparation
##


class RunModelOnData:
    """
    Run the model prediction on given data.

    Attributes
    ----------
    vox_size : float, 'min'
    current_plane : str
    models : Dict[str, Inference]
    view_ops : Dict[str, Dict[str, Any]]
    conform_to_1mm_threshold : float, optional
        threshold until which the image will be conformed to 1mm res

    Methods
    -------
    __init__()
        Construct object.
    set_and_create_outdir()
        Sets and creates output directory.
    conform_and_save_orig()
        Saves original image.
    set_subject()
        Setter.
    get_subject_name()
        Getter.
    set_model()
        Setter.
    run_model()
        Calculates prediction.
    get_img()
        Getter.
    save_img()
        Saves image as file.
    set_up_model_params()
        Setter.
    get_num_classes()
        Getter.
    """

    vox_size: float | Literal["min"]
    current_plane: Plane
    models: dict[Plane, Inference]
    view_ops: dict[Plane, dict[str, Any]]
    conform_to_1mm_threshold: Optional[float]
    device: torch.device
    viewagg_device: torch.device
    _pool: Executor

    def __init__(
            self,
            lut: Path,
            ckpt_ax: Optional[Path] = None,
            ckpt_sag: Optional[Path] = None,
            ckpt_cor: Optional[Path] = None,
            cfg_ax: Optional[Path] = None,
            cfg_sag: Optional[Path] = None,
            cfg_cor: Optional[Path] = None,
            device: str = "auto",
            viewagg_device: str = "auto",
            threads: int = 1,
            batch_size: int = 1,
            vox_size: VoxSizeOption = "min",
            async_io: bool = False,
            conform_to_1mm_threshold: float = 0.95,
    ):
        """
        Construct RunModelOnData object.

        Parameters
        ----------
        viewagg_device : str, default="auto"
            Device to run viewagg on. Can be auto, cuda or cpu.
        """
        # TODO Fix docstring of RunModelOnData.__init__
        self._threads = threads
        torch.set_num_threads(self._threads)
        self._async_io = async_io

        self.sf = 1.0

        self.device = find_device(device)

        if self.device.type == "cpu" and viewagg_device in ("auto", "cpu"):
            self.viewagg_device = self.device
        else:
            # check, if GPU is big enough to run view agg on it
            # (this currently takes the memory of the passed device)
            self.viewagg_device = find_device(
                viewagg_device,
                flag_name="viewagg_device",
                min_memory=4 * (2**30),
            )

        LOGGER.info(f"Running view aggregation on {self.viewagg_device}")

        try:
            self.lut = du.read_classes_from_lut(lut)
        except FileNotFoundError:
            raise ValueError(
                f"Could not find the ColorLUT in {lut}, please make sure the "
                f"--lut argument is valid."
            )
        self.labels = self.lut["ID"].values
        self.torch_labels = torch.from_numpy(self.lut["ID"].values)
        self.names = ["SubjectName", "Average", "Subcortical", "Cortical"]
        self.cfg_fin, cfg_cor, cfg_sag, cfg_ax = args2cfg(
            cfg_ax, cfg_cor, cfg_sag, batch_size=batch_size,
        )
        # the order in this dictionary dictates the order in the view aggregation
        self.view_ops = {
            "coronal": {"cfg": cfg_cor, "ckpt": ckpt_cor},
            "sagittal": {"cfg": cfg_sag, "ckpt": ckpt_sag},
            "axial": {"cfg": cfg_ax, "ckpt": ckpt_ax},
        }
        self.num_classes = max(
            view["cfg"].MODEL.NUM_CLASSES for view in self.view_ops.values()
        )
        self.models = {}
        for plane, view in self.view_ops.items():
            if all(view[key] is not None for key in ("cfg", "ckpt")):
                self.models[plane] = Inference(
                    view["cfg"], ckpt=view["ckpt"], device=self.device, lut=self.lut,
                )

        if vox_size == "min":
            self.vox_size = "min"
        elif 0.0 < float(vox_size) <= 1.0:
            self.vox_size = float(vox_size)
        else:
            raise ValueError(
                f"Invalid value for vox_size, must be between 0 and 1 or 'min', was "
                f"{vox_size}."
            )
        self.conform_to_1mm_threshold = conform_to_1mm_threshold

    @property
    def pool(self) -> Executor:
        """
        Return, and maybe create the objects executor object (with the number of threads
        specified in __init__).
        """
        if not hasattr(self, "_pool"):
            if not self._async_io:
                self._pool = SerialExecutor()
            else:
                self._pool = ThreadPoolExecutor(self._threads)
        return self._pool

    def __del__(self):
        """Class destructor."""
        if hasattr(self, "_pool"):
            # only wait on futures, if we specifically ask (see end of the script, so we
            # do not wait if we encounter a fail case)
            self._pool.shutdown(True)

    def conform_and_save_orig(
        self, subject: SubjectDirectory,
    ) -> tuple[nib.analyze.SpatialImage, np.ndarray]:
        """
        Conform and saves original image.

        Parameters
        ----------
        subject : SubjectDirectory
            Subject directory object.

        Returns
        -------
        tuple[nib.analyze.SpatialImage, np.ndarray]
            Conformed image.
        """
        orig, orig_data = du.load_image(subject.orig_name, "orig image")
        LOGGER.info(f"Successfully loaded image from {subject.orig_name}.")

        # Save input image to standard location, but only
        if subject.can_resolve_attribute("copy_orig_name"):
            self.pool.submit(self.save_img, subject.copy_orig_name, orig_data, orig)

        if not conf.is_conform(
            orig,
            conform_vox_size=self.vox_size,
            check_dtype=True,
            verbose=True,
            conform_to_1mm_threshold=self.conform_to_1mm_threshold,
        ):
            LOGGER.info("Conforming image")
            orig = conf.conform(
                orig,
                conform_vox_size=self.vox_size,
                conform_to_1mm_threshold=self.conform_to_1mm_threshold,
            )
            orig_data = np.asanyarray(orig.dataobj)

        # Save conformed input image
        if subject.can_resolve_attribute("conf_name"):
            self.pool.submit(
                self.save_img, subject.conf_name, orig_data, orig, dtype=np.uint8
            )
        else:
            raise RuntimeError(
                "Cannot resolve the name to the conformed image, please specify an "
                "absolute path."
            )

        return orig, orig_data

    def set_model(self, plane: Plane):
        """
        Set the current model for the specified plane.

        Parameters
        ----------
        plane : Plane
            The plane for which to set the current model.
        """
        self.current_plane = plane

    def get_prediction(
        self, image_name: str, orig_data: np.ndarray, zoom: np.ndarray | Sequence[int],
    ) -> np.ndarray:
        """
        Run and get prediction.

        Parameters
        ----------
        image_name : str
            Original image filename.
        orig_data : np.ndarray
            Original image data.
        zoom : np.ndarray, tuple
            Original zoom.

        Returns
        -------
        np.ndarray
            Predicted classes.
        """
        shape = orig_data.shape + (self.get_num_classes(),)
        kwargs = {
            "device": self.viewagg_device,
            "dtype": torch.float16,
            "requires_grad": False,
        }

        pred_prob = torch.zeros(shape, **kwargs)

        # inference and view aggregation
        for plane, model in self.models.items():
            LOGGER.info(f"Run {plane} prediction")
            self.set_model(plane)
            # pred_prob is updated inplace to conserve memory
            pred_prob = model.run(pred_prob, image_name, orig_data, zoom, out=pred_prob)

        # Get hard predictions
        pred_classes = torch.argmax(pred_prob, 3)
        del pred_prob
        # map to freesurfer label space
        pred_classes = du.map_label2aparc_aseg(pred_classes, self.labels)
        # return numpy array
        # TODO: split_cortex_labels requires a numpy ndarray input, maybe we can also
        #  use Mapper here
        pred_classes = du.split_cortex_labels(pred_classes.cpu().numpy())
        return pred_classes

    def save_img(
        self,
        save_as: str | Path,
        data: np.ndarray | torch.Tensor,
        orig: nib.analyze.SpatialImage,
        dtype: Optional[type] = None,
    ) -> None:
        """
        Save image as a file.

        Parameters
        ----------
        save_as : str, Path
            Filename to give the image.
        data : np.ndarray, torch.Tensor
            Image data.
        orig : nib.analyze.SpatialImage
            Original Image.
        dtype : type, optional
            Data type to use for saving the image. If None, the original data type is
            used (Default value = None).
        """
        save_as = Path(save_as)
        # Create output directory if it does not already exist.
        if not save_as.parent.exists():
            LOGGER.info(
                f"Output image directory {save_as.parent} does not exist. "
                f"Creating it now..."
            )
            save_as.parent.mkdir(parents=True)

        np_data = data if isinstance(data, np.ndarray) else data.cpu().numpy()
        if dtype is not None:
            _header = orig.header.copy()
            _header.set_data_dtype(dtype)
        else:
            _header = orig.header
        du.save_image(_header, orig.affine, np_data, save_as, dtype=dtype)
        LOGGER.info(
            f"Successfully saved image {'asynchronously ' if self._async_io else ''}  as {save_as}."
        )

    def async_save_img(
        self,
        save_as: str | Path,
        data: np.ndarray | torch.Tensor,
        orig: nib.analyze.SpatialImage,
        dtype: type | None = None,
    ) -> Future[None]:
        """
        Save the image asynchronously and return a concurrent.futures.Future to track,
        when this finished.

        Parameters
        ----------
        save_as : str, Path
            Filename to give the image.
        data : Union[np.ndarray, torch.Tensor]
            Image data.
        orig : nib.analyze.SpatialImage
            Original Image.
        dtype : type, optional
            Data type to use for saving the image. If None, the original data type is
            used.

        Returns
        -------
        Future[None]
            A Future object to synchronize (and catch/handle exceptions in the save_img
            method).
        """
        return self.pool.submit(self.save_img, save_as, data, orig, dtype)

    def set_up_model_params(
            self,
            plane: Plane,
            cfg: "yacs.config.CfgNode",
            ckpt: "torch.Tensor",
    ) -> None:
        """
        Set up the model parameters from the configuration and checkpoint.
        """
        self.view_ops[plane]["cfg"] = cfg
        self.view_ops[plane]["ckpt"] = ckpt

    def get_num_classes(self) -> int:
        """
        Return the number of classes.

        Returns
        -------
        int
            The number of classes.
        """
        return self.num_classes

    def pipeline_conform_and_save_orig(
        self, subjects: SubjectList,
    ) -> Iterator[tuple[SubjectDirectory, tuple[nib.analyze.SpatialImage, np.ndarray]]]:
        """
        Pipeline for conforming and saving original images asynchronously.

        Parameters
        ----------
        subjects : SubjectList
            List of subjects to process.

        Yields
        ------
        tuple[SubjectDirectory, tuple[nib.analyze.SpatialImage, np.ndarray]]
            Subject directory and a tuple with the image and its data.
        """
        if not self._async_io:
            # do not pipeline, direct iteration and function call
            for subject in subjects:
                # yield subject and load orig
                yield subject, self.conform_and_save_orig(subject)
        else:
            # pipeline the same
            for data in pipeline(self.pool, self.conform_and_save_orig, subjects):
                yield data


def make_parser():
    """
    Create the argparse object.

    Returns
    -------
    argparse.ArgumentParser
        The parser object.
    """
    parser = argparse.ArgumentParser(description="Evaluation metrics")

    # 1. Options for input directories and filenames
    parser = parser_defaults.add_arguments(
        parser, ["t1", "sid", "in_dir", "tag", "csv_file", "lut", "remove_suffix"]
    )

    # 2. Options for output
    parser = parser_defaults.add_arguments(
        parser,
        [
            "asegdkt_segfile",
            "conformed_name",
            "brainmask_name",
            "aseg_name",
            "sd",
            "seg_log",
            "qc_log",
        ],
    )

    # 3. Checkpoint to load
    files: dict[Plane, str | Path] = {k: "default" for k in PLANES}
    parser = parser_defaults.add_plane_flags(
        parser,
        "checkpoint",
        files,
        CHECKPOINT_PATHS_FILE
    )

    # 4. CFG-file with default options for network
    parser = parser_defaults.add_plane_flags(
        parser,
        "config",
        files,
        CHECKPOINT_PATHS_FILE
    )

    # 5. technical parameters
    parser = parser_defaults.add_arguments(
        parser,
        [
            "vox_size",
            "conform_to_1mm_threshold",
            "device",
            "viewagg_device",
            "batch_size",
            "async_io",
            "threads",
            "allow_root",
        ],
    )
    return parser

def main(
        *,
        orig_name: Path | str,
        out_dir: Path,
        segfile: str,
        ckpt_ax: Path,
        ckpt_sag: Path,
        ckpt_cor: Path,
        cfg_ax: Path,
        cfg_sag: Path,
        cfg_cor: Path,
        seg_log: Path,
        qc_log: str = "",
        log_name: str = "",
        allow_root: bool = False,
        conf_name: str = "mri/orig.mgz",
        in_dir: Optional[Path] = None,
        sid: Optional[str] = None,
        search_tag: Optional[str] = None,
        csv_file: Optional[str | Path] = None,
        lut: Optional[Path | str] = None,
        remove_suffix: str = "",
        brainmask_name: str = "mri/mask.mgz",
        aseg_name: str = "mri/aseg.auto_noCC.mgz",
        vox_size: VoxSizeOption = "min",
        device: str = "auto",
        viewagg_device: str = "auto",
        batch_size: int = 1,
        async_io: bool = True,
        threads: int = -1,
        conform_to_1mm_threshold: float = 0.95,
        **kwargs,
):
    # Warning if run as root user
    allow_root or assert_no_root()

    if len(kwargs) > 0:
        LOGGER.warning(f"Unknown arguments {list(kwargs.keys())} in {__file__}:main.")

    qc_file_handle = None
    if qc_log != "":
        try:
            qc_file_handle = open(qc_log, "w")
        except NotADirectoryError:
            LOGGER.warning(
                "The directory in the provided QC log file path does not exist!"
            )
            LOGGER.warning("The QC log file will not be saved.")

    # Download checkpoints if they do not exist
    # see utils/checkpoint.py for default paths
    LOGGER.info("Checking or downloading default checkpoints ...")
    
    urls = load_checkpoint_config_defaults("url", filename=CHECKPOINT_PATHS_FILE)

    get_checkpoints(ckpt_ax, ckpt_cor, ckpt_sag, urls=urls)

    config = SubjectDirectoryConfig(
        orig_name=orig_name,
        pred_name=segfile,
        conf_name=conf_name,
        in_dir=in_dir,
        csv_file=csv_file,
        sid=sid,
        search_tag=search_tag,
        brainmask_name=brainmask_name,
        remove_suffix=remove_suffix,
        out_dir=out_dir,
    )
    config.copy_org_name = "mri/orig/001.mgz"

    # Get all subjects of interest
    subjects = SubjectList(config, segfile="pred_name", copy_orig_name="copy_orig_name")
    subjects.make_subjects_dir()

    # Set Up Model
    eval = RunModelOnData(
        lut=lut,
        ckpt_ax=ckpt_ax,
        ckpt_sag=ckpt_sag,
        ckpt_cor=ckpt_cor,
        cfg_ax=cfg_ax,
        cfg_sag=cfg_sag,
        cfg_cor=cfg_cor,
        device=device,
        viewagg_device=viewagg_device,
        threads=threads,
        batch_size=batch_size,
        vox_size=vox_size,
        async_io=async_io,
        conform_to_1mm_threshold=conform_to_1mm_threshold,
    )

    qc_failed_subject_count = 0

    iter_subjects = eval.pipeline_conform_and_save_orig(subjects)
    futures = []
    for subject, (orig_img, data_array) in iter_subjects:
        # Run model
        try:
            # The orig_t1_file is only used to populate verbose messages here
            pred_data = eval.get_prediction(
                subject.orig_name, data_array, orig_img.header.get_zooms()
            )
            futures.append(
                eval.async_save_img(
                    subject.segfile, pred_data, orig_img, dtype=np.int16
                )
            )

            # Create aseg and brainmask

            # There is a funny edge case in legacy FastSurfer 2.0, where the behavior is
            # not well-defined, if orig_name is an absolute path, but out_dir is not
            # set. Then, we would create a sub-folder in the folder of orig_name using
            # the subject_id (passed by --sid or extracted from the orig_name) and use
            # that as the subject folder.
            bm = None
            store_brainmask = subject.can_resolve_filename(brainmask_name)
            store_aseg = subject.can_resolve_filename(aseg_name)
            if store_brainmask or store_aseg:
                LOGGER.info("Creating brainmask based on segmentation...")
                bm = rta.create_mask(copy.deepcopy(pred_data), 5, 4)
            if store_brainmask:
                # get mask
                mask_name = subject.filename_in_subject_folder(brainmask_name)
                futures.append(
                    eval.async_save_img(mask_name, bm, orig_img, dtype=np.uint8)
                )
            else:
                LOGGER.info(
                    "Not saving the brainmask, because we could not figure out where "
                    "to store it. Please specify a subject id with {sid[flag]}, or an "
                    "absolute brainmask path with {brainmask_name[flag]}.".format(
                        **subjects.flags,
                    )
                )

            if store_aseg:
                # reduce aparc to aseg and mask regions
                LOGGER.info("Creating aseg based on segmentation...")
                aseg = rta.reduce_to_aseg(pred_data)
                aseg[bm == 0] = 0
                aseg = rta.flip_wm_islands(aseg)
                aseg_name = subject.filename_in_subject_folder(aseg_name)
                # Change datatype to np.uint8, else mri_cc will fail!
                futures.append(
                    eval.async_save_img(aseg_name, aseg, orig_img, dtype=np.uint8)
                )
            else:
                LOGGER.info(
                    "Not saving the aseg file, because we could not figure out where "
                    "to store it. Please specify a subject id with {sid[flag]}, or an "
                    "absolute aseg path with {aseg_name[flag]}.".format(
                        **subjects.flags,
                    )
                )

            # Run QC check
            LOGGER.info("Running volume-based QC check on segmentation...")
            seg_voxvol = np.prod(orig_img.header.get_zooms())
            if not check_volume(pred_data, seg_voxvol):
                LOGGER.warning(
                    "Total segmentation volume is too small. Segmentation may be "
                    "corrupted."
                )
                if qc_file_handle is not None:
                    qc_file_handle.write(subject.id + "\n")
                    qc_file_handle.flush()
                qc_failed_subject_count += 1
        except RuntimeError as e:
            if not handle_cuda_memory_exception(e):
                return e.args[0]

    if qc_file_handle is not None:
        qc_file_handle.close()

    # Batch case: report ratio of QC warnings
    if len(subjects) > 1:
        LOGGER.info(
            f"Segmentations from {qc_failed_subject_count} out of {len(subjects)} "
            f"processed cases failed the volume-based QC check."
        )

    # wait for async processes to finish
    for f in futures:
        _ = f.result()
    return 0


if __name__ == "__main__":
    parser = make_parser()
    _args = parser.parse_args()

    # Set up logging
    from FastSurferCNN.utils.logging import setup_logging
    setup_logging(_args.log_name)

    sys.exit(main(**vars(_args)))
