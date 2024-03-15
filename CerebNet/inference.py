# Copyright 2022 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
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

# IMPORTS
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING
from concurrent.futures import Future, ThreadPoolExecutor

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from FastSurferCNN.utils import logging, Plane, PLANES
from FastSurferCNN.utils.threads import get_num_threads
from FastSurferCNN.utils.mapper import JsonColorLookupTable, TSVLookupTable
from FastSurferCNN.utils.common import (
    find_device,
    SubjectList,
    SubjectDirectory,
    SerialExecutor,
)
from CerebNet.data_loader.augmentation import ToTensorTest
from CerebNet.data_loader.dataset import SubjectDataset
from CerebNet.datasets.utils import crop_transform
from CerebNet.models.networks import build_model
from CerebNet.utils import checkpoint as cp

if TYPE_CHECKING:
    import yacs.config

logger = logging.get_logger(__name__)


class Inference:
    """
    Manages inference operations, including batch processing, data loading, and model
    predictions for neuroimaging data.
    """
    def __init__(
        self,
        cfg: "yacs.config.CfgNode",
        threads: int = -1,
        async_io: bool = False,
        device: str = "auto",
        viewagg_device: str = "auto",
    ):
        """
        Create the inference object to manage inferencing, batch processing, data
        loading, etc.

        Parameters
        ----------
        cfg : yacs.config.CfgNode
            Yaml configuration to populate default values for parameters.
        threads : int, optional
            Number of threads to use, -1 is max (all), which is also the default.
        async_io : bool, default=False
            Whether io is run asynchronously.
        device : str, default="auto"
            Device to perform inference on.
        viewagg_device : str, default="auto"
            Device to aggregate views on.
        """
        self.pool = None
        self._threads = None
        self.threads = threads
        _threads = get_num_threads() if self._threads is None else self._threads
        torch.set_num_threads(_threads)
        self.pool = ThreadPoolExecutor(self._threads) if async_io else SerialExecutor()
        self.cfg = cfg
        self._async_io = async_io

        # Set random seed from config_files.
        np.random.seed(cfg.RNG_SEED)
        torch.manual_seed(cfg.RNG_SEED)

        _device = find_device(device)
        if _device == "cpu" and viewagg_device == "auto":
            _viewagg_device = torch.device("cpu")
        else:
            _viewagg_device = find_device(
                viewagg_device,
                flag_name="viewagg_device",
                min_memory=2 * (2**30),
            )

        self.batch_size = cfg.TEST.BATCH_SIZE
        _models = self._load_model(cfg)
        self.device = _device
        self.viewagg_device = _viewagg_device


        def prep_lut(
                file: Path, *args, **kwargs,
        ) -> Future[TSVLookupTable | JsonColorLookupTable]:
            _cls = TSVLookupTable
            cls = {".json": JsonColorLookupTable, ".txt": _cls, ".tsv": _cls}
            return self.pool.submit(cls[file.suffix], file, *args, **kwargs)

        def lut_path(module: str, file: str) -> Path:
            return cp.FASTSURFER_ROOT / module / "config" / file

        cerebnet_labels_file = lut_path("CerebNet", "CerebNet_ColorLUT.tsv")
        _cerebnet_mapper = prep_lut(cerebnet_labels_file, header=True)

        self.freesurfer_lut_file = lut_path("FastSurferCNN", "FreeSurferColorLUT.txt")
        fs_color_map = prep_lut(self.freesurfer_lut_file, header=False)

        cerebnet2sagittal_lut = lut_path("CerebNet", "CerebNet2Sagittal.json")
        sagittal_cereb2cereb_mapper = prep_lut(cerebnet2sagittal_lut)

        cerebnet2freesurfer_lut = lut_path("CerebNet", "CerebNet2FreeSurfer.json")
        cereb2freesurfer_mapper = prep_lut(cerebnet2freesurfer_lut)

        self.cerebnet_labels = _cerebnet_mapper.result().labelname2id()
        self.freesurfer_name2id = fs_color_map.result().labelname2id()
        cereb_name2fs_name = cereb2freesurfer_mapper.result().labelname2id()
        cerebsag_name2cereb_name = sagittal_cereb2cereb_mapper.result().labelname2id()

        cereb_id2name = self.cerebnet_labels.__reversed__()
        self.cereb_name2fs_id = cereb_name2fs_name.chain(self.freesurfer_name2id)
        self.cereb_id2fs_id = cereb_id2name.chain(self.cereb_name2fs_id)
        self.cerebsag_id2cereb_name = cereb_id2name.chain(cerebsag_name2cereb_name)
        self.models = {k: m.to(self.device) for k, m in _models.items()}

    @property
    def threads(self) -> int:
        return -1 if self._threads is None else self._threads

    @threads.setter
    def threads(self, threads: int):
        self._threads = threads if threads > 0 else None

    def __del__(self):
        """Make sure the pool gets shut down when the Inference object gets deleted."""
        if self.pool is not None:
            self.pool.shutdown(True)

    def _load_model(self, cfg) -> Dict[Plane, torch.nn.Module]:
        """Loads the three models per plane."""

        def __load_model(cfg: "yacs.config.CfgNode", plane: Plane) -> torch.nn.Module:
            params = {k.lower(): v for k, v in dict(cfg.MODEL).items()}
            params["plane"] = plane
            if plane == "sagittal":
                if params["num_classes"] != params["num_classes_sag"]:
                    params["num_classes"] = params["num_classes_sag"]
            checkpoint_path = Path(cfg.TEST[f"{plane.upper()}_CHECKPOINT_PATH"])
            model = build_model(params)
            if not checkpoint_path.is_file():
                # if the checkpoint path is not a file, but a folder search in there for
                # the newest checkpoint
                checkpoint_path = cp.get_checkpoint_path(checkpoint_path).pop()
            cp.load_from_checkpoint(checkpoint_path, model)
            model.eval()
            return model

        from functools import partial

        _load_model_func = partial(__load_model, cfg)
        return dict(zip(PLANES, self.pool.map(_load_model_func, PLANES)))

    @torch.no_grad()
    def _predict_single_subject(
        self, subject_dataset: SubjectDataset
    ) -> Dict[Plane, List[torch.Tensor]]:
        """Predict the classes based on a SubjectDataset."""
        img_loader = DataLoader(
            subject_dataset, batch_size=self.batch_size, shuffle=False
        )
        prediction_logits = {}
        try:
            for plane in PLANES:
                subject_dataset.set_plane(plane)

                predictions = []
                from CerebNet.data_loader.data_utils import slice_lia2ras, slice_ras2lia

                for img in img_loader:
                    # CerebNet is trained on RAS+ conventions, so we need to map between
                    # lia (FastSurfer) and RAS+
                    # map LIA 2 RAS
                    img = slice_lia2ras(plane, img, thick_slices=True)
                    batch = img.to(self.device)
                    pred = self.models[plane](batch)
                    # map RAS 2 LIA
                    pred = slice_ras2lia(plane, pred)
                    pred = pred.to(device=self.viewagg_device, dtype=torch.float16)
                    predictions.append(pred)
                prediction_logits[plane] = predictions
        except RuntimeError as e:
            from FastSurferCNN.utils.common import handle_cuda_memory_exception

            handle_cuda_memory_exception(e)
            raise e
        return prediction_logits

    def _post_process_preds(
        self, preds: Dict[Plane, List[torch.Tensor]]
    ) -> Dict[Plane, torch.Tensor]:
        """
        Permutes axes, so it has consistent sagittal, coronal, axial, channels format.
        Also maps classes of sagittal predictions into the global label space.

        Parameters
        ----------
        preds:
            predicted logits.

        Returns
        -------
            dictionary of permuted logits.
        """
        axis_permutation = {
            # a,_, s, c -> s, c, a, _
            "axial": (3, 0, 2, 1),
            # c, _, s, a -> s, c, a, _
            "coronal": (2, 3, 0, 1),
            # s, _, c, a -> s, c, a, _
            "sagittal": (0, 3, 2, 1),
        }

        def _convert(plane: Plane) -> torch.Tensor:
            pred = torch.cat(preds[plane], dim=0)
            if plane == "sagittal":
                pred = self.cerebsag_id2cereb_name.map_probs(pred, axis=1, reverse=True)
            return pred.permute(axis_permutation[plane])

        return {plane: _convert(plane) for plane in preds.keys()}

    def _view_aggregation(self, logits: Dict[Plane, torch.Tensor]) -> torch.Tensor:
        """
        Aggregate the view (axial, coronal, sagittal) into one volume and get the class of the largest probability. (argmax)

        Args:
            logits: dictionary of per plane predicted logits (axial, coronal, sagittal)

        Returns:
            Tensor of classes (of largest aggregated logits)
        """
        aggregated_logits = torch.add(
            (logits["axial"] + logits["coronal"]) * 0.4, logits["sagittal"], alpha=0.2
        )
        _, labels = torch.max(aggregated_logits, dim=3)
        return labels

    def _calc_segstats(
        self, seg_data: np.ndarray, norm_data: np.ndarray, vox_vol: float
    ) -> "pandas.DataFrame":
        """
        Computes volume and volume similarity
        """

        def _get_ids_startswith(_label_map: Dict[int, str], prefix: str) -> List[int]:
            return [
                id
                for id, name in _label_map.items()
                if name.startswith(prefix) and not name.endswith("Medullare")
            ]

        freesurfer_id2cereb_name = self.cereb_name2fs_id.__reversed__()
        freesurfer_id2name = self.freesurfer_name2id.__reversed__()
        label_map = dict(freesurfer_id2cereb_name)
        meta_labels = {
            8: ("Left", "Left-Cerebellum-Cortex"),
            47: ("Right", "Right-Cerebellum-Cortex"),
            632: ("Vermis", "Cbm_Vermis"),
        }
        merge_map = {
            id: _get_ids_startswith(label_map, prefix=prefix)
            for id, (prefix, _) in meta_labels.items()
        }

        # calculate PVE
        from FastSurferCNN.segstats import pv_calc

        table = pv_calc(
            seg_data,
            norm_data,
            norm_data,
            list(filter(lambda l: l != 0, label_map.keys())),
            vox_vol=vox_vol,
            threads=self.threads,
            patch_size=32,
            merged_labels=merge_map,
        )

        # fill the StructName field
        for i in range(len(table)):
            _id = table[i]["SegId"]
            if _id in meta_labels.keys():
                table[i]["StructName"] = meta_labels[_id][1]
            elif _id in freesurfer_id2cereb_name:
                table[i]["StructName"] = freesurfer_id2name[_id]
            else:
                # noinspection PyTypeChecker
                table[i]["StructName"] = "Merged-Label-" + str(_id)

        import pandas as pd

        dataframe = pd.DataFrame(table, index=np.arange(len(table)))
        dataframe = dataframe[dataframe["NVoxels"] != 0].sort_values("SegId")
        dataframe.index = np.arange(1, len(dataframe) + 1)
        return dataframe

    def _save_cerebnet_seg(
            self,
            cerebnet_seg: np.ndarray,
            filename: str | Path,
            orig: nib.analyze.SpatialImage
    ) -> "Future[None]":
        """
        Saving the segmentations asynchronously.

        Parameters
        ----------
        cerebnet_seg : np.ndarray
            Segmentation data.
        filename : Path, str
            Path and file name to the saved file.
        orig : nib.analyze.SpatialImage
            File container (with header and affine) used to populate header and affine
            of the segmentation.

        Returns
        -------
        Future[None]
            A Future to determine when the file was saved. Result is None.
        """
        from FastSurferCNN.data_loader.data_utils import save_image

        if cerebnet_seg.shape != orig.shape:
            raise RuntimeError("Cereb segmentation shape inconsistent with Orig shape!")
        logger.info(f"Saving CerebNet cerebellum segmentation at {filename}")
        return self.pool.submit(
            save_image, orig.header, orig.affine, cerebnet_seg, filename, dtype=np.int16
        )

    def _get_subject_dataset(
        self, subject: SubjectDirectory
    ) -> Tuple[Optional[np.ndarray], Optional[Path], SubjectDataset]:
        """
        Load and prepare input files asynchronously, then locate the cerebellum and
        provide a localized patch.
        """

        from FastSurferCNN.data_loader.data_utils import load_image, load_maybe_conform

        norm_file, norm_data, norm = None, None, None
        if subject.has_attribute("cereb_statsfile"):
            if not subject.can_resolve_attribute("cereb_statsfile"):
                from FastSurferCNN.utils.parser_defaults import ALL_FLAGS

                raise ValueError(
                    f"Cannot resolve the intended filename "
                    f"{subject.get_attribute('cereb_statsfile')} for the "
                    f"cereb_statsfile, maybe specify an absolute path via "
                    f"{ALL_FLAGS['cereb_statsfile'](dict)['flag']}."
                )
            if not subject.has_attribute(
                "norm_name"
            ) or not subject.fileexists_by_attribute("norm_name"):
                from FastSurferCNN.utils.parser_defaults import ALL_FLAGS

                raise ValueError(
                    f"Cannot resolve the file name "
                    f"{subject.get_attribute('norm_name')} for the bias field "
                    f"corrected image, maybe specify an absolute path via "
                    f"{ALL_FLAGS['norm_name'](dict)['flag']} or the file does not "
                    f"exist."
                )

            norm_file = subject.filename_by_attribute("norm_name")
            # finally, load the bias field file
            norm = self.pool.submit(
                load_maybe_conform, norm_file, norm_file, vox_size=1.0
            )

        # localization
        if not subject.fileexists_by_attribute("asegdkt_segfile"):
            raise RuntimeError(
                f"The aseg.DKT-segmentation file '{subject.asegdkt_segfile}' did not "
                f"exist, please run FastSurferVINN first."
            )
        seg = self.pool.submit(
            load_image, subject.filename_by_attribute("asegdkt_segfile")
        )
        # create conformed image
        conf_img = self.pool.submit(
            load_maybe_conform,
            subject.filename_by_attribute("conf_name"),
            subject.filename_by_attribute("orig_name"),
            vox_size=1.0,
        )

        seg, seg_data = seg.result()
        conf_file, conf_img, conf_data = conf_img.result()
        subject_dataset = SubjectDataset(
            img_org=conf_img,
            brain_seg=seg,
            patch_size=self.cfg.DATA.PATCH_SIZE,
            slice_thickness=self.cfg.DATA.THICKNESS,
            primary_slice=self.cfg.DATA.PRIMARY_SLICE_DIR,
        )
        subject_dataset.transforms = ToTensorTest()
        if norm is not None:
            norm_file, _, norm_data = norm.result()
        return norm_data, norm_file, subject_dataset

    def run(self, subject_directories: SubjectList):
        logger.info(time.strftime("%y-%m-%d_%H:%M:%S"))

        from tqdm.contrib.logging import logging_redirect_tqdm

        start_time = time.time()
        with logging_redirect_tqdm():
            if self._async_io:
                from FastSurferCNN.utils.common import pipeline as iterate
            else:
                from FastSurferCNN.utils.common import iterate
            iter_subjects = iterate(
                self.pool, self._get_subject_dataset, subject_directories
            )
            futures = []
            for idx, (subject, (norm, norm_file, subject_dataset)) in tqdm(
                enumerate(iter_subjects), total=len(subject_directories), desc="Subject"
            ):
                try:
                    # predict CerebNet, returns logits
                    preds = self._predict_single_subject(subject_dataset)
                    # create the folder for the output file, if it does not exist
                    _mkdir = self.pool.submit(
                        subject.segfile.parent.mkdir, exist_ok=True, parents=True,
                    )

                    # postprocess logits (move axes, map sagittal to all classes)
                    preds_per_plane = self._post_process_preds(preds)
                    # view aggregation in logit space and find max label
                    cerebnet_seg = self._view_aggregation(preds_per_plane)

                    # map predictions into FreeSurfer Label space & move segmentation to
                    # cpu
                    cerebnet_seg = self.cereb_id2fs_id.map(cerebnet_seg).cpu()
                    pred_time = time.time()

                    # uncrop the segmentation
                    bounding_box = subject_dataset.get_bounding_offsets()
                    full_cereb_seg = crop_transform(
                        cerebnet_seg,
                        offsets=tuple(-o for o in bounding_box["offsets"]),
                        target_shape=bounding_box["source_shape"],
                    ).numpy()

                    _ = (
                        _mkdir.result()
                    )  # this is None, but synchronizes the creation of the directory
                    futures.append(
                        self._save_cerebnet_seg(
                            full_cereb_seg,
                            subject.segfile,
                            subject_dataset.get_nibabel_img(),
                        )
                    )

                    if subject.has_attribute("cereb_statsfile"):
                        # vox_vol = np.prod(norm.header.get_zooms()).item()
                        # CerebNet always has vox_vol 1
                        if norm is None:
                            raise RuntimeError("norm not loaded as expected!")
                        df = self._calc_segstats(full_cereb_seg, norm, vox_vol=1.0)
                        from FastSurferCNN.segstats import write_statsfile

                        # in batch processing, we are finished with this subject and the
                        # output of this data can be outsourced to a different process
                        futures.append(
                            self.pool.submit(
                                write_statsfile,
                                subject.filename_by_attribute("cereb_statsfile"),
                                df,
                                vox_vol=1.0,
                                segfile=subject.segfile,
                                normfile=norm_file,
                                lut=self.freesurfer_lut_file,
                            )
                        )

                    logger.info(
                        f"Subject {idx + 1}/{len(subject_directories)} with id "
                        f"'{subject.id}' processed in {pred_time - start_time :.2f} "
                        f"sec."
                    )
                except Exception as e:
                    logger.exception(e)
                    return "\n".join(map(str, e.args))
                else:
                    start_time = time.time()

            # wait for tasks to finish
            for f in futures:
                _ = f.result()

        return 0
