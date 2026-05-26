# Copyright 2024
# AI in Medical Imaging, German Center for Neurodegenerative Diseases(DZNE), Bonn
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

import time
from pathlib import Path
from typing import cast

import nibabel as nib
import numpy as np

from FastSurferCNN.utils import logging
from HypVINN.utils import ModalityMode, RegistrationMode

LOGGER = logging.get_logger(__name__)


def t1_to_t2_registration(
        t1_path: Path,
        t2_path: Path,
        output_path: Path,
        lta_path: Path,
        registration_type: RegistrationMode = "coreg",
        threads: int = -1,
) -> Path:
    """
    Register the T2 image to the T1 reference using neuroreg.

    Parameters
    ----------
    t1_path : Path
        The path to the T1 image.
    t2_path : Path
        The path to the T2 image.
    output_path : Path
        The path to the output/registered image.
    lta_path : Path
        The path to the lta transform.
    registration_type : RegistrationMode, default="coreg"
        The type of registration to be used. It can be either "coreg" or "none".
    threads : int, default=-1
        The number of threads to be used. If it is less than or equal to 0, the number
        of threads will be automatically determined.

    Returns
    -------
    Path
        The path to the registered T2 image.

    Raises
    ------
    RuntimeError
        If the requested neuroreg registration backend fails.
    """
    from neuroreg import coreg

    if registration_type == "coreg":
        LOGGER.info("Running neuroreg.coreg for T2-to-T1 registration.")
        coreg(
            str(t2_path),
            str(t1_path),
            lta_name=str(lta_path),
            mapped_name=str(output_path),
        )
    else:
        raise ValueError(f"Unknown registration type: {registration_type}")

    return output_path


def hypvinn_preproc(
        mode: ModalityMode,
        reg_mode: RegistrationMode,
        t1_path: Path,
        t2_path: Path,
        subject_dir: Path,
        threads: int = -1,
) -> Path:
    """
    Preprocess the input images for HypVINN.

    Parameters
    ----------
    mode : ModalityMode
        The mode for HypVINN. It should be "t1t2".
    reg_mode : RegistrationMode
        The registration mode. If it is not "none", the function will register the
        T2 image to the T1 reference.
    t1_path : Path
        The path to the T1 image.
    t2_path : Path
        The path to the T2 image.
    subject_dir : Path
        The directory of the subject.
    threads : int, default=-1
        The number of threads to be used. If it is less than or equal to 0, the number of threads will be
        automatically determined.

    Returns
    -------
    Path
        The path to the preprocessed T2 image.

    Raises
    ------
    RuntimeError
        If the mode is not "t1t2", or if the registration mode is not "none" and the registration fails.
    """
    if mode != "t1t2":
        raise RuntimeError("hypvinn_preproc should only be called for t1t2 mode.")
    registered_t2_path = subject_dir / "mri/T2_nu_reg.mgz"
    if reg_mode != "none":
        from nibabel.analyze import AnalyzeImage
        load_res = time.time()
        # Print Warning if Resolution from both images is different
        t1_zoom = cast(AnalyzeImage, nib.load(t1_path)).header.get_zooms()
        t2_zoom = cast(AnalyzeImage, nib.load(t2_path)).header.get_zooms()

        if not np.allclose(np.array(t1_zoom), np.array(t2_zoom), rtol=0.05):
            LOGGER.warning(
                f"Resolution from T1 {t1_zoom} and T2 {t2_zoom} image are different.\n"
                f"T2 image will be interpolated to the resolution of the T1 image."
            )

        LOGGER.info("Registering T2 to T1 ...")
        t1_to_t2_registration(
            t1_path=t1_path,
            t2_path=t2_path,
            output_path=registered_t2_path,
            lta_path=subject_dir / "mri/transforms/t2tot1.lta",
            registration_type=reg_mode,
            threads=threads,
        )
        LOGGER.info(f"Registration finish in {time.time() - load_res:0.4f} seconds!")
    else:
        LOGGER.info(
            "No registration step, registering T1w and T2w is required when running "
            "the multi-modal input mode.\nUnregistered images can generate wrong "
            "predictions. Ignore this message, if input images are already registered."
        )
        try:
            registered_t2_path.symlink_to(os.path.relpath(t2_path, registered_t2_path.parent))
        except FileNotFoundError as e:
            msg = f"Could not create symlink. Does the folder {registered_t2_path.parent} exist?"
            LOGGER.error(msg)
            raise FileNotFoundError(msg) from e
        except (RuntimeError, OSError):
            LOGGER.info(f"Could not create symlink for {registered_t2_path}, copying.")
            from shutil import copy
            copy(t2_path, registered_t2_path)

    LOGGER.info("---" * 30)

    return registered_t2_path
