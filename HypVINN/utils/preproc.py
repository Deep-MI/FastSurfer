# Copyright 2024 AI in Medical Imaging, German Center for Neurodegenerative Diseases(DZNE), Bonn
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

import argparse

import time
from pathlib import Path

import nibabel as nib
import os
import numpy as np
from HypVINN.data_loader.data_utils import rescale_image
from FastSurferCNN.data_loader import data_utils as du
from FastSurferCNN.utils import logging
from HypVINN.utils import ModalityMode, RegistrationMode

LOGGER = logging.get_logger(__name__)


def t1_to_t2_registration(
        t1_path: Path,
        t2_path: Path,
        subject_dir: Path,
        registration_type: RegistrationMode = "coreg",
        threads: int = -1,
) -> Path:
    """
    Register T1 to T2 images using either mri_coreg or mri_robust_register.

    Parameters
    ----------
    t1_path : Path
        The path to the T1 image.
    t2_path : Path
        The path to the T2 image.
    subject_dir : Path
        The directory of the subject.
    registration_type : RegistrationMode, default="coreg"
        The type of registration to be used. It can be either "coreg" or "robust".
    threads : int, default=-1
        The number of threads to be used. If it is less than or equal to 0, the number of threads will be automatically
        determined.

    Returns
    -------
    Path
        The path to the registered T2 image.

    Raises
    ------
    RuntimeError
        If mri_coreg, mri_vol2vol, or mri_robust_register fails to run or if they cannot be found.
    """
    from FastSurferCNN.utils.run_tools import Popen
    from FastSurferCNN.utils.threads import get_num_threads
    import shutil

    if threads <= 0:
        threads = get_num_threads()

    lta_path = subject_dir / "mri/transforms/t2tot1.lta"
    t2_reg_path = subject_dir / "mri/T2_nu_reg.mgz"

    if registration_type == "coreg":
        exe = shutil.which("mri_coreg")
        if not bool(exe):
            if os.environ.get("FREESURFER_HOME", ""):
                exe = os.environ["FREESURFER_HOME"] + "/bin/mri_coreg"
            else:
                raise RuntimeError(
                    "Could not find mri_coreg, source FreeSurfer or set the "
                    "FREESURFER_HOME environment variable"
                )
        args = [exe, "--mov", t2_path, "--targ", t1_path, "--reg", lta_path]
        args = list(map(str, args)) + ["--threads", str(threads)]
        LOGGER.info("Running " + " ".join(args))
        retval = Popen(args).finish()
        if retval.retcode != 0:
            LOGGER.error(f"mri_coreg failed with error code {retval.retcode}. ")
            raise RuntimeError("mri_coreg failed registration")

        else:
            LOGGER.info(f"{exe} finished in {retval.runtime}!")
            exe = shutil.which("mri_vol2vol")
            if not bool(exe):
                if os.environ.get("FREESURFER_HOME", ""):
                    exe = os.environ["FREESURFER_HOME"] + "/bin/mri_vol2vol"
                else:
                    raise RuntimeError(
                        "Could not find mri_vol2vol, source FreeSurfer or set "
                        "the FREESURFER_HOME environment variable"
                    )
            args = [
                exe,
                "--mov", t2_path,
                "--targ", t1_path,
                "--reg", lta_path,
                "--o", t2_reg_path,
                "--cubic",
                "--keep-precision",
            ]
            args = list(map(str, args))
            LOGGER.info("Running " + " ".join(args))
            retval = Popen(args).finish()
            if retval.retcode != 0:
                LOGGER.error(
                    f"mri_vol2vol failed with error code {retval.retcode}."
                )
                raise RuntimeError("mri_vol2vol failed applying registration")
            LOGGER.info(f"{exe} finished in {retval.runtime}!")
    else:
        exe = shutil.which("mri_robust_register")
        if not bool(exe):
            if os.environ.get("FREESURFER_HOME", ""):
                exe = os.environ["FREESURFER_HOME"] + "/bin/mri_robust_register"
            else:
                raise RuntimeError(
                    "Could not find mri_robust_register, source FreeSurfer or "
                    "set the FREESURFER_HOME environment variable"
                )
        args = [
            exe,
            "--mov", t2_path,
            "--dst", t1_path,
            "--lta", lta_path,
            "--mapmov", t2_reg_path,
            "--cost NMI",
        ]
        args = list(map(str, args))
        LOGGER.info("Running " + " ".join(args))
        retval = Popen(args).finish()
        if retval.retcode != 0:
            LOGGER.error(
                f"mri_robust_register failed with error code {retval.retcode}."
            )
            raise RuntimeError("mri_robust_register failed registration")
        LOGGER.info(f"{exe} finished in {retval.runtime}!")

    return t2_reg_path


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
        The registration mode. If it is not "none", the function will register T1 to T2 images.
    t1_path : Path
        The path to the T1 image.
    t2_path : Path
        The path to the T2 image.
    subject_dir : Path
        The directory of the subject.
    threads : int, default=-1
        The number of threads to be used. If it is less than or equal to 0, the number of threads will be automatically
        determined.

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
        raise RuntimeError(
            "hypvinn_preproc should only be called for t1t2 mode."
        )
    if reg_mode != "none":
        load_res = time.time()
        # Print Warning if Resolution from both images is different
        t1_zoom = nib.load(t1_path).header.get_zooms()
        t2_zoom = nib.load(t2_path).header.get_zooms()

        if not np.allclose(np.array(t1_zoom), np.array(t2_zoom), rtol=0.05):
            LOGGER.info(
                f"Warning: Resolution from T1 ({t1_zoom}) and T2 ({t2_zoom}) image "
                f"are different.\nResolution of the T2 image will be interpolated "
                "to the one of the T1 image."
            )

        LOGGER.info("Registering T1 to T2 ...")
        t2_path = t1_to_t2_registration(
            t1_path=t1_path,
            t2_path=t2_path,
            subject_dir=subject_dir,
            registration_type=reg_mode,
            threads=threads,
        )
        LOGGER.info(
            f"Registration finish in {time.time() - load_res:0.4f} seconds!"
        )
    else:
        LOGGER.info(
            "Warning: No registration step, registering T1w and T2w is "
            "required when running the multi-modal input mode.\n "
            "No register images can generate wrong predictions. Omit this "
            "message if input images are already registered."
        )

    LOGGER.info("---" * 30)

    return t2_path
