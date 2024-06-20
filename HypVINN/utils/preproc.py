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

from FastSurferCNN.utils import logging
from HypVINN.utils import ModalityMode, RegistrationMode

LOGGER = logging.get_logger(__name__)


def t1_to_t2_registration(
        t1_path: Path,
        t2_path: Path,
        out_dir: Path,
        registration_type: RegistrationMode = "coreg",
) -> Path:
    from FastSurferCNN.utils.run_tools import Popen
    import shutil

    lta_path = out_dir / "mri/transforms/t2tot1.lta"

    t2_reg_path = out_dir / "mri/T2_nu_reg.mgz"

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
        LOGGER.info("Running " + " ".join(args))
        retval = Popen(args).finish()
        if retval.retcode != 0:
            LOGGER.error(f"mri_coreg failed with error code {retval.retcode}. ")
            raise RuntimeError("mri_coreg failed registration")

        else:
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
            LOGGER.info("Running " + " ".join(args))
            retval = Popen(args).finish()
            if retval.retcode != 0:
                LOGGER.error(
                    f"mri_vol2vol failed with error code {retval.retcode}."
                )
                raise RuntimeError("mri_vol2vol failed applying registration")
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
        LOGGER.info("Running " + " ".join(args))
        retval = Popen(args).finish()
        if retval.retcode != 0:
            LOGGER.error(
                f"mri_robust_register failed with error code {retval.retcode}."
            )
            raise RuntimeError("mri_robust_register failed registration")

    return t2_reg_path


def hyvinn_preproc(
        mode: ModalityMode,
        reg_mode: RegistrationMode,
        t1_path: Path,
        t2_path: Path,
        out_dir: Path,
) -> Path:

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
                f"Warning: Resolution from T1 ({t1_zoom}) and T2 "
                f"({t2_zoom}) image are different.\n "
                "Resolution of the T2 image will be interpolated "
                "to the one of the T1 image."
            )

        LOGGER.info("Registering T1 to T2 ...")
        t2_path = t1_to_t2_registration(
            t1_path=t1_path,
            t2_path=t2_path,
            out_dir=out_dir,
            registration_type=reg_mode,
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
