
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


# IMPORTS
import sys
sys.path.append("./FastSurferCNN")
from FastSurferCNN.eval import fast_surfer_cnn
import subprocess as sp
import shlex
import argparse


def setup_options():
    # Validation settings
    parser = argparse.ArgumentParser(description='Wrapper for fastsurfer (FastSurferCNN + recon-surf)')

    # Requiered options
    # 1. Directory information (location of T1 images to segment and where to store the output)
    parser.add_argument('--t1', help="T1 full head input (not bias corrected). ABSOLUTE Path!", type=str, required=True)
    parser.add_argument('--seg', default="aparc.DKTatlas+aseg.deep.mgz", type=str,
                        help="Name of segmentation (similar to aparc+aseg). Default: aparc.DKTatlas+aseg.deep.mgz. "
                             "Specifiy ABSOLUTE Path!")
    parser.add_argument('--sd', '--subjects_dir',
                        help="Output directory $SUBJECTS_DIR (FreeSurfer style) for surfaces & Co. ABSOLUTE Path!")
    parser.add_argument('--sid', '--subjectID', help="Subject ID for directory inside $SUBJECTS_DIR to be created.")

    # 2. Options for log-file and search-tag
    parser.add_argument('--t', '--tag', dest='search_tag', default="*",
                        help='Search tag to process only certain subjects in the directory specified with --t1_dir. '
                             'If a single image should be analyzed, set the tag with its id. Default: processes all.')
    parser.add_argument('--log', dest='logfile', default='deep_surfer.log',
                        help='Name of log-file. Will be stored in same directory as segmentation. '
                             'Default: deep_surfer.log')

    # 3. Location of Pre-trained weights
    parser.add_argument('--network_sagittal_path', dest='network_sagittal_path',
                        help="path to pre-trained weights of sagittal network",
                        default='./checkpoints/Sagittal_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl')
    parser.add_argument('--network_coronal_path', dest='network_coronal_path',
                        help="pre-trained weights of coronal network",
                        default='./checkpoints/Coronal_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl')
    parser.add_argument('--network_axial_path', dest='network_axial_path',
                        help="pre-trained weights of axial network",
                        default='./checkpoints/Axial_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl')

    # 4. Clean up and GPU/CPU options (disable cuda, change batchsize, order of interpolation for conforming image)
    parser.add_argument('--clean', dest='cleanup', help="Flag to clean up FastSurferCNN segmentation",
                        action='store_true')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA usage in FastSurferCNN (no GPU usage)')
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for inference. Default: 16")
    parser.add_argument('--order', dest='order', type=int, default=1,
                        help="order of interpolation (0=nearest,1=linear(default),2=quadratic,3=cubic)")

    # 5. Further options for recon-surf part
    parser.add_argument('--mc', default="", action="store_const", const=" --mc",
                        help="Switch on marching cube for surface creation. "
                             "Otherwise, run tesselate lego land (mri_tesselate).")
    parser.add_argument('--qspec', default="", action="store_const", const=" --qspec",
                        help="Switch on spectral spherical projection for qsphere. "
                             "Otherwise, run mri_qspehere from FreeSurfer.")
    parser.add_argument('--nofsaparc', default="", action="store_const", const=" --nofsaparc",
                        help="Skip FS aparc segmentations and ribbon for speedup")
    parser.add_argument('--surfreg', default="", action="store_const", const=" --surfreg",
                        help="Switch on Surface registration (for cross-subject correspondance)")
    parser.add_argument('--parallel', default="", action="store_const", const=" --parallel",
                        help="Run both hemispheres in parallel (speed-up)")

    parser.add_argument('--py', type=str, default="python3.6",
                        help="which python version to use. Default: python3.6")
    parser.add_argument('--n_threads', type=str, default="1",
                        help="Number of threads to use (for openMP and ITK). Default=1")

    args = parser.parse_args()
    return args


def fastsurfer(options):
    """
    Run FastSurfer --> FastSurferCNN + recon-surf
      * FastSurferCNN: generates aparc.DKTatlas segmentation from T1-weighted MRI brain scan
      * recon-surf: uses segmentation to create surface models and associated measures (basic FreeSurfer output)
    :param argparse.Arguments options: Set up options passed via the command line
                                      (see run_fastsurfer.py --help for details)
    :return: Error code of recon-surf (if any)
    """

    # Run FastSurferCNN to get segmentation
    fast_surfer_cnn(options.t1, options.seg, options)

    # Run recon-surf to get surfaces
    cmd1 = "./recon-surf.sh --sid " + options.sid + \
           " --sd " + options.sd + \
           " --t1 " + options.t1 + \
           " --seg " + options.seg + \
           options.mc + \
           options.qspec + \
           options.nofsaparc + \
           options.surfreg + \
           options.parallel + \
           " --threads " + options.n_threads + \
           " --py " + options.py

    command_split = shlex.split(cmd1)
    p = sp.run(command_split, cwd="./recon_surf/")
    return p.returncode


if __name__ == "__main__":
    options_sel = setup_options()
    returncode = fastsurfer(options_sel)
