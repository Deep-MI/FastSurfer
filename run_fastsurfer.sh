#!/bin/bash

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


VERSION='$Id$'

# Set default values for arguments
subject=""
t1=""
seg=""
seg_log=""
weights_sag="../checkpoints/Sagittal_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl"
weights_ax="../checkpoints/Axial_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl"
weights_cor="../checkpoints/Coronal_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl"
clean_seg=""
cuda=""
batch_size="16"
order="1"
mc=""
qspec=""
nofsaparc=""
fssurfreg=""
doParallel=""
dev=""
threads="1"
python="python3.6"
fastsurfercnndir="./FastSurferCNN"
reconsurfdir="./recon_surf"

function usage()
{
    echo ""
    echo "run_fastsurfer.sh takes a T1 full head image and creates
              (i) a segmentation using FastSurferCNN (equivalent to FreeSurfers aparc.DKTatlas+aseg.mgz)
              (ii) surfaces, thickness etc as a FS subject dir using recon-surf"
    echo ""
    echo "./run_fastsurfer.sh"
    echo -e "\t--fs_license <freesurfer_license_file>  Path to FreeSurfer license key file. Register (for free) at https://surfer.nmr.mgh.harvard.edu/registration.html to obtain it if you do not have FreeSurfer installed so far."
    echo -e "\t--sid <subjectID>                      Subject ID for directory inside \$SUBJECTS_DIR to be created"
    echo -e "\t--sd  <subjects_dir>                   Output directory \$SUBJECTS_DIR (pass via environment or here)"
    echo -e "\t--t1  <T1_input>                       T1 full head input (not bias corrected)"
    echo -e "\t--seg <segmentation_input>             Name of intermediate DL-based segmentation file (similar to aparc+aseg). Default: aparc.DKTatlas+aseg.deep.mgz. Requires an ABSOLUTE Path!"
    echo -e "\t--seg_log <segmentation_log>           Log-file for the segmentation (FastSurferCNN)"
    echo -e "\t--weights_sag <weights_sagittal>       Pretrained weights of sagittal network. Default: ../checkpoints/Sagittal_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl"
    echo -e "\t--weights_ax <weights_axial>           Pretrained weights of axial network. Default: ../checkpoints/Axial_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl"
    echo -e "\t--weights_cor <weights_coronal>        Pretrained weights of coronal network. Default: ../checkpoints/Coronal_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl"
    echo -e "\t--clean_seg <clean_segmentation>       Flag to clean up FastSurferCNN segmentation"
    echo -e "\t--no_cuda <disable_cuda>               Flag to disable CUDA usage in FastSurferCNN (no GPU usage, inference on CPU)"
    echo -e "\t--batch <batch_size>                   Batch size for inference. Default: 16."
    echo -e "\t--order <order_of_interpolation>       Order of interpolation for mri_convert T1 before segmentation (0=nearest,1=linear(default),2=quadratic,3=cubic)"
    echo -e "\t--mc                                   Switch on marching cube for surface creation"
    echo -e "\t--qspec                                Switch on spectral spherical projection for qsphere"
    echo -e "\t--nofsaparc                            Skip FS aparc segmentations and ribbon for speedup"
    echo -e "\t--surfreg                              Run Surface registration with FreeSurfer (for cross-subject correspondence)"
    echo -e "\t--parallel                             Run both hemispheres in parallel"
    echo -e "\t--threads <int>                        Set openMP and ITK threads to <int>"
    echo -e "\t--dev                                  Switch on if dev-version of FreeSurfer was sourced"
    echo -e "\t--py <python_cmd>                      Command for python, default 'python3.6'"
    echo -e "\t-h --help                              Print Help"
    echo ""
}

# PRINT USAGE if called without params
if [[ $# -eq 0 ]]
then
  usage
  exit
fi

# PARSE Command line
inputargs=("$@")
POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    --fs_license)
    export FS_LICENSE="$2"
    shift # past argument
    shift # past value
    ;;
    --sid)
    subject="$2"
    shift # past argument
    shift # past value
    ;;
    --sd)
    sd="$2"
    shift # past argument
    shift # past value
    ;;
    --t1)
    t1="$2"
    shift # past argument
    shift # past value
    ;;
    --seg)
    seg="$2"
    shift # past argument
    shift # past value
    ;;
    --seg_log)
    seg_log="$2"
    shift # past argument
    shift # past value
    ;;
    --weights_sag)
    weights_sag="$2"
    shift # past argument
    shift # past value
    ;;
    --weights_ax)
    weights_ax="$2"
    shift # past argument
    shift # past value
    ;;
    --weights_cor)
    weights_cor="$2"
    shift # past argument
    shift # past value
    ;;
    --clean_seg)
    clean_seg="--clean"
    shift # past argument
    ;;
    --no_cuda)
    cuda="--no_cuda"
    shift # past argument
    ;;
    --batch)
    batch_size="$2"
    shift # past argument
    shift # past value
    ;;
    --order)
    order=$2
    shift # past argument
    shift # past value
    ;;
    --mc)
    mc="--mc"
    shift # past argument
    ;;
    --qspec)
    qspec="--qspec"
    shift # past argument
    ;;
    --nofsaparc)
    nofsaparc="--nofsaparc"
    shift # past argument
    ;;
    --surfreg)
    fssurfreg="--surfreg"
    shift # past argument
    ;;
    --parallel)
    doParallel="--parallel"
    shift # past argument
    ;;
    --threads)
    threads="$2"
    shift # past argument
    shift # past value
    ;;
    --dev)
    dev="--dev"
    shift # past argument
    ;;
    --py)
    python="$2"
    shift # past argument
    shift # past value
    ;;
    -h|--help)
    usage
    exit
    ;;
    *)    # unknown option
    echo ERROR: Flag $key unrecognized.
    exit 1
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters


# CHECKS
if [ -z "$t1" ]
 then
  echo "ERROR: must supply T1 input (conformed, full head) via --t1"
  exit 1;
fi

if [ -z "$subject" ]
 then
  echo "ERROR: must supply subject name via --sid"
  exit 1;
fi

if [ -z "$seg" ]
 then
  echo "ERROR: must supply output path + name to brain segmentation via --seg"
  exit 1;
fi

########################################## START ########################################################


# "============= Running FastSurferCNN (Creating Segmentation aparc.DKTatlas.aseg.mgz) ==============="
# use FastSurferCNN to create cortical parcellation + anatomical segmentation into 93 classes.
pushd $fastsurfercnndir
cmd="$python eval.py --in_name $t1 --out_name $seg --order $order --network_sagittal_path $weights_sag --network_axial_path $weights_ax --network_coronal_path $weights_cor --batch_size $batch_size --simple_run $clean_seg $cuda"
echo $cmd |& tee -a $seg_log
$cmd |& tee -a $seg_log
if [ ${PIPESTATUS[0]} -ne 0 ] ; then exit 1 ; fi
popd

# ============= Running recon-surf (surfaces, thickness etc.) ===============
# use recon-surf to create surface models based on the FastSurferCNN segmentation.
pushd $reconsurfdir
cmd="./recon-surf.sh --sid $subject --sd $sd --t1 $t1 --seg $seg $mc $qspec $nofsaparc $fssurfreg $doParallel $dev --threads $threads --py $python"
$cmd
if [ ${PIPESTATUS[0]} -ne 0 ] ; then exit 1 ; fi
popd
########################################## End ########################################################
