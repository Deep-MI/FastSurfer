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
if [ -z "$FASTSURFER_HOME" ]
then
  echo "Setting ENV variable FASTSURFER_HOME to current working directory ${PWD}. "
  echo "Change via enviroment to location of your choice if this is undesired (export FASTSURFER_HOME=/dir/to/FastSurfer)"
  export FASTSURFER_HOME=${PWD}
fi
fastsurfercnndir="$FASTSURFER_HOME/FastSurferCNN"
reconsurfdir="$FASTSURFER_HOME/recon_surf"

# Regular flags defaults
subject=""
t1=""
seg=""
conformed_name=""
seg_log=""
weights_sag="$FASTSURFER_HOME/checkpoints/Sagittal_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl"
weights_ax="$FASTSURFER_HOME/checkpoints/Axial_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl"
weights_cor="$FASTSURFER_HOME/checkpoints/Coronal_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl"
clean_seg=""
viewagg="check"
cuda=""
batch_size="8"
order="1"
seg_only="0"
seg_cc=""
vol_segstats=""
surf_only="0"
fstess=""
fsqsphere=""
fsaparc=""
fssurfreg=""
doParallel=""
threads="1"
python="python3.6"

# Dev flags defaults
vcheck=""
vfst1=""

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
    echo -e "\t--seg <segmentation_input>             Name of intermediate DL-based segmentation file (similar to aparc+aseg). When using FastSurfer, this segmentation is already conformed, since inference is always based on a conformed image. Requires an ABSOLUTE Path! Default location: \$SUBJECTS_DIR/\$sid/mri/aparc.DKTatlas+aseg.deep.mgz."
    echo -e "\t--conformed_name <conformed_image_name>             Name of the file in which the conformed input image will be saved. Default location: \$SUBJECTS_DIR/\$sid/mri/orig.mgz."
    echo -e "\t--seg_log <segmentation_log>           Log-file for the segmentation (FastSurferCNN). Default: \$SUBJECTS_DIR/\$sid/scripts/deep-seg.log"
    echo -e "\t--weights_sag <weights_sagittal>       Pretrained weights of sagittal network. Default: \$FASTSURFER_HOME/checkpoints/Sagittal_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl"
    echo -e "\t--weights_ax <weights_axial>           Pretrained weights of axial network. Default: \$FASTSURFER_HOME/checkpoints/Axial_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl"
    echo -e "\t--weights_cor <weights_coronal>        Pretrained weights of coronal network. Default: \$FASTSURFER_HOME/checkpoints/Coronal_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl"
    echo -e "\t--clean_seg <clean_segmentation>       Flag to clean up FastSurferCNN segmentation"
    echo -e "\t--run_viewagg_on <check,gpu,cpu>       Define where the view aggregation should be run on. By default, the program checks if you have enough memory \
                                                      to run the view aggregation on the gpu. The total memory is considered for this decision. If this fails, or \
                                                      you actively overwrote the check with setting > --run_viewagg_on cpu <, view agg is run on the cpu. \
                                                      Equivalently, if you define > --run_viewagg_on gpu <, view agg will be run on the gpu (no memory check will be done)."
    echo -e "\t--no_cuda <disable_cuda>               Flag to disable CUDA usage in FastSurferCNN (no GPU usage, inference on CPU)"
    echo -e "\t--batch <batch_size>                   Batch size for inference. Default: 8."
    echo -e "\t--order <order_of_interpolation>       Order of interpolation for mri_convert T1 before segmentation (0=nearest,1=linear(default),2=quadratic,3=cubic)"
    echo -e "\t--seg_only                             Run only FastSurferCNN (generate segmentation, do not run surface pipeline)"
    echo -e "\t--seg_with_cc_only                     Run FastSurferCNN (generate segmentation) and recon_surf until corpus callosum (CC) is added in (no surface models will be created in this case!)"
    echo -e "\t--surf_only                            Run surface pipeline only. The segmentation input has to exist already in this case."
    echo -e "\t--vol_segstats                         Additionally return volume-based aparc.DKTatlas+aseg statistics for DL-based segmentation (does not require surfaces). Can be used in combination with --seg_only in which case recon-surf only runs till CC is added (akin to --seg_with_cc_only)."
    echo -e "\t--fstess                               Switch on mri_tesselate for surface creation (default: mri_mc)"
    echo -e "\t--fsqsphere                            Use FreeSurfer iterative inflation for qsphere (default: spectral spherical projection)"
    echo -e "\t--fsaparc                              Additionally create FS aparc segmentations and ribbon. Skipped by default (--> DL prediction is used which is faster, and usually these mapped ones are fine)"
    echo -e "\t--surfreg                              Run Surface registration with FreeSurfer (for cross-subject correspondence)"
    echo -e "\t--parallel                             Run both hemispheres in parallel"
    echo -e "\t--threads <int>                        Set openMP and ITK threads to <int>"
    echo -e "\t--py <python_cmd>                      Command for python, default 'python3.6'"
    echo -e "\t-h --help                              Print Help"
    echo ""
    echo "Dev Flags"
    echo -e "\t--ignore_fs_version                    Switch on to avoid check for FreeSurfer version. Program will otherwise terminate if v6.0 is not sourced. Can be used for testing dev versions."
    echo -e "\t--no_fs_T1                             Do not generate T1.mgz (normalized nu.mgz included in standard FreeSurfer output) and create brainmask.mgz directly from norm.mgz instead. Saves approx. 1:30 min."
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
    if [ -f "$2" ]; then
        export FS_LICENSE="$2"
    else
        echo "Provided FreeSurfer license file $2 could not be found. Make sure to provide the full path and name. Exiting..."
        exit 1;
    fi
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
    --conformed_name)
    conformed_name="$2"
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
    --run_viewagg_on)
    viewagg="$2"
    shift # past argument
    shift # past value
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
    --seg_only)
    seg_only="1"
    shift # past argument
    ;;
    --seg_with_cc_only)
    seg_cc="--seg_with_cc_only"
    shift # past argument
    ;;
    --surf_only)
    surf_only="1"
    shift # past argument
    ;;
    --vol_segstats)
    vol_segstats="--vol_segstats"
    shift # past argument
    ;;
    --fstess)
    fstess="--fstess"
    shift # past argument
    ;;
    --fsqsphere)
    fsqsphere="--fsqsphere"
    shift # past argument
    ;;
    --fsaparc)
    fsaparc="--fsaparc"
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
    --py)
    python="$2"
    shift # past argument
    shift # past value
    ;;
    --ignore_fs_version)
    vcheck="--ignore_fs_version"
    shift # past argument
    ;;
    --no_fs_T1 )
    vfst1="--no_fs_T1"
    shift # past argument
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
if [ -z "$t1" ] || [ ! -f "$t1" ]
  then
    echo "ERROR: T1 image ($t1) could not be found. Must supply an existing T1 input (conformed, full head) via --t1 (absolute path and name)."
    # needed to create orig.mgz and to get file name. This will eventually be changed.
    exit 1;
fi

if [ -z "$subject" ]
 then
    echo "ERROR: must supply subject name via --sid"
    exit 1;
fi

if [ -z "$seg" ]
  then
    seg="${sd}/${subject}/mri/aparc.DKTatlas+aseg.deep.mgz"
fi

if [ -z "$conformed_name" ]
  then
    conformed_name="${sd}/${subject}/mri/orig.mgz"
fi

if [ -z "$seg_log" ]
 then
    seg_log="${sd}/${subject}/scripts/deep-seg.log"
fi

if [ -z "$PYTHONUNBUFFERED" ]
then
  export PYTHONUNBUFFERED=0
fi

if [ "${seg: -3}" != "${conformed_name: -3}" ]
  then
    echo "ERROR: Specified segmentation output and conformed image output do not have same file type."
    echo "You passed --seg ${seg} and --conformed_name ${conformed_name}."
    echo "Make sure these have the same file-format and adjust the names passed to the flags accordingly!"
    exit 1;
fi

if [ "$surf_only" == "1" ] && [ ! -f "$seg" ]
  then
    echo "ERROR: To run the surface pipeline only, whole brain segmentation must already exist."
    echo "You passed --surf_only but the whole-brain segmentation ($seg) could not be found."
    echo "If the segmentation is not saved in the default location (\$SUBJECTS_DIR/\$SID/mri/aparc.DKTatlas+aseg.deep.mgz), specify the absolute path and name via --seg"
    exit 1;
fi

if [ "$surf_only" == "1" ] && [ "$seg_only" == "1" ]
  then
    echo "ERROR: You specified both --surf_only and --seg_only. Therefore neither part of the pipeline will be run."
    echo "To run the whole FastSurfer pipeline, omit both flags."
    exit 1;
fi

if [ "$seg_only" == "1" ] && [ ! -z "$vol_segstats" ]
  then
    seg_cc="--seg_with_cc_only"
    seg_only=0
    echo "You requested segstats without running the surface pipeline. In this case, recon-surf will"
    echo "run until the corpus callsoum is added to the segmentation and the norm.mgz is generated "
    echo "(needed for partial volume correction). "
    echo "The stats will be stored as (\$SUBJECTS_DIR/\$SID/stats/aparc.DKTatlas+aseg.deep.volume.stats). "
fi
########################################## START ########################################################


if [ "$surf_only" == "0" ]; then
  # "============= Running FastSurferCNN (Creating Segmentation aparc.DKTatlas.aseg.mgz) ==============="
  # use FastSurferCNN to create cortical parcellation + anatomical segmentation into 95 classes.
  mkdir -p "$(dirname "$seg_log")"
  echo "Log file for fastsurfercnn eval.py" > $seg_log
  date  |& tee -a $seg_log
  echo "" |& tee -a $seg_log

  pushd $fastsurfercnndir
  cmd="$python eval.py --in_name $t1 --out_name $seg --conformed_name $conformed_name --order $order --network_sagittal_path $weights_sag --network_axial_path $weights_ax --network_coronal_path $weights_cor --batch_size $batch_size --simple_run $clean_seg --run_viewagg_on $viewagg $cuda"
  echo $cmd |& tee -a $seg_log
  $cmd |& tee -a $seg_log
  if [ ${PIPESTATUS[0]} -ne 0 ] ; then exit 1 ; fi
  popd
fi

if [ "$seg_only" == "0" ]; then
  # ============= Running recon-surf (surfaces, thickness etc.) ===============
  # use recon-surf to create surface models based on the FastSurferCNN segmentation.
  pushd $reconsurfdir
  cmd="./recon-surf.sh --sid $subject --sd $sd --t1 $conformed_name --seg $seg $seg_cc $vol_segstats $fstess $fsqsphere $fsaparc $fssurfreg $doParallel --threads $threads --py $python $vcheck $vfst1"
  $cmd
  if [ ${PIPESTATUS[0]} -ne 0 ] ; then exit 1 ; fi
  popd
fi

########################################## End ########################################################
