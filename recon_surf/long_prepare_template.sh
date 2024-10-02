#!/bin/bash

# Copyright 2024 AI in Medical Imaging, German Center for Neurodegenerative Diseases (DZNE), Bonn
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

###################################################################################################
#
#
#  FastSurfer Longitudinal Base Template Creation
# 
# 
#  This script is part of the FastSurfer longitudinal pipeline. It runs a few pre-processing steps
#  to setup a base template for processing with a slighly modified version of run_fastsurfer.
# 
#  1. It obtains a brainmask on all time points for a specific subject
#  2. It uses mri_robust_template to co-register all time points into a mid-space
#     and sets up the base (subject template) directory with the aligned images the median image
#     and the forward and backward rigid transformations
# 
#  Following this script, one would process the median (subject template) image (-base) and then 
#  initialize each time point (-long) with information form this template for the surface module.
# 
#  Potential future things:
#   - Check if a bias field correction before the registration is helpful.
#   - Check if single view FastSurferVINN network is sufficient or if multi-view helps.
#   - Check if centroid based alignemnt of the segmentation helps for initializing robust_template.
#   - Add flag for adding a new time point to an existing base/template.
#
#  FreeSurfer requirements: 
#  mri_convert, mri_robust_template, mri_mask, mri_concatenate_lta, make_upright
#
###################################################################################################


# Set default values for arguments
if [[ -z "${BASH_SOURCE[0]}" ]]; then
    THIS_SCRIPT="$0"
else
    THIS_SCRIPT="${BASH_SOURCE[0]}"
fi
if [[ -z "$FASTSURFER_HOME" ]]
then
  FASTSURFER_HOME=$(cd "$(dirname "$THIS_SCRIPT")" &> /dev/null && pwd)
  echo "Setting ENV variable FASTSURFER_HOME to script directory ${FASTSURFER_HOME}. "
  echo "Change via environment to location of your choice if this is undesired (export FASTSURFER_HOME=/dir/to/FastSurfer)"
  export FASTSURFER_HOME
fi

# Paths
fastsurfercnndir="$FASTSURFER_HOME/FastSurferCNN"
reconsurfdir="$FASTSURFER_HOME/recon_surf"
# Maybe Todo: don't hard code checkpoints here:
checkpointsdir="$FASTSURFER_HOME/checkpoints"
weights_sag="$checkpointsdir/aparc_vinn_sagittal_v2.0.0.pkl"
weights_ax="$checkpointsdir/aparc_vinn_axial_v2.0.0.pkl"
weights_cor="$checkpointsdir/aparc_vinn_coronal_v2.0.0.pkl"

# TODO: these are fixed here, but should be passed via command line
batch_size=16
cuda=""
python="python3.10 -s" # avoid user-directory package inclusion
vox_size="min"

# setup variables that are actually passed
tid=""
sd="$SUBJECTS_DIR"
tpids=()
t1s=()

source "${reconsurfdir}/functions.sh"

function usage()
{
cat << EOF

Usage: long_prepare_template.sh --sid <sid> --sd <sdir> --t1 <t1_input> [OPTIONS]

long_prepare_template.sh takes a list of T1 full head image and creates:
     (i)   a template subject directory 
     (ii)  skull stripped and co-registerd images
     (iii) median image as template for this subject

FLAGS:

  --tid <templateID>        ID for subject template/base directory inside
                              \$SUBJECTS_DIR to be created"
  --t1s <T1_1> <T1_2> ..    T1 full head inputs for each time point (do not need
                              to be bias corrected). Requires ABSOLUTE paths!
  --tpids <tID1> >tID2> ..  IDs for future time points directories inside
                              \$SUBJECTS_DIR to be created later (during --long)
  --sd  <subjects_dir>      Output directory \$SUBJECTS_DIR (or pass via env var)
  -h --help                 Print Help

EOF
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
# make key lowercase
key=$(echo "$1" | tr '[:upper:]' '[:lower:]')

shift # past argument
case $key in
  --tid) tid="$1" ; shift ;;
  --tpids)
    while [[ $# -gt 0 ]] && [[ $1 != -* ]] 
    do
      tpids+=("$1")
      shift  # past value
    done
    ;;
  --t1s)
    while [[ $# -gt 0 ]] && [[ $1 != -* ]] 
    do
      t1s+=("$1")
      shift  # past value
    done
    ;;
  --sd) sd="$1" ; export SUBJECTS_DIR="$1" ; shift  ;;
  -h|--help) usage ; exit ;;
  *)    # unknown option
    # if not empty arguments, error & exit
    if [[ "$key" != "" ]] ; then echo "ERROR: Flag '$key' unrecognized." ;  exit 1 ; fi
    ;;
esac
done




#### CHECKS

if [ -z "$t1s" ]
 then
  echo "ERROR: must supply T1 inputs (full head) via --t1s"
  exit 1
fi

if [ -z "$tpids" ]
 then
  echo "ERROR: must supply time points ids via --tpids"
  exit 1
fi

if [ -z "$tid" ]
 then
  echo "ERROR: must supply subject template name via --tid"
  exit 1
fi

# check that t1s list is same length as tpids
if [ "${#tpids[@]}" -ne "${#t1s[@]}" ]
 then
  echo "ERROR: length of tpids must equal t1s"
  exit 1
fi

# check that SUBJECTS_DIR exists
if [[ -z "${sd}" ]]
then
  echo "ERROR: No subject directory defined via --sd. This is required!"
  exit 1;
fi
if [[ ! -d "${sd}" ]]
then
  echo "INFO: The subject directory did not exist, creating it now."
  if ! mkdir -p "$sd" ; then echo "ERROR: directory creation failed" ; exit 1; fi
fi
if [[ "$(stat -c "%u:%g" "$sd")" == "0:0" ]] && [[ "$(id -u)" != "0" ]] && [[ "$(stat -c "%a" "$sd" | tail -c 2)" -lt 6 ]]
then
  echo "ERROR: The subject directory ($sd) is owned by root and is not writable. FastSurfer cannot write results! "
  echo "This can happen if the directory is created by docker. Make sure to create the directory before invoking docker!"
  exit 1;
fi



# Setup Base/Template Directory and Log file
LF="$SUBJECTS_DIR/$tid/scripts/long_prepare_template.log"
mkdir -p "$(dirname "$LF")"


if [[ -f "$LF" ]]; then log_existed="true"
else log_existed="false"
fi

VERSION=$($python "$FASTSURFER_HOME/FastSurferCNN/version.py" "${version_args[@]}")
echo "Version: $VERSION" | tee -a "$LF"
echo "Log file for long_prepare_template" >> "$LF"
  { date 2>&1 ; echo "" ; } | tee -a "$LF"
echo "" | tee -a "$LF"
echo "export SUBJECTS_DIR=$SUBJECTS_DIR" | tee -a "$LF"
echo "cd `pwd`" | tee -a "$LF"
echo $0 ${inputargs[*]} | tee -a $LF
echo "" | tee -a "$LF"
cat $FREESURFER_HOME/build-stamp.txt 2>&1 | tee -a "$LF"
uname -a  2>&1 | tee -a "$LF"


### IF THE SCRIPT GETS TERMINATED, ADD A MESSAGE
trap "{ echo \"run_fastsurfer.sh terminated via signal at \$(date -R)!\" >> \"$LF\" ; }" SIGINT SIGTERM


# check that all t1s exist and that geo is the same (after log setup to keep this info in log file)
geodiff=0
for s in "${t1s[@]}"
do
  # check if input exist
  if [ ! -f $s ]
  then
    echo "ERROR: Input T1 $s does not exist!" | tee -a "$LF"
    exit 1
  fi
  # check if geometry differs across time
  if [ "$s" != "${t1s[0]}" ]
  then
    cmd="mri_diff --notallow-pix --notallow-geo $s ${t1s[0]}"
    RunIt "$cmd" $LF
    if [ $status ]
    then
      geodiff=1
    fi
  fi
done
if [ "$geodiff" == "1" ]
then
  {
    echo " "
    echo "*******************************************************************************" 
    echo "WARNING: Image parameters differ across time, maybe due to aquisition changes?"
    echo "         Consistent changes in, e.g., resolution can potentially bias a "
    echo "         longitudinal study! You can check image parameters by running mri_info"
    echo "         on each input image. Will continue in 10 seconds ..."
    echo "*******************************************************************************"
    echo " "
  } | tee -a "$LF"
  sleep 10
fi



################################### MASK INPUTS ###################################


# here we prepare images for base creation (image registration) below
# i)   either do nothing (maybe we try that)
# ii)  or compute brainmask (e.g. via single or multi view segmentation) <--- !!!
# iii) or even compute bias field removed images (one-shot bias field removal)

{
  echo " "
  echo "================== Creating Brainmask per TP ========================="
  echo " "
} | tee -a "$LF"

# copy inputs locally as nii.gz files (keep log where they came from)
extension=".nii.gz"
for ((i=0;i<${#tpids[@]};++i)); do
  #printf "%s with T1 %s\n" "${tpids[i]}" "${t1s[i]}"
  echo "${tpids[i]} with T1 ${t1s[i]}" | tee -a "$LF"
  mdir="$SUBJECTS_DIR/$tid/long-inputs/${tpids[i]}"
  mkdir -p $mdir
  # Import (copy) raw inputs (convert to extension format)
  cmd="mri_convert ${t1s[i]} $mdir/T1_raw${extension}"
  RunIt "$cmd" $LF
  # conform
  conform="$mdir/T1_orig${extension}"
  #cmd="mri_convert -c ${t1s[i]} $conform"
  cmd="$python $fastsurfercnndir/data_loader/conform.py -i ${t1s[i]} -o $conform --vox_size $vox_size --dtype any --verbose"
  RunIt "$cmd" $LF
  # segment conform image
  # with the goal to create brainmask for mainly registration
  # (here we can only use one network)
  #seg="$mdir/aparc+aseg.orig${extension}" #currently eval cannot output nii.gz!!!!!
  seg="$mdir/aparc+aseg.orig.mgz"
  order=1
  clean_seg=""
  #pushd $fscnndir
  cmd="$python $fastsurfercnndir/eval.py --in_name $conform --out_name $seg --order $order \
         --network_sagittal_path $weights_sag \
         --network_axial_path $weights_ax \
         --network_coronal_path $weights_cor \
         --batch_size $batch_size --simple_run $clean_seg $cuda"
  RunIt "$cmd" $LF
  #popd    

  echo " " | tee -a $LF
  echo "============= Creating aseg.auto_noCCseg (map aparc labels back) ===============" | tee -a $LF
  echo " " | tee -a $LF
  # reduce labels to aseg, then create mask (dilate 5, erode 4, largest component), also mask aseg to remove outliers
  # output will be uchar (else mri_cc will fail below)
  mask="$mdir/mask${extension}"
  aseg="$mdir/aseg.auto_noCCseg${extension}"
  mask="$mdir/mask.mgz"
  aseg="$mdir/aseg.auto_noCCseg.mgz"
  # not sure this works with nifti !!! no cannot output nifti automatically:
  cmd="$python $reconsurfdir/reduce_to_aseg.py -i $seg -o $aseg --outmask $mask"
  RunIt "$cmd" $LF
  # mask is binary, we need to use on orig
  cmd="mri_mask $conform $mask $mdir/brainmask${extension}"
  RunIt "$cmd" $LF
done

# skip intensity normalization or bias field removal for now




#################################### CO-REGISTER INPUTS ############################################

{
  echo " "
  echo "================== Co-registering TPs ========================="
  echo " "
} | tee -a "$LF"

# create a file with all time points names
# this cannot be "base-tps" else recon-surf (and inside recon-all) will fail
BaseSubjsListFname="$SUBJECTS_DIR/$tid/base-tps.fastsurfer"
rm -f ${BaseSubjsListFname}
mkdir -p $SUBJECTS_DIR/$tid/mri/transforms
subjInVols=()
normInVols=()
ltaXforms=()

for s in "${tpids[@]}"
do
  echo $s
  echo "${s}" >> ${BaseSubjsListFname}
  mdir="$SUBJECTS_DIR/$tid/long-inputs/${s}"
  invol="$mdir/T1_orig${extension}"
  subjInVols+=($invol)
  normvol="$mdir/brainmask${extension}"
  normInVols+=($normvol)
  ltaname=${s}_to_${tid}.lta
  ltaXforms+=(${SUBJECTS_DIR}/$tid/mri/transforms/${ltaname})
done


if [ ${#tpids[@]} == 1 ]
then
  # if only a single time point, create fake 'base' by making the image upright
  # this assures that also subjects with a single time point get processes as the other
  # subjects in the longitudinal stream

  # 1. make the norm upright (base space)
  cmd="make_upright ${normInVols[0]} \
       ${SUBJECTS_DIR}/$tid/mri/norm_template.mgz ${ltaXforms[0]}"
  #echo "$cmd"
  RunIt "$cmd" $LF

  # 2. create the upright orig volume
  cmd="mri_convert -rt cubic \
       -at ${ltaXforms[0]} ${subjInVols[0]} ${SUBJECTS_DIR}/$tid/mri/orig.mgz"
  #echo "$cmd"
  RunIt "$cmd" $LF

else #more than 1 time point:

  robust_template_avg_arg=1  # median

  # create the 'mean/median' norm volume:
  cmd="mri_robust_template --mov ${normInVols[@]}"
  cmd="$cmd --lta ${ltaXforms[@]}"
  cmd="$cmd --template ${SUBJECTS_DIR}/$tid/mri/norm_template.mgz"
  cmd="$cmd --average ${robust_template_avg_arg}"
  cmd="$cmd --sat 4.685"
  #echo "$cmd"
  RunIt "$cmd" $LF

  # create the 'mean/median' input (orig) volume:
  cmd="mri_robust_template --mov ${subjInVols[@]}"
  cmd="$cmd --average ${robust_template_avg_arg}"
  cmd="$cmd --ixforms ${ltaXforms[@]}"
  cmd="$cmd --noit"
  t1=${SUBJECTS_DIR}/$tid/mri/orig.mgz
  cmd="$cmd --template $t1"
  #echo "$cmd"
  RunIt "$cmd" $LF

fi # more than one time point

# now create the inverse transforms
#cd $subjdir/mri/transforms > /dev/null
#$PWD |& tee -a $LF
odir=${SUBJECTS_DIR}/$tid/mri/transforms
for s in "${tpids[@]}"
do
  cmd="mri_concatenate_lta -invert1"
  cmd="$cmd $odir/${s}_to_${tid}.lta"
  cmd="$cmd identity.nofile"
  cmd="$cmd $odir/${tid}_to_${s}.lta"
  #echo "$cmd"
  RunIt "$cmd" $LF
done

# for consistency with FreeSurfer longitudinal strea, one could
# map all cross sectional mask to base space and compute the union. 
# currently we simply re-create the mask in base during segmentation.
