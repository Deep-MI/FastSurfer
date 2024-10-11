#!/bin/bash

# Copyright 2023 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
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
FS_VERSION_SUPPORT="7.4.1"

# Regular flags default
subject=""; # Subject name
python="python3.10" # python version
DoParallel=0 # if 1, run hemispheres in parallel
threads="1" # number of threads to use for running FastSurfer
allow_root=""         # flag for allowing execution as root user

# Dev flags default
check_version=1.      # Check for supported FreeSurfer version (terminate if not detected)

if [ -z "$FASTSURFER_HOME" ]
then
  binpath="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )/"
else
  binpath="$FASTSURFER_HOME/recon_surf/"
fi


# check bash version > 4
function version { echo "$@" | awk -F. '{ printf("%d%03d%03d%03d\n", $1,$2,$3,$4); }'; }
if [ $(version ${BASH_VERSION}) -lt $(version "4.0.0") ]; then
    echo "bash ${BASH_VERSION} is too old. Should be newer than 4.0, please upgrade!"
    exit 1
fi


function usage()
{
cat << EOF

Usage: recon-surfreg.sh --sid <sid> --sd <sdir> [OPTIONS]

recon-surfreg.sh creates the ?h.sphere and ?h.sphere.reg from an existing
subject directory, if this step was skipped in recon-surf.sh with --no_surfreg

FLAGS:
  --sid <subjectID>       Subject ID to create directory inside \$SUBJECTS_DIR
  --sd  <subjects_dir>    Output directory \$SUBJECTS_DIR (or pass via env var)
  --parallel              Run both hemispheres in parallel
  --threads <int>         Set openMP and ITK threads to <int>
  --py <python_cmd>       Command for python, default $python
  --fs_license <license>  Path to FreeSurfer license key file. Register at
                            https://surfer.nmr.mgh.harvard.edu/registration.html
                            for free to obtain it if you do not have FreeSurfer
			    installed already
  -h --help               Print Help

Dev Flags:
  --ignore_fs_version     Switch on to avoid check for FreeSurfer version.
                            Program will otherwise terminate if $FS_VERSION_SUPPORT is
                            not sourced. Can be used for testing dev versions.
  --allow_root            Allow execution as root user

REFERENCES:

If you use this for research publications, please cite:

Henschel L, Conjeti S, Estrada S, Diers K, Fischl B, Reuter M, FastSurfer - A
 fast and accurate deep learning based neuroimaging pipeline, NeuroImage 219
 (2020), 117012. https://doi.org/10.1016/j.neuroimage.2020.117012

Henschel L*, Kuegler D*, Reuter M. (*co-first). FastSurferVINN: Building
 Resolution-Independence into Deep Learning Segmentation Methods - A Solution
 for HighRes Brain MRI. NeuroImage 251 (2022), 118933.
 http://dx.doi.org/10.1016/j.neuroimage.2022.118933

EOF

}

# Load RunIt, timecmd, RunBatchJobs from functions.sh
source "$binpath/functions.sh"

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
    --sid)
    subject="$2"
    shift # past argument
    shift # past value
    ;;
    --sd)
    export SUBJECTS_DIR="$2"
    shift # past argument
    shift # past value
    ;;
    --parallel)
    DoParallel=1
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
    --ignore_fs_version)
    check_version=0
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
echo " "
echo "sid $subject"
echo " "


# Warning if run as root user
check_allow_root

if [ "$subject" == "subject" ]
then
  echo "Subject ID cannot be \"subject\", please choose a different sid"
  # Explanation, see https://github.com/Deep-MI/FastSurfer/issues/186
  # this is a bug in FreeSurfer's argparse when calling "mri_brainvol_stats subject"
  exit 1
fi

if [ -z "$SUBJECTS_DIR" ]
then
  echo "\$SUBJECTS_DIR not set. Either set it via the shell prior to running recon-surfreg.sh or supply it via the --sd flag."
  exit 1
fi

if [ -z "$FREESURFER_HOME" ]
then
  echo "Did not find \$FREESURFER_HOME. A working version of FreeSurfer $FS_VERSION_SUPPORT is needed to run recon-surfreg locally."
  echo "Make sure to export and source FreeSurfer before running recon-surfreg.sh: "
  echo "export FREESURFER_HOME=/path/to/your/local/fs$FS_VERSION_SUPPORT"
  echo "source \$FREESURFER_HOME/SetUpFreeSurfer.sh"
  exit 1
fi
# needed in FS72 due to a bug in recon-all --fill using FREESURFER instead of FREESURFER_HOME
export FREESURFER=$FREESURFER_HOME   

if [ "$check_version" == "1" ]
then
  if grep -q -v ${FS_VERSION_SUPPORT} $FREESURFER_HOME/build-stamp.txt
  then
    echo "ERROR: You are trying to run recon-surfreg with FreeSurfer version $(cat $FREESURFER_HOME/build-stamp.txt)."
    echo "We are currently supporting only FreeSurfer $FS_VERSION_SUPPORT "
    echo "Therefore, make sure to export and source the correct FreeSurfer version before running recon-surfreg.sh: "
    echo "export FREESURFER_HOME=/path/to/your/local/fs$FS_VERSION_SUPPORT"
    echo "source \$FREESURFER_HOME/SetUpFreeSurfer.sh"
    exit 1
  fi
fi

if [ -z "$PYTHONUNBUFFERED" ]
then
  export PYTHONUNBUFFERED=0
fi


if [ -z "$subject" ]
then
  echo "ERROR: must supply subject name via --sid"
  exit 1
fi

# set threads for openMP and itk
# if OMP_NUM_THREADS is not set and available resources are too vast, mc will fail with segmentation fault!
# Therefore we set it to 1 as default above, if nothing is specified.
fsthreads=""
export OMP_NUM_THREADS=$threads
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$threads
if [ "$threads" -gt "1" ]
then
  fsthreads="-threads $threads -itkthreads $threads"
fi

if [ $(echo -n "${SUBJECTS_DIR}/${subject}" | wc -m) -gt 185 ]
then
  echo "ERROR: subject directory path is very long."
  echo "This is known to cause errors due to some commands run by freesurfer versions built for Ubuntu."
  echo "--sd + --sid should be less than 185 characters long."
  exit 1
fi

for hemi in rh lh; do

  # Check if running on an existing registry
  if [ -f "$SUBJECTS_DIR/$subject/surf/$hemi.sphere" ] || [ -f "$SUBJECTS_DIR/$subject/surf/$hemi.sphere.reg" ] || [ -f "$SUBJECTS_DIR/$subject/surf/$hemi.angles.txt" ]; then
    echo "ERROR: running on top of an existing registry!"
    echo "The output directory must not contain a surfreg from a previous invocation of recon-surf."
    exit 1
  fi

done

# collect info
StartTime=`date`;
tSecStart=`date '+%s'`;
year=`date +%Y`
month=`date +%m`
day=`date +%d`
hour=`date +%H`
min=`date +%M`


sdir=$SUBJECTS_DIR/$subject/surf
ldir=$SUBJECTS_DIR/$subject/label


# Set up log file
DoneFile=$SUBJECTS_DIR/$subject/scripts/recon-surfreg.done
if [ $DoneFile != /dev/null ] ; then  rm -f $DoneFile ; fi
LF=$SUBJECTS_DIR/$subject/scripts/recon-surfreg.log
if [ $LF != /dev/null ] ; then  rm -f $LF ; fi
echo "Log file for recon-surfreg.sh" >> $LF
date  2>&1 | tee -a $LF
echo "" | tee -a $LF
echo "export SUBJECTS_DIR=$SUBJECTS_DIR" | tee -a $LF
echo "cd `pwd`"  | tee -a $LF
echo $0 ${inputargs[*]} | tee -a $LF
echo "" | tee -a $LF
cat $FREESURFER_HOME/build-stamp.txt 2>&1 | tee -a $LF
echo $VERSION | tee -a $LF
uname -a  2>&1 | tee -a $LF


# Print parallelization parameters
echo " " | tee -a $LF
if [ "$DoParallel" == "1" ]
then
  echo " RUNNING both hemis in PARALLEL " | tee -a $LF
else
  echo " RUNNING both hemis SEQUENTIALLY " | tee -a $LF
fi
echo " RUNNING $OMP_NUM_THREADS number of OMP THREADS " | tee -a $LF
echo " RUNNING $ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS number of ITK THREADS " | tee -a $LF
echo " " | tee -a $LF


#if false; then

########################################## START ########################################################


# ================================================== SURFACES ==========================================================


CMDFS=""

for hemi in lh rh; do

  CMDF="$SUBJECTS_DIR/$subject/scripts/$hemi.processing.cmdf"
  CMDFS="$CMDFS $CMDF"
  rm -rf $CMDF

  echo "echo \" \"" | tee -a $CMDF
  echo "echo \"============ Creating surfaces $hemi - FS sphere, surfreg ===============\"" | tee -a $CMDF
  echo "echo \" \"" | tee -a $CMDF

  # Surface registration for cross-subject correspondence (registration to fsaverage)
  cmd="recon-all -subject $subject -hemi $hemi -sphere -no-isrunning $fsthreads"
  RunIt "$cmd" $LF "$CMDF"

  # (mr) FIX: sometimes FreeSurfer Sphere Reg. fails and moves pre and post central
  # one gyrus too far posterior, FastSurferCNN's image-based segmentation does not
  # seem to do this, so we initialize the spherical registration with the better
  # cortical segmentation from FastSurferCNN, this replaces recon-al -surfreg
  # 1. get alpha, beta, gamma for global alignment (rotation) based on aseg centers
  # (note the former fix, initializing with pre-central label, is not working in FS7.2
  # as they broke the label initialization in mris_register)
  cmd="$python ${binpath}/rotate_sphere.py \
       --srcsphere $sdir/${hemi}.sphere \
       --srcaparc $ldir/$hemi.aparc.DKTatlas.mapped.annot \
       --trgsphere $FREESURFER_HOME/subjects/fsaverage/surf/${hemi}.sphere \
       --trgaparc $FREESURFER_HOME/subjects/fsaverage/label/${hemi}.aparc.annot \
       --out $sdir/${hemi}.angles.txt"
  RunIt "$cmd" $LF "$CMDF"
  # 2. use global rotation as initialization to non-linear registration:
  cmd="mris_register -curv -norot -rotate \`cat $sdir/${hemi}.angles.txt\` \
       $sdir/${hemi}.sphere \
       $FREESURFER_HOME/average/${hemi}.folding.atlas.acfb40.noaparc.i12.2016-08-02.tif \
       $sdir/${hemi}.sphere.reg"
  RunIt "$cmd" $LF "$CMDF"
  # command to generate new aparc to check if registration was OK
  # run only for debugging
  #cmd="mris_ca_label -l $SUBJECTS_DIR/$subject/label/${hemi}.cortex.label \
  #     -aseg $SUBJECTS_DIR/$subject/mri/aseg.presurf.mgz \
  #     -seed 1234 $subject $hemi $SUBJECTS_DIR/$subject/surf/${hemi}.sphere.reg \
  #     $SUBJECTS_DIR/$subject/label/${hemi}.aparc.DKTatlas-guided.annot"

  if [ "$DoParallel" == "0" ] ; then
      echo " " | tee -a $LF
      echo " RUNNING $hemi sequentially ... " | tee -a $LF
      echo " " | tee -a $LF
    chmod u+x $CMDF
    RunIt "$CMDF" $LF
  fi

done  # hemi loop ----------------------------------



if [ "$DoParallel" == 1 ] ; then
    echo " " | tee -a $LF
    echo " RUNNING HEMIs in PARALLEL !!! " | tee -a $LF
    echo " " | tee -a $LF
    RunBatchJobs $LF $CMDFS
fi


echo " " | tee -a $LF
echo "================= DONE =========================================================" | tee -a $LF
echo " " | tee -a $LF

# Collect info
EndTime=`date`
tSecEnd=`date '+%s'`
tRunHours=`echo \($tSecEnd - $tSecStart\)/3600|bc -l`
tRunHours=`printf %6.3f $tRunHours`

echo "Started at $StartTime " | tee -a $LF
echo "Ended   at $EndTime" | tee -a $LF
echo "#@#%# recon-surfreg-run-time-hours $tRunHours" | tee -a $LF

# Create the Done File
echo "------------------------------" > $DoneFile
echo "SUBJECT $subject"           >> $DoneFile
echo "START_TIME $StartTime"      >> $DoneFile
echo "END_TIME $EndTime"          >> $DoneFile
echo "RUNTIME_HOURS $tRunHours"   >> $DoneFile
echo "USER `id -un`"              >> $DoneFile 2> /dev/null
echo "HOST `hostname`"            >> $DoneFile
echo "PROCESSOR `uname -m`"       >> $DoneFile
echo "OS `uname -s`"              >> $DoneFile
echo "UNAME `uname -a`"           >> $DoneFile
echo "VERSION $VERSION"           >> $DoneFile
echo "CMDPATH $0"                 >> $DoneFile
echo "CMDARGS ${inputargs[*]}"    >> $DoneFile

echo "recon-surfreg.sh $subject finished without error at `date`"  | tee -a $LF

cmd="$python ${binpath}utils/extract_recon_surf_time_info.py -i $LF -o $SUBJECTS_DIR/$subject/scripts/recon-surfreg_times.yaml"
RunIt "$cmd" "/dev/null"
