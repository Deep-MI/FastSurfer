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
FS_VERSION_SUPPORT="stable-pub-v6"
timecmd="fs_time"

# Regular flags default
t1=""; # Path and name of T1 input
seg=""; # Path and name of segmentation
subject=""; # Subject name
seg_cc=0; # if 1, run pipeline only till corpus callosum is added (no surfaces will be created)
vol_segstats=0; # if 1, return volume-based aparc.DKTatlas+aseg stats based on dl-prediction
fstess=0;       # run mri_tesselate (FS way), if 0 = run mri_mc
fsqsphere=0;    # run inflate1 and qsphere (FSway), if 0 run spectral projection
fsaparc=0;	# run FS aparc (and cortical ribbon), if 0 map aparc from seg input
fssurfreg=0;  # run FS surface registration to fsaverage, if 0 omit this step
python="python3.6" # python version
DoParallel=0 # if 1, run hemispheres in parallel
threads="1" # number of threads to use for running FastSurfer

# Dev flags default
check_version=1; # Run version check for FreeSurfer (terminate if anything but v6.0 is detected)
get_t1=1; # Generate T1.mgz from nu.mgz and brainmask from it (default)

if [ -z "$FASTSURFER_HOME" ]
then
  binpath="./"
else
  binpath="$FASTSURFER_HOME/recon_surf/"
fi

function usage()
{
    echo ""
    echo "recon-surf.sh takes a segmentation and T1 full head image and creates surfaces, thickness etc as a FS subject dir"
    echo ""
    echo "./recon-surf.sh"
    echo -e "\t--sid <subjectID>             Subject ID for directory inside \$SUBJECTS_DIR to be created"
    echo -e "\t--sd  <subjects_dir>          Output directory \$SUBJECTS_DIR (pass via environment or here)"
    echo -e "\t--t1  <T1_input>              T1 full head input (not bias corrected). This must be a conformed image (dimensions: 256x256x256, voxel size: 1x1x1,
    LIA orientation, and data type UCHAR). Images can be conformed using FastSurferCNN's conform.py script (usage example: python3 FastSurferCNN/data_loader/conform.py -i <T1_input> -o <conformed_T1_output>)."
    echo -e "\t--seg <segmentation_input>    Name of intermediate DL-based segmentation file (similar to aparc+aseg). This must be conformed (dimensions: 256x256x256, voxel size: 1x1x1, LIA orientation). FastSurferCNN's segmentations are conformed by default; please ensure that segmentations produced otherwise are conformed. Requires an ABSOLUTE Path! Default location: \$SUBJECTS_DIR/\$sid/mri/aparc.DKTatlas+aseg.deep.mgz."
    echo -e "\t--seg_with_cc_only            Run recon_surf until corpus callosum is added in (no surface models will be created in this case!)"
    echo -e "\t--vol_segstats                Additionally return volume-based aparc.DKTatlas+aseg statistics for DL-based segmentation (does not require surfaces)."
    echo -e "\t--fstess                      Switch on mri_tesselate for surface creation (default: mri_mc)"
    echo -e "\t--fsqsphere                   Use FreeSurfer iterative inflation for qsphere (default: spectral spherical projection)"
    echo -e "\t--fsaparc                     Additionally create FS aparc segmentations and ribbon. Skipped by default (--> DL prediction is used which is faster, and usually these mapped ones are fine)"
    echo -e "\t--surfreg                     Run Surface registration with FreeSurfer (for cross-subject correspondence)"
    echo -e "\t--parallel                    Run both hemispheres in parallel"
    echo -e "\t--threads <int>               Set openMP and ITK threads to <int>"
    echo -e "\t--py <python_cmd>             Command for python, default 'python36'"
    echo -e "\t--fs_license <freesurfer_license_file>  Path to FreeSurfer license key file. Register (for free) at https://surfer.nmr.mgh.harvard.edu/registration.html to obtain it if you do not have FreeSurfer installed so far."
    echo -e "\t-h --help                     Print Help"
    echo ""
    echo "Dev Flags"
    echo -e "\t--ignore_fs_version           Switch on to avoid check for FreeSurfer version. Program will otherwise terminate if v6.0 is not sourced. Can be used for testing dev versions."
    echo -e "\t--no_fs_T1                    Do not generate T1.mgz (normalized nu.mgz included in standard FreeSurfer output) and create brainmask.mgz directly from norm.mgz instead. Saves approx. 1:30 min."
    echo ""
}



function RunIt()
{
# parameters
# $1 : cmd  (command to run)
# $2 : LF   (log file)
# $3 : CMDF (command file) optional
# if CMDF is passed, then LF is ignored and cmd is echoed into CMDF and not run
  cmd=$1
  LF=$2
  if [[ $# -eq 3 ]]
  then
    CMDF=$3
    echo "echo \"$cmd\" " |& tee -a $CMDF
    echo "$timecmd $cmd " |& tee -a $CMDF
    echo "if [ \${PIPESTATUS[0]} -ne 0 ] ; then exit 1 ; fi" >> $CMDF
  else
    echo $cmd |& tee -a $LF
    $timecmd $cmd |& tee -a $LF
    if [ ${PIPESTATUS[0]} -ne 0 ] ; then exit 1 ; fi
  fi
}

function RunBatchJobs()
{
# parameters
# $1 : LF
# $2 ... : CMDFS
  LOG_FILE=$1
  # launch jobs found in command files (shift past first logfile arg).
  # job output goes to a logfile named after the command file, which
  # later gets appended to LOG_FILE

  echo
  echo "RunBatchJobs: Logfile: $LOG_FILE"

  PIDS=()
  LOGS=()
  shift
  for cmdf in $*; do
    echo "RunBatchJobs: CMDF: $cmdf"
    chmod u+x $cmdf
    JOB="$cmdf"
    LOG=$cmdf.log
    echo "" >& $LOG
    echo " $JOB" >> $LOG
    echo "" >> $LOG
    exec $JOB >> $LOG 2>&1 &
    PIDS=(${PIDS[@]} $!)
    LOGS=(${LOGS[@]} $LOG)

  done
  # wait till all processes have finished
  PIDS_STATUS=()
  for pid in "${PIDS[@]}"; do
    echo "Waiting for PID $pid of (${PIDS[@]}) to complete..."
    wait $pid
    PIDS_STATUS=(${PIDS_STATUS[@]} $?)
  done
  # now append their logs to the main log file
  for log in "${LOGS[@]}"
  do
    cat $log >> $LOG_FILE
    rm -f $log
  done
  echo "PIDs (${PIDS[@]}) completed and logs appended."
  # and check for failures
  for pid_status in "${PIDS_STATUS[@]}"
  do
    if [ "$pid_status" != "0" ] ; then
      exit 1
    fi
  done
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
    --seg_with_cc_only)
    seg_cc=1
    shift # past argument
    ;;
    --vol_segstats)
    vol_segstats=1
    shift # past argument
    ;;
    --fstess)
    fstess=1
    shift # past argument
    ;;
    --fsqsphere)
    fsqsphere=1
    shift # past argument
    ;;
    --fsaparc)
    fsaparc=1
    shift # past argument
    ;;
    --surfreg)
    fssurfreg=1
    shift # past argument
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
    --no_fs_T1 )
    get_t1=0
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
echo
echo sid $subject
echo T1  $t1
echo seg $seg
echo

if [ -z "$SUBJECTS_DIR" ]
then
  echo "\$SUBJECTS_DIR not set. Either set it via the shell prior to running recon_surf.sh or supply it via the --sd flag."
  exit 1;
fi

if [ -z "$FREESURFER_HOME" ]
then
  echo "Did not find \$FREESURFER_HOME. A working version of FreeSurfer v6.0 is needed to run recon-surf locally."
  echo "Make sure to export and source FreeSurfer before running recon-surf.sh: "
  echo "export FREESURFER_HOME=/path/to/your/local/fs60"
  echo "source \$FREESURFER_HOME/SetUpFreeSurfer.sh"
  exit 1;
fi

if [ "$check_version" == "1" ]
then
  if grep -q -v ${FS_VERSION_SUPPORT} $FREESURFER_HOME/build-stamp.txt
  then
    echo "ERROR: You are trying to run recon-surf with FreeSurfer version $(cat $FREESURFER_HOME/build-stamp.txt)."
    echo "We are currently only supporting FreeSurfer 6 Stable releases. Older versions are not compatible"
    echo "and we do not recommend to use FreeSurfer v7 together with FastSurfer at this time."
    echo "Therefore, make sure to export and source the correct FreeSurfer version before running recon-surf.sh: "
    echo "export FREESURFER_HOME=/path/to/your/local/fs60"
    echo "source \$FREESURFER_HOME/SetUpFreeSurfer.sh"
    exit 1;
  fi
fi

if [ -z "$PYTHONUNBUFFERED" ]
then
  export PYTHONUNBUFFERED=0
fi

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
  # Set to default
  seg="${SUBJECTS_DIR}/${subject}/mri/aparc.DKTatlas+aseg.deep.mgz"
fi

if [ ! -f "$seg" ]
then
  # No segmentation found, exit with error
  echo "ERROR: Segmentation ($seg) could not be found! "
  echo "Segmentation must either exist in default location (\$SUBJECTS_DIR/\$SID/mri/aparc.DKTatlas+aseg.deep.mgz) or you must supply the absolute path and name via --seg."
  exit 1;
fi

if [ "$DoParallel" == "1" ]
then
  echo " " |& tee -a $LF
  echo " RUNNING both hemis in PARALLEL " |& tee -a $LF
  echo " " |& tee -a $LF
else
  echo " " |& tee -a $LF
  echo " RUNNING both hemis SEQUENTIALLY " |& tee -a $LF
  echo " " |& tee -a $LF
fi

# set threads for openMP and itk
fsthreads=""
export OMP_NUM_THREADS=$threads
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$threads
if [ "$threads" -gt "1" ]
then
  fsthreads="-threads $threads -itkthreads $threads"
fi

# if OMP_NUM_THREADS is not set and available ressources are too vast, mc will fail with segmentation fault!
# Therefore we set it to 1, if nothing is specified.
if [ ! -z $OMP_NUM_THREADS ]
then
  echo " " |& tee -a $LF
  echo " RUNNING $OMP_NUM_THREADS number of OMP THREADS " |& tee -a $LF
  echo " " |& tee -a $LF
else
  export OMP_NUM_THREADS=1
  echo " " |& tee -a $LF
  echo " RUNNING $OMP_NUM_THREADS number of OMP THREADS " |& tee -a $LF
  echo " " |& tee -a $LF
fi

if [ ! -z $ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS ]
then
  echo " " |& tee -a $LF
  echo " RUNNING $ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS number of ITK THREADS " |& tee -a $LF
  echo " " |& tee -a $LF
fi

# collect info
StartTime=`date`;
tSecStart=`date '+%s'`;
year=`date +%Y`
month=`date +%m`
day=`date +%d`
hour=`date +%H`
min=`date +%M`


# Setup dirs
mkdir -p $SUBJECTS_DIR/$subject/scripts
mkdir -p $SUBJECTS_DIR/$subject/mri/transforms
mkdir -p $SUBJECTS_DIR/$subject/mri/tmp
mkdir -p $SUBJECTS_DIR/$subject/surf
mkdir -p $SUBJECTS_DIR/$subject/label
mkdir -p $SUBJECTS_DIR/$subject/stats

mdir=$SUBJECTS_DIR/$subject/mri
sdir=$SUBJECTS_DIR/$subject/surf
ldir=$SUBJECTS_DIR/$subject/label

mask=$mdir/mask.mgz


# Set up log file
DoneFile=$SUBJECTS_DIR/$subject/scripts/recon-surf.done
if [ $DoneFile != /dev/null ] ; then  rm -f $DoneFile ; fi
LF=$SUBJECTS_DIR/$subject/scripts/recon-surf.log
if [ $LF != /dev/null ] ; then  rm -f $LF ; fi
echo "Log file for recon-surf.sh" >> $LF
date  |& tee -a $LF
echo "" |& tee -a $LF
echo "export SUBJECTS_DIR=$SUBJECTS_DIR" |& tee -a $LF
echo "cd `pwd`"  |& tee -a $LF
echo $0 ${inputargs[*]} |& tee -a $LF
echo "" |& tee -a $LF
cat $FREESURFER_HOME/build-stamp.txt |& tee -a $LF
echo $VERSION |& tee -a $LF
uname -a  |& tee -a $LF

#if false; then

########################################## START ########################################################

echo " " |& tee -a $LF
echo "================== Creating orig and rawavg from input =========================" |& tee -a $LF
echo " " |& tee -a $LF

# check for input conformance
cmd="$python ${binpath}../FastSurferCNN/data_loader/conform.py -i $t1 --check_only --verbose"
RunIt "$cmd" $LF

cmd="$python ${binpath}../FastSurferCNN/data_loader/conform.py -i $seg --check_only --seg_input --verbose"
RunIt "$cmd" $LF

# create orig.mgz and aparc.DKTatlas+aseg.orig.mgz (copy of segmentation)
cmd="mri_convert $t1 $mdir/orig.mgz"
RunIt "$cmd" $LF

cmd="mri_convert $seg $mdir/aparc.DKTatlas+aseg.orig.mgz"
RunIt "$cmd" $LF

# link to rawavg (needed by pctsurfcon)
pushd $mdir
cmd="ln -sf orig.mgz rawavg.mgz"
RunIt "$cmd" $LF
popd

echo " " |& tee -a $LF
echo "============= Creating aseg.auto_noCCseg (map aparc labels back) ===============" |& tee -a $LF
echo " " |& tee -a $LF

# reduce labels to aseg, then create mask (dilate 5, erode 4, largest component), also mask aseg to remove outliers
# output will be uchar (else mri_cc will fail below)
cmd="$python ${binpath}reduce_to_aseg.py -i $mdir/aparc.DKTatlas+aseg.orig.mgz -o $mdir/aseg.auto_noCCseg.mgz --outmask $mask"
RunIt "$cmd" $LF


echo " " |& tee -a $LF
echo "============= Computing Talairach Transform and NU (bias corrected) ============" |& tee -a $LF
echo " " |& tee -a $LF

pushd $mdir

# nu processing is changed here compared to recon-all: we use the brainmask from the segmentation to improve the nu correction (and speedup)

# orig_nu 44 sec nu correct
cmd="mri_nu_correct.mni --no-rescale --i $mdir/orig.mgz --o $mdir/orig_nu.mgz --n 1 --proto-iters 1000 --distance 50 --mask $mdir/mask.mgz"
RunIt "$cmd" $LF

# talairach.xfm: compute talairach full head (25sec)
cmd="talairach_avi --i $mdir/orig_nu.mgz --xfm $mdir/transforms/talairach.xfm"
RunIt "$cmd" $LF

# talairach.lta:  convert to lta
cmd="lta_convert --inmni $mdir/transforms/talairach.xfm --outlta $mdir/transforms/talairach.lta --src $mdir/orig.mgz --trg $FREESURFER_HOME/average/mni305.cor.mgz --ltavox2vox"
RunIt "$cmd" $LF

# create better nu.mgz using talairach transform
NuIterations=2 # default 1.5T
NuIterations="1 --proto-iters 1000 --distance 50"  # default 3T
cmd="mri_nu_correct.mni --i $mdir/orig.mgz --o $mdir/nu.mgz --uchar $mdir/transforms/talairach.xfm --n $NuIterations --mask $mdir/mask.mgz"
RunIt "$cmd" $LF
# Add xfm to nu
cmd="mri_add_xform_to_header -c $mdir/transforms/talairach.xfm $mdir/nu.mgz $mdir/nu.mgz"
RunIt "$cmd" $LF

popd


echo " " |& tee -a $LF
echo "============ Creating brainmask from aseg and norm, and update aseg ============" |& tee -a $LF
echo " " |& tee -a $LF

# create norm by masking nu
cmd="mri_mask $mdir/nu.mgz $mdir/mask.mgz $mdir/norm.mgz"
RunIt "$cmd" $LF

if [ "$get_t1" == "1" ]
then
  # create T1.mgz from nu
  cmd="mri_normalize -g 1 -mprage $mdir/nu.mgz $mdir/T1.mgz"
  RunIt "$cmd" $LF

  # create brainmask by masking T1
  cmd="mri_mask $mdir/T1.mgz $mdir/mask.mgz $mdir/brainmask.mgz"
  RunIt "$cmd" $LF
else
  # Default: create brainmask by linkage to norm.mgz (masked nu.mgz)
  pushd $mdir
  cmd="ln -sf norm.mgz brainmask.mgz"
  RunIt "$cmd" $LF
  popd
fi

# create aseg.auto including cc segmentation and add cc into aparc.DKTatlas+aseg.deep; 46 sec: (not sure if this is needed), requires norm.mgz
cmd="mri_cc -aseg aseg.auto_noCCseg.mgz -o aseg.auto.mgz -lta $mdir/transforms/cc_up.lta $subject"
RunIt "$cmd" $LF

cmd="$python ${binpath}paint_cc_into_pred.py -in_cc $mdir/aseg.auto.mgz -in_pred $seg -out $mdir/aparc.DKTatlas+aseg.deep.withCC.mgz"
RunIt "$cmd" $LF

if [ "$vol_segstats" == "1" ]
  # Calculate volume-based segstats for deep learning prediction (with CC, requires norm.mgz as invol)
then
    cmd="mri_segstats --seg $mdir/aparc.DKTatlas+aseg.deep.withCC.mgz --sum $mdir/../stats/aparc.DKTatlas+aseg.deep.volume.stats --pv $mdir/norm.mgz --empty --brainmask $mdir/brainmask.mgz --brain-vol-from-seg --excludeid 0 --subcortgray --in $mdir/norm.mgz --in-intensity-name norm --in-intensity-units MR --etiv --id 2, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26, 28, 31, 41, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 63, 77, 251, 252, 253, 254, 255, 1002, 1003, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1034, 1035, 2002, 2003, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2034, 2035 --ctab /$FREESURFER_HOME/FreeSurferColorLUT.txt --subject $subject"
    RunIt "$cmd" $LF
fi

if [ "$seg_cc" == "1" ]
  # stop workflow at this point (only segmentation with corpus callosum was requested, surface models will be skipped.
then
    echo " " |& tee -a $LF
    echo "User requested segmentation only (with Corpus Callosum and/or vol segstats)." |& tee -a $LF
    echo "Therefore, pipeline finishes at this point. No surfaces will be created." |& tee -a $LF
    echo "================= DONE =========================================================" |& tee -a $LF
    echo " " |& tee -a $LF
    echo "recon-surf.sh $subject finished without error at `date`"  |& tee -a $LF
    echo " " |& tee -a $LF
    exit 0;
fi

echo " " |& tee -a $LF
echo "========= Creating filled from brain (brainfinalsurfs, wm.asegedit, wm)  =======" |& tee -a $LF
echo " " |& tee -a $LF

cmd="recon-all -s $subject -asegmerge -normalization2 -maskbfs -segmentation -fill $fsthreads"
RunIt "$cmd" $LF


# ================================================== SURFACES ==========================================================


CMDFS=""

for hemi in lh rh; do

CMDF="$SUBJECTS_DIR/$subject/scripts/$hemi.processing.cmdf"
CMDFS="$CMDFS $CMDF"
rm -rf $CMDF

echo "#!/bin/bash" > $CMDF
echo "echo " |& tee -a $CMDF
echo "echo \"================== Creating surfaces $hemi - orig.nofix ==================\"" |& tee -a $CMDF
echo "echo " |& tee -a $CMDF


if [ "$fstess" == "1" ]
then
  cmd="recon-all -s $subject -hemi $hemi -tessellate -smooth1 -no-isrunning $fsthreads"
  RunIt "$cmd" $LF $CMDF
else
  # instead of mri_tesselate lego land use marching cube

    if [ $hemi == "lh" ]; then
        hemivalue=255;
    else
        hemivalue=127;
    fi

    # extract initial surface "?h.orig.nofix"
    cmd="mri_pretess $mdir/filled.mgz $hemivalue $mdir/brain.mgz $mdir/filled-pretess$hemivalue.mgz"
    RunIt "$cmd" $LF $CMDF

    # Marching cube does not return filename and wrong volume info!
    cmd="mri_mc $mdir/filled-pretess$hemivalue.mgz $hemivalue $sdir/$hemi.orig.nofix"
    RunIt "$cmd" $LF $CMDF

    # Rewrite surface orig.nofix to fix vertex locs bug (scannerRAS instead of surfaceRAS set with mc)
    cmd="$python ${binpath}rewrite_mc_surface.py --input $sdir/$hemi.orig.nofix --output $sdir/$hemi.orig.nofix --filename_pretess $mdir/filled-pretess$hemivalue.mgz"
    RunIt "$cmd" $LF $CMDF

    # Check if the surfaceRAS was correctly set and exit otherwise (sanity check in case nibabel changes their default header behaviour)
    cmd="mris_info $sdir/$hemi.orig.nofix | grep -q 'vertex locs : surfaceRAS'"
    echo "echo \"$cmd\" " |& tee -a $CMDF
    echo "$timecmd $cmd " |& tee -a $CMDF
    echo "if [ \${PIPESTATUS[1]} -ne 0 ] ; then echo \"Incorrect header information detected: vertex locs is not set to surfaceRAS. Exiting... \"; exit 1 ; fi" >> $CMDF

    # Reduce to largest component (usually there should only be one)
    cmd="mris_extract_main_component $sdir/$hemi.orig.nofix $sdir/$hemi.orig.nofix"
    RunIt "$cmd" $LF $CMDF

    # -smooth1 (explicitly state 10 iteration (default) but may change in future
    cmd="mris_smooth -n 10 -nw -seed 1234 $sdir/$hemi.orig.nofix $sdir/$hemi.smoothwm.nofix"
    RunIt "$cmd" $LF $CMDF

fi



echo "echo  " |& tee -a $CMDF
echo "echo \"=================== Creating surfaces $hemi - qsphere ====================\"" |& tee -a $CMDF
echo "echo " |& tee -a $CMDF

#surface inflation (54sec both hemis) (needed for qsphere and for topo-fixer)
cmd="recon-all -s $subject -hemi $hemi -inflate1 -no-isrunning $fsthreads"
RunIt "$cmd" $LF $CMDF


if [ "$fsqsphere" == "1" ]
then
  # quick spherical mapping (2min48sec)
  cmd="recon-all -s $subject -hemi $hemi -qsphere -no-isrunning $fsthreads"
  RunIt "$cmd" $LF $CMDF

else

    # instead of mris_sphere, directly project to sphere with spectral approach
    # equivalent to -qsphere
    # (23sec)
    cmd="$python ${binpath}spherically_project_wrapper.py --hemi $hemi --sdir $sdir --subject $subject --threads=$threads --py $python --binpath ${binpath}"

    RunIt "$cmd" $LF $CMDF

fi


echo "echo " |& tee -a $CMDF
echo "echo \"=================== Creating surfaces $hemi - fix ========================\"" |& tee -a $CMDF
echo "echo " |& tee -a $CMDF

## -fix
cmd="recon-all -s $subject -hemi $hemi -fix -no-isrunning $fsthreads"
RunIt "$cmd" $LF $CMDF
  ## copy nofix to orig and inflated for next step
  # -white (don't know how to call this from recon-all as it needs -whiteonly setting and by default it also creates the pial.
  # create first WM surface white.preaparc from topo fixed orig surf, also first cortex label (1min), (3min for deep learning surf)

  if [ "$fstess" == "1" ]
  then
    cmd="mris_make_surfaces -aseg aseg.presurf -white white.preaparc -noaparc -whiteonly -mgz -T1 brain.finalsurfs $subject $hemi"
  else
     # seems like surfaces from mri_mc cause segfaults in mris_make_surf from 6.0, so we need to use a copy of dev:
     cmd="${binpath}mris_make_surfaces -aseg aseg.presurf -white white.preaparc -noaparc -whiteonly -mgz -T1 brain.finalsurfs $subject $hemi"
  fi
  RunIt "$cmd" $LF $CMDF



echo "echo \" \"" |& tee -a $CMDF
echo "echo \"================== Creating surfaces $hemi - inflate2 ====================\"" |& tee -a $CMDF
echo "echo \" \"" |& tee -a $CMDF


# create nicer inflated surface from topo fixed (not needed, just later for visualization)
cmd="recon-all -s $subject -hemi $hemi -smooth2 -inflate2 -curvHK -curvstats -no-isrunning $fsthreads"
RunIt "$cmd" $LF $CMDF


echo "echo \" \"" |& tee -a $CMDF
echo "echo \"=========== Creating surfaces $hemi - map input seg to surf ===============\"" |& tee -a $CMDF
echo "echo \" \"" |& tee -a $CMDF

    # sample input segmentation (aparc.DKTatlas+aseg orig) onto wm surface:
    # map input aparc to surface (requires thickness (and thus pail) to compute projfrac 0.5), here we do projmm which allows us to compute based only on white
    # this is dangerous, as some cortices could be < 0.6 mm, but then there is no volume label probably anyway.
    # Also note that currently we cannot mask non-cortex regions here, should be done in mris_anatomical stats later
    # the smoothing helps
    cmd="mris_sample_parc -ct $FREESURFER_HOME/average/colortable_desikan_killiany.txt -file ${binpath}$hemi.DKTatlaslookup.txt -projmm 0.6 -f 5  -surf white.preaparc $subject $hemi aparc.DKTatlas+aseg.orig.mgz aparc.DKTatlas.mapped.prefix.annot"
    RunIt "$cmd" $LF $CMDF

    cmd="$python ${binpath}smooth_aparc.py --insurf $sdir/$hemi.white.preaparc --inaparc $ldir/$hemi.aparc.DKTatlas.mapped.prefix.annot --incort $ldir/$hemi.cortex.label --outaparc $ldir/$hemi.aparc.DKTatlas.mapped.annot"
    RunIt "$cmd" $LF $CMDF


if [ "$fsaparc" == "1" ] || [ "$fssurfreg" == "1" ] ; then
  echo "echo \" \"" |& tee -a $CMDF
  echo "echo \"============ Creating surfaces $hemi - FS sphere, surfreg ===============\"" |& tee -a $CMDF
  echo "echo \" \"" |& tee -a $CMDF

  # Surface registration for cross-subject correspondance (registration to fsaverage)
  cmd="recon-all -s $subject -hemi $hemi -sphere -no-isrunning $fsthreads"
  RunIt "$cmd" $LF "$CMDF"
  
  # (mr) FIX: sometimes FreeSurfer Sphere Reg. fails and moves pre and post central 
  # one gyrus too far posterior, FastSurferCNN's image-based segmentation does not 
  # seem to do this, so we initialize the spherical registration with the better 
  # label from FastSurferCNN, this replaces recon-al -surfreg
  # 1. extract label 24 = precentral from FastSurferCNN mapped annotation
  cmd="mri_annotation2label --subject $subject --hemi $hemi --label 24 --labelbase ${hemi}.mapped --annotation aparc.DKTatlas.mapped --surface white.preaparc"
  RunIt "$cmd" $LF "$CMDF"
  # 2. guide spherical registration to align label 24 to precentral in the atlas
  cmd="mris_register -curv \
       -L ${ldir}/${hemi}.mapped-024.label \
       $FREESURFER_HOME/average/${hemi}.DKTaparc.atlas.acfb40.noaparc.i12.2016-08-02.gcs precentral \
       $SUBJECTS_DIR/$subject/surf/${hemi}.sphere \
       $FREESURFER_HOME/average/${hemi}.folding.atlas.acfb40.noaparc.i12.2016-08-02.tif \
       $SUBJECTS_DIR/$subject/surf/${hemi}.sphere.reg"
  RunIt "$cmd" $LF "$CMDF"
  # command to generate new aparc to check if registration was OK
  # run only for debugging
  #cmd="mris_ca_label -l $SUBJECTS_DIR/$subject/label/${hemi}.cortex.label \
  #     -aseg $SUBJECTS_DIR/$sid/mri/aseg.presurf.mgz \
  #     -seed 1234 $sid $hemi $SUBJECTS_DIR/$sid/surf/${hemi}.sphere.reg \
  #     $SUBJECTS_DIR/$sid/label/${hemi}.aparc.DKTatlas-guided.annot"
  
fi


if [ "$fsaparc" == "1" ] ; then

echo "echo \" \"" |& tee -a $CMDF
echo "echo \"============ Creating surfaces $hemi - FS seg..pial ===============\"" |& tee -a $CMDF
echo "echo \" \"" |& tee -a $CMDF

    # 20-25 min for traditional surface segmentation (each hemi)
    # this creates aparc and also computes jacobian
    cmd="recon-all -s $subject -hemi $hemi -jacobian_white -avgcurv -cortparc -no-isrunning $fsthreads"
    RunIt "$cmd" $LF $CMDF

    # creates pial using aparc, marching cube surfaces makes mris_make_surfaces crash here, so use dev binary in that case
    if [ "$fstess" == "1" ]
    then
      cmd="mris_make_surfaces -orig_white white.preaparc -orig_pial white.preaparc -aseg aseg.presurf -mgz -T1 brain.finalsurfs $subject $hemi"
    else
      cmd="${binpath}mris_make_surfaces -orig_white white.preaparc -orig_pial white.preaparc -aseg aseg.presurf -mgz -T1 brain.finalsurfs $subject $hemi"
    fi
    RunIt "$cmd" $LF $CMDF
else
echo "echo \" \"" |& tee -a $CMDF
echo "echo \"================ Creating surfaces $hemi - pial direct ===================\"" |& tee -a $CMDF
echo "echo \" \"" |& tee -a $CMDF

    # 3 min compute pial and cortex label, and thickness without using aparc :
    if [ "$fstess" == "1" ]
    then
      cmd="mris_make_surfaces -noaparc -nowhite -orig_white white.preaparc -orig_pial white.preaparc -aseg aseg.presurf -mgz -T1 brain.finalsurfs $subject $hemi"
    else
      cmd="${binpath}mris_make_surfaces -noaparc -nowhite -orig_white white.preaparc -orig_pial white.preaparc -aseg aseg.presurf -mgz -T1 brain.finalsurfs $subject $hemi"
    fi
    RunIt "$cmd" $LF $CMDF
    echo "pushd $sdir" >> $CMDF
    cmd="ln -sf $hemi.white.preaparc $hemi.white"
    RunIt "$cmd" $LF $CMDF
    echo "popd" >> $CMDF
fi

echo "echo \" \"" |& tee -a $CMDF
echo "echo \"================ Creating surfaces $hemi - surfvol  ======================\"" |& tee -a $CMDF
echo "echo \" \"" |& tee -a $CMDF

# (10 sec, 5 each hemi) compute vertex wise volume (?h.volume) and mid.area (?h.area.mid) :
# not really needed but so quick, lets just do it
cmd="recon-all -s $subject -hemi $hemi -surfvolume -no-isrunning $fsthreads"
RunIt "$cmd" $LF $CMDF

if [ "$DoParallel" == "0" ] ; then 
    echo " " |& tee -a $LF
    echo " RUNNING $hemi sequentially ... " |& tee -a $LF
    echo " " |& tee -a $LF
  chmod u+x $CMDF
  RunIt "$CMDF" $LF
fi


done  # hemi loop ----------------------------------



if [ "$DoParallel" == 1 ] ; then 
    echo " " |& tee -a $LF
    echo " RUNNING HEMIs in PARALLEL !!! " |& tee -a $LF
    echo " " |& tee -a $LF
    RunBatchJobs $LF $CMDFS
fi



echo " " |& tee -a $LF
echo "============================ Creating surfaces - ribbon ===========================" |& tee -a $LF
echo " " |& tee -a $LF
  # -cortribbon 4 minutes, ribbon is used in mris_anatomical stats to remove voxels from surface based volumes that should not be cortex
  # anatomical stats can run without ribon, but will omit some surface based measures then
  # wmparc needs ribbon, probably other stuff (aparc to aseg etc). 
  # could be stripped but lets run it to have these measures below
  cmd="recon-all -s $subject -cortribbon $fsthreads"
  RunIt "$cmd" $LF



if [ "$fsaparc" == "1" ] ; then

echo " " |& tee -a $LF
  echo "============= Creating surfaces - other FS seg and stats =======================" |& tee -a $LF
echo " " |& tee -a $LF

  cmd="recon-all -s $subject  -parcstats -cortparc2 -parcstats2 -cortparc3 -parcstats3 -pctsurfcon -hyporelabel $fsthreads"
  RunIt "$cmd" $LF

# in fs6.0 recon-all -aparc2aseg uses the talairach.m3z which does not exist (in fsdev recon-all does this):
# mri_aparc2aseg --s $subject --volmask --aseg aseg.presurf.hypos 
  cmd="mri_aparc2aseg --s $subject --volmask --aseg aseg.presurf.hypos"
  RunIt "$cmd" $LF  
  
  cmd="mri_aparc2aseg --s $subject --volmask --annot aparc.a2009s --aseg aseg.presurf.hypos --a2009s"
  RunIt "$cmd" $LF
  
  cmd="recon-all -s $subject -apas2aseg -segstats -wmparc -balabels $fsthreads"
  RunIt "$cmd" $LF

fi  # (FS-APARC)


echo " " |& tee -a $LF
echo "===================== Creating surfaces - mapped stats =========================" |& tee -a $LF
echo " " |& tee -a $LF
 

 # 2x18sec create stats from mapped aparc
for hemi in lh rh; do
  cmd="mris_anatomical_stats -th3 -mgz -cortex $ldir/$hemi.cortex.label -f $sdir/../stats/$hemi.aparc.DKTatlas.mapped.stats -b -a $ldir/$hemi.aparc.DKTatlas.mapped.annot -c $ldir/aparc.annot.mapped.ctab $subject $hemi white"
  RunIt "$cmd" $LF
done


if [ "$fsaparc" == "0" ] ; then

echo " " |& tee -a $LF
echo "============= Creating surfaces - pctsurfcon, hypo, segstats ====================" |& tee -a $LF
echo " " |& tee -a $LF

# pctsurfcon (has no way to specify which annot to use, so we need to link ours as aparc is not available)
  pushd $ldir
  cmd="ln -sf lh.aparc.DKTatlas.mapped.annot lh.aparc.annot"
  RunIt "$cmd" $LF
  cmd="ln -sf rh.aparc.DKTatlas.mapped.annot rh.aparc.annot"
  RunIt "$cmd" $LF
  popd
  for hemi in lh rh; do 
    cmd="pctsurfcon --s $subject --$hemi-only"
    RunIt "$cmd" $LF
  done
  pushd $ldir
  cmd="rm *h.aparc.annot"
  RunIt "$cmd" $LF
  popd

  # 18 sec hyporelabel (generate aseg.presurf.hypos.mgz from aseg.presurf and ?h.white)
  cmd="recon-all -s $subject -hyporelabel $fsthreads"
  RunIt "$cmd" $LF
fi

# generate aparc.DKTatlas+aseg.mapped.mgz, needed later to paint-in white matter labels also
# 55sec mapping aparc.DKTatlas.mapped back to volume (could be a nicer aparc+aseg compared to input, due to surface help, not verified yet)
cmd="mri_aparc2aseg --s $subject --volmask --aseg aseg.presurf.hypos --annot aparc.DKTatlas.mapped --o $mdir/aparc.DKTatlas+aseg.mapped.mgz"
RunIt "$cmd" $LF

if [ "$fsaparc" == "0" ] ; then

  # 4sec creating an aseg from the aparc.DKTatlas+aseg.mapped.mgz (instead of aseg.presurf.hypos..)
  # we call it aseg, because that is needed below in recon-all segstats
  cmd="apas2aseg --i $mdir/aparc.DKTatlas+aseg.mapped.mgz --o $mdir/aseg.mgz "
  RunIt "$cmd" $LF

  # get stats for the aseg (note these are surface fine tuned, that may be good or bad, below we also do the stats for the input aseg (plus some processing)
  cmd="recon-all -s $subject -segstats $fsthreads"
  RunIt "$cmd" $LF
 
  # balabels need sphere.reg
  if [ "$fssurfreg" == "1" ] ; then
      # can be produced if surf registration exists
      cmd="recon-all -s $subject -balabels $fsthreads"
      RunIt "$cmd" $LF "$CMDF"
  fi

fi



echo " " |& tee -a $LF
echo "===================== Creating wmparc from mapped =======================" |& tee -a $LF
echo " " |& tee -a $LF

  # 1m 11sec also create stats for aseg.presurf.hypos (which is basically the aseg derived from the input with CC and hypos)
  # difference between this and the surface improved one above are probably tiny, so the surface improvement above can probably be skipped to save time
  # in dev version the seed can be given in command line, but not in 6.0:
  cmd="mri_segstats --seg $mdir/aseg.presurf.hypos.mgz --sum $mdir/../stats/aseg.presurf.hypos.stats --pv $mdir/norm.mgz --empty --brainmask $mdir/brainmask.mgz --brain-vol-from-seg --excludeid 0 --excl-ctxgmwm --supratent --subcortgray --in $mdir/norm.mgz --in-intensity-name norm --in-intensity-units MR --etiv --surf-wm-vol --surf-ctx-vol --totalgray --euler --ctab /$FREESURFER_HOME/ASegStatsLUT.txt --subject $subject"
  RunIt "$cmd" $LF

  # -wmparc based on mapped aparc labels (from input seg) (1min40sec) needs ribbon and we need to point it to aparc.DKTatlas.mapped:
  cmd="mri_aparc2aseg --s $subject --labelwm --hypo-as-wm --rip-unknown --volmask --o $mdir/wmparc.DKTatlas.mapped.mgz --ctxseg $mdir/aparc.DKTatlas+aseg.mapped.mgz --annot aparc.DKTatlas.mapped --annot-table $ldir/aparc.annot.mapped.ctab"
  RunIt "$cmd" $LF
   
  # takes a few mins
  # in dev version the seed can be given in command line, but not in 6.0:
  cmd="mri_segstats --seg $mdir/wmparc.DKTatlas.mapped.mgz --sum $mdir/../stats/wmparc.DKTatlas.mapped.stats --pv $mdir/norm.mgz --excludeid 0 --brainmask $mdir/brainmask.mgz --in $mdir/norm.mgz --in-intensity-name norm --in-intensity-units MR --subject $subject --surf-wm-vol --ctab $FREESURFER_HOME/WMParcStatsLUT.txt"
  RunIt "$cmd" $LF

# Create symlinks for downstream analysis (sub-segmentations, TRACULA, etc.)
if [ "$fsaparc" == "0" ] ; then
  # Symlink of aparc.DKTatlas+aseg.mapped.mgz
  pushd $mdir
  cmd="ln -sf aparc.DKTatlas+aseg.mapped.mgz aparc.DKTatlas+aseg.mgz"
  RunIt "$cmd" $LF
  cmd="ln -sf aparc.DKTatlas+aseg.mapped.mgz aparc+aseg.mgz"
  RunIt "$cmd" $LF
  popd

  # Symlink of wmparc.mapped
  pushd $mdir
  cmd="ln -sf wmparc.DKTatlas.mapped.mgz wmparc.mgz"
  RunIt "$cmd" $LF
  popd

  # Symbolic link for mapped surface parcellations
  pushd $ldir
  cmd="ln -sf lh.aparc.DKTatlas.mapped.annot lh.aparc.DKTatlas.annot"
  RunIt "$cmd" $LF
  cmd="ln -sf rh.aparc.DKTatlas.mapped.annot rh.aparc.DKTatlas.annot"
  RunIt "$cmd" $LF
fi

echo " " |& tee -a $LF
echo "================= DONE =========================================================" |& tee -a $LF
echo " " |& tee -a $LF

# Collect info
EndTime=`date`
tSecEnd=`date '+%s'`
tRunHours=`echo \($tSecEnd - $tSecStart\)/3600|bc -l`
tRunHours=`printf %6.3f $tRunHours`

echo "Started at $StartTime " |& tee -a $LF
echo "Ended   at $EndTime" |& tee -a $LF
echo "#@#%# recon-surf-run-time-hours $tRunHours" |& tee -a $LF

# Create the Done File
echo "------------------------------" > $DoneFile
echo "SUBJECT $subject"           >> $DoneFile
echo "START_TIME $StartTime"      >> $DoneFile
echo "END_TIME $EndTime"          >> $DoneFile
echo "RUNTIME_HOURS $tRunHours"   >> $DoneFile
echo "USER `id -un`"              >> $DoneFile
echo "HOST `hostname`"            >> $DoneFile
echo "PROCESSOR `uname -m`"       >> $DoneFile
echo "OS `uname -s`"              >> $DoneFile
echo "UNAME `uname -a`"           >> $DoneFile
echo "VERSION $VERSION"           >> $DoneFile
echo "CMDPATH $0"                 >> $DoneFile
echo "CMDARGS ${inputargs[*]}"    >> $DoneFile

echo "recon-surf.sh $subject finished without error at `date`"  |& tee -a $LF

cmd="$python ${binpath}utils/extract_recon_surf_time_info.py -i $LF -o $SUBJECTS_DIR/$subject/scripts/recon-surf_times.yaml"
RunIt "$cmd"
