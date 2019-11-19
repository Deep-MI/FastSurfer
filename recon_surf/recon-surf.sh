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

t1="";
seg="";
subject="";
fstess=1;       # run mri_tesselate (FS way), if 0 = run mri_mc
fsqsphere=1;    # run inflate1 and qsphere (FSway), if 0 run spectral projection
fsaparc=1;	# run FS aparc (and cortical ribbon), if 0 map aparc from seg input
fssurfreg=0;  # run FS surface registration to fsaverage, if 0 omit this step

timecmd="fs_time"
binpath="./"
python="python36"
DoParallel=0
threads="1"

function usage()
{
    echo ""
    echo "recon-surf.sh takes a segmentation and T1 full head image and creates surfaces, thickness etc as a FS subject dir"
    echo ""
    echo "./recon-surf.sh"
    echo -e "\t--sid <subjectID>             Subject ID for directory inside \$SUBJECTS_DIR to be created"
    echo -e "\t--sd  <subjects_dir>          Output directory \$SUBJECTS_DIR (pass via environment or here)"
    echo -e "\t--t1  <T1_input>              T1 full head input (not bias corrected)"
    echo -e "\t--seg <segmentation_input>    Segmentation (similar to aparc+aseg)"
    echo -e "\t--mc                          Switch on marching cube for surface creation"
    echo -e "\t--qspec                       Switch on spectral spherical projection for qsphere"
    echo -e "\t--nofsaparc                   Skip FS aparc segmentations and ribbon for speedup"
    echo -e "\t--surfreg                     Run Surface registration with FreeSurfer (for cross-subject correspondance)"
    echo -e "\t--parallel                    Run both hemispheres in parallel"
    echo -e "\t--threads <int>               Set openMP and ITK threads to <int>"
    echo -e "\t--dev                         Switch on usage of dev-version for FreeSurfer"
    echo -e "\t--py <python_cmd>             Command for python, default 'python36'"
    echo -e "\t-h --help                     Print Help"
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
    --mc)
    fstess=0
    shift # past argument
    ;;
    --qspec)
    fsqsphere=0
    shift # past argument
    ;;
    --nofsaparc)
    fsaparc=0
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
    --dev)
    binpath=""
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
echo
echo sid $subject
echo T1  $t1
echo seg $seg
echo

if [ -z "$SUBJECTS_DIR" ]
then
  echo \$SUBJECTS_DIR not set
  exit 1;
fi

if [ -z "$t1" ]
 then
  echo "ERROR: must supply T1 input (conformed, full head) via --t1"
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
  echo "ERROR: must supply brain segmentation via --seg"
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

# create orig.mgz and aparc+aseg.orig.mgz (copy of segmentation)
cmd="mri_convert $t1 $mdir/orig.mgz"
RunIt "$cmd" $LF

cmd="mri_convert $seg $mdir/aparc+aseg.orig.mgz"
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
cmd="$python reduce_to_aseg.py -i $mdir/aparc+aseg.orig.mgz -o $mdir/aseg.auto_noCCseg.mgz --outmask $mask"
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


# create norm and brainmask by masking nu
cmd="mri_mask $mdir/nu.mgz $mdir/mask.mgz $mdir/norm.mgz"
RunIt "$cmd" $LF
pushd $mdir
cmd="ln -sf norm.mgz brainmask.mgz"
RunIt "$cmd" $LF
popd


# create aseg.auto including cc segmentation 46 sec: (not sure if this is needed), requries norm.mgz
cmd="mri_cc -aseg aseg.auto_noCCseg.mgz -o aseg.auto.mgz -lta $mdir/transforms/cc_up.lta $subject"
RunIt "$cmd" $LF


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

    cmd="mri_mc $mdir/filled-pretess$hemivalue.mgz $hemivalue $sdir/$hemi.orig.nofix"
    RunIt "$cmd" $LF $CMDF

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
    cmd="$python spherically_project_wrapper.py --hemi $hemi --sdir $sdir --subject $subject --threads=$threads --py $python"

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
    cmd="${binpath}mris_make_surfaces -aseg ../mri/aseg.presurf -white white.preaparc -noaparc -whiteonly -mgz -T1 brain.finalsurfs $subject $hemi"
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

# sample input segemntation (aparc+aseg orig) onto wm surface:
    # map input aparc to surface (requrires thickness (and thus pail) to compute projfrac 0.5), here we do projmm which allows us to compute based only on white
    # this is dangerous, as some cortices could be < 0.6 mm, but then there is no volume label probably anyway.
    # Also note that currently we cannot mask non-cortex regions here, should be done in mris_anatomical stats later
    # the smoothing helps
    cmd="mris_sample_parc -ct $FREESURFER_HOME/average/colortable_desikan_killiany.txt -file ./$hemi.DKTatlaslookup.txt -projmm 0.6 -f 5  -surf white.preaparc $subject $hemi aparc+aseg.orig.mgz aparc.mapped.prefix.annot"
    RunIt "$cmd" $LF $CMDF
    
    cmd="$python smooth_aparc.py --insurf $sdir/$hemi.white.preaparc --inaparc $ldir/$hemi.aparc.mapped.prefix.annot --incort $ldir/$hemi.cortex.label --outaparc $ldir/$hemi.aparc.mapped.annot"
    RunIt "$cmd" $LF $CMDF
    
 
 
if [ "$fsaparc" == "1" ] ; then
  
echo "echo \" \"" |& tee -a $CMDF
echo "echo \"============ Creating surfaces $hemi - FS sphere, seg..pial ===============\"" |& tee -a $CMDF
echo "echo \" \"" |& tee -a $CMDF

    # 20-25 min for traditional surface segmentation (each hemi)
    # this registers to sphere, creates aparc and creates pial using aparc, also computes jacobian
    cmd="recon-all -s $subject -hemi $hemi -sphere -surfreg -jacobian_white -avgcurv -cortparc -pial -no-isrunning $fsthreads"
    RunIt "$cmd" $LF $CMDF

else
echo "echo \" \"" |& tee -a $CMDF
echo "echo \"================ Creating surfaces $hemi - pial direct ===================\"" |& tee -a $CMDF
echo "echo \" \"" |& tee -a $CMDF

    # 3 min compute pial and cortex label, and thickness without using aparc :
    cmd="${binpath}mris_make_surfaces -noaparc -nowhite -orig_white white.preaparc -orig_pial white.preaparc -aseg aseg.presurf -mgz -T1 brain.finalsurfs $subject $hemi"
    RunIt "$cmd" $LF $CMDF
    echo "pushd $sdir" >> $CMDF
    cmd="ln -sf $hemi.white.preaparc $hemi.white"
    RunIt "$cmd" $LF $CMDF
    echo "popd" >> $CMDF
fi

if [ "$fssurfreg" == "1" ] ; then
  echo "echo \" \"" |& tee -a $CMDF
  echo "echo \"============ Creating surfaces $hemi - FS sphere, surfreg ===============\"" |& tee -a $CMDF
  echo "echo \" \"" |& tee -a $CMDF

  # Surface registration for cross-subject correspondance (registration to fsaverage)
  cmd="recon-all -s $subject -hemi $hemi -sphere -surfreg -no-isrunning $fsthreads"
  RunIt "$cmd" $LF "$CMDF"
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
  
  cmd="recon-all -s $subject -apas2aseg -segstats -wmparc -balabels $fsthreads"
  RunIt "$cmd" $LF

fi  # (FS-APARC)


echo " " |& tee -a $LF
echo "===================== Creating surfaces - mapped stats =========================" |& tee -a $LF
echo " " |& tee -a $LF
 

 # 2x18sec create stats from mapped aparc
for hemi in lh rh; do
  cmd="mris_anatomical_stats -th3 -mgz -cortex $ldir/$hemi.cortex.label -f $sdir/../stats/$hemi.aparc.mapped.stats -b -a $ldir/$hemi.aparc.mapped.annot -c $ldir/aparc.annot.mapped.ctab $subject $hemi white"
  RunIt "$cmd" $LF
done


if [ "$fsaparc" == "0" ] ; then

echo " " |& tee -a $LF
echo "============= Creating surfaces - pctsurfcon, hypo, segstats ====================" |& tee -a $LF
echo " " |& tee -a $LF

# pctsurfcon (has no way to specify which annot to use, so we need to link ours as aparc is not available)
  pushd $ldir
  cmd="ln -sf lh.aparc.mapped.annot lh.aparc.annot"
  RunIt "$cmd" $LF
  cmd="ln -sf rh.aparc.mapped.annot rh.aparc.annot"
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


  # 18 sec hyporelabel run whatever else can be done without sphere, cortical ribbon and segmentations
  cmd="recon-all -s $subject -hyporelabel $fsthreads"
  RunIt "$cmd" $LF

  # 55sec mapping aparc.mapped back to volume (should be a nicer aparc+aseg compared to input, due to surface help)???
  cmd="mri_aparc2aseg --s $subject --volmask --aseg aseg.presurf.hypos --annot aparc.mapped --o $mdir/aparc.mapped+aseg.mgz  "
  RunIt "$cmd" $LF

  # 4sec creating an aseg from the aparc.mapped+aseg.mgz (should be better than the aseg.presurf.hypos..)
  # we call it aseg, because that is needed below in recon-all segstats
  cmd="apas2aseg --i $mdir/aparc.mapped+aseg.mgz --o $mdir/aseg.mgz "
  RunIt "$cmd" $LF

  # get stats for the aseg (note these are surface fine tuned, that may be good or bad, below we also do the stats for the input aseg (plus some processing)
  cmd="recon-all -s $subject -segstats $fsthreads"
  RunIt "$cmd" $LF
 
  # balabels need sphere.reg

fi



echo " " |& tee -a $LF
echo "===================== Creating wmparc from mapped =======================" |& tee -a $LF
echo " " |& tee -a $LF

  # 1m 11sec also create stats for aseg.presurf.hypos (which is basically the aseg derived from the input with CC and hypos)
  # difference between this and the surface improved one above are probably tiny, so the surface improvement above can probably be skipped to save time
  # in dev version the seed can be given in command line, but not in 6.0:
  cmd="mri_segstats --seg $mdir/aseg.presurf.hypos.mgz --sum $mdir/../stats/aseg.presurf.hypos.stats --pv $mdir/norm.mgz --empty --brainmask $mdir/brainmask.mgz --brain-vol-from-seg --excludeid 0 --excl-ctxgmwm --supratent --subcortgray --in $mdir/norm.mgz --in-intensity-name norm --in-intensity-units MR --etiv --surf-wm-vol --surf-ctx-vol --totalgray --euler --ctab /$FREESURFER_HOME/ASegStatsLUT.txt --subject $subject"
  RunIt "$cmd" $LF

  # -wmparc based on mapped aparc labels (from input seg) (1min40sec) needs ribbon and we need to point it to aparc.mapped:
  # labels are messed up, due to the aparc mapped surface labels which are incorrect, we need a lookup above.
  cmd="mri_aparc2aseg --s $subject --labelwm --hypo-as-wm --rip-unknown --volmask --o $mdir/wmparc.mapped.mgz --ctxseg $mdir/aparc+aseg.orig.mgz --annot aparc.mapped --annot-table $ldir/aparc.annot.mapped.ctab"
  RunIt "$cmd" $LF
   
  # takes a few mins
  # in dev version the seed can be given in command line, but not in 6.0:
  cmd="mri_segstats --seg $mdir/wmparc.mapped.mgz --sum $mdir/../stats/wmparc.mapped.stats --pv $mdir/norm.mgz --excludeid 0 --brainmask $mdir/brainmask.mgz --in $mdir/norm.mgz --in-intensity-name norm --in-intensity-units MR --subject $subject --surf-wm-vol --ctab $FREESURFER_HOME/WMParcStatsLUT.txt"
  RunIt "$cmd" $LF

echo " " |& tee -a $LF
echo "================= DONE =========================================================" |& tee -a $LF
echo " " |& tee -a $LF
echo "recon-surf.sh $subject finished without error at `date`"  |& tee -a $LF 
echo " " |& tee -a $LF
