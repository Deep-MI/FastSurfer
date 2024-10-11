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
t1=""                 # Path and name of T1 input
asegdkt_segfile=""    # Path and name of segmentation
subject=""            # Subject name
fstess=0              # run mri_tesselate (FS way), if 0 = run mri_mc
fsqsphere=0           # run inflate1 and qsphere (FSway), if 0 run spectral projection
fsaparc=0             # run FS aparc (and cortical ribbon), if 0 map aparc from asegdkt_segfile
fssurfreg=1           # run FS surface registration to fsaverage, if 0 omit this step
python="python3.10"   # python version
DoParallel=0          # if 1, run hemispheres in parallel
threads="1"           # number of threads to use for running FastSurfer
allow_root=""         # flag for allowing execution as root user
atlas3T="false"       # flag to use/do not use the 3t atlas for talairach registration/etiv
segstats_legacy="false" # flag to enable segstats legacy mode
base=0                # flag for longitudinal template (base) run
long=0                # flag for longitudinal time point run
baseid=""             # baseid for logitudinal time point run

# Dev flags default
check_version=1       # Check for supported FreeSurfer version (terminate if not detected)
get_t1=1              # Generate T1.mgz from nu.mgz and brainmask from it (default)
hires_voxsize_threshold=0.999  # Threshold below which the hires options are passed

if [ -z "$FASTSURFER_HOME" ]
then
  binpath="$(cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )/"
  FASTSURFER_HOME="$(cd -- "$(dirname "$binpath")" >/dev/null 2>&1 ; pwd -P )/"
else
  binpath="$FASTSURFER_HOME/recon_surf/"
fi


# check bash version > 3.1 (needed for printf %q)
function version { echo "$@" | awk -F. '{ printf("%d%03d%03d%03d\n", $1,$2,$3,$4); }'; }
if [ "$(version "${BASH_VERSION}")" -lt "$(version "3.1.0")" ]; then
    echo "bash ${BASH_VERSION} is too old. Should be newer than 3.1, please upgrade!"
    exit 1
fi


function usage()
{
cat << EOF

Usage: recon-surf.sh --sid <sid> --sd <sdir> --t1 <t1> --asegdkt_segfile <asegdkt_segfile> [OPTIONS]

recon-surf.sh takes a segmentation and T1 full head image and creates surfaces,
thickness etc as a FS subject dir.

FLAGS:
  --sid <subjectID>       Subject ID to create directory inside \$SUBJECTS_DIR 
  --sd  <subjects_dir>    Output directory \$SUBJECTS_DIR (or pass via env var)
  --t1  <T1_input>        T1 full head input (not bias corrected). This must be
                            a conformed image (dimensions: 256x256x256, voxel
                            size: 1x1x1, LIA orientation, and data type UCHAR).
                            Images can be conformed using FastSurferCNN's
                            conform.py script (usage example: python3
                            FastSurferCNN/data_loader/conform.py -i <T1_input>
                            -o <conformed_T1_output>). Requires an ABSOLUTE Path!
  --asegdkt_segfile <asegdkt_segfile>
                          Name of intermediate DL-based segmentation file
                            (similar to aparc+aseg). This must be conformed
                            (voxel size: isotropic, LIA orientation, and, if voxel
                            size 1mm, dimensions: 256x256x256). FastSurferCNN's
                            segmentations are conformed by default; please ensure
                            that segmentations produced otherwise are conformed.
                            Requires an ABSOLUTE Path! Default location:
                            \$SUBJECTS_DIR/\$sid/mri/aparc.DKTatlas+aseg.deep.mgz
  --fstess                Revert to FreeSurfer mri_tesselate for surface creation
                            (default: mri_mc)
  --fsqsphere             Revert to FreeSurfer iterative inflation for qsphere
                            (default: spectral spherical projection)
  --fsaparc               Additionally create FS aparc segmentations and ribbon.
                            Skipped by default (--> DL prediction is used which
                            is faster, and usually these mapped ones are fine).
                            Note, if you switch this on it will create all cortical
                            parcellations with FreeSurfer's spherical atlases and
                            also map these into the aparc+aseg file instead of
                            the FastSurfer ones. FastSurfer's cortical DKT atlas
                            results can still be found in:
                            <hemi>.aparc.DKTatlas.mapped.stats
  --3T                    Use the 3T atlas for talairach registration (gives better
                            eTIV estimates for 3T MR images, default: 1.5T atlas).
  --parallel              Run both hemispheres in parallel
  --threads <int>         Set openMP and ITK threads to <int>
  --py <python_cmd>       Command for python, default ${python}
  --fs_license <license>  Path to FreeSurfer license key file. Register at
                            https://surfer.nmr.mgh.harvard.edu/registration.html
                            for free to obtain it if you do not have FreeSurfer
                            installed already.
  --base                  For longitudinal template (base) creation.
  --long <baseid>         For longitudinal time point creation, pass the ID of
                            the base (template) which needs to exist already in
                            the same subjects_dir.
  -h --help               Print Help

Dev Flags:
  --ignore_fs_version     Switch on to avoid check for FreeSurfer version.
                            Program will otherwise terminate if $FS_VERSION_SUPPORT is 
                            not sourced. Can be used for testing dev versions.
  --no_fs_T1              Do not generate T1.mgz (normalized nu.mgz included in
                            standard FreeSurfer output) and create brainmask.mgz
                            directly from norm.mgz instead. Saves 1:30 min.
  --no_surfreg            Do not run Surface registration with FreeSurfer (for
                            cross-subject correspondence). Not recommended, but
                            speeds up processing if you just need the stats and
                            don't want to do thickness analysis on the cortex.
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

# Load the RunIt and the RunBatchJobs functions, also sets up timecmd
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
# make key lowercase
key=$(echo "$1" | tr '[:upper:]' '[:lower:]')
shift # past argument

case $key in
  --sid) subject="$1" ; shift ;;
  --sd) export SUBJECTS_DIR="$1" ; shift ;;
  --t1) t1="$1" ; shift ;;
  --asegdkt_segfile | --aparc_aseg_segfile | --seg)
    if [ "$key" == "--seg" ] || [ "$key" == "--aparc_aseg_segfile" ]; then
      echo "WARNING: $key <filename> is deprecated and will be removed, use --asegdkt_segfile <filename>."
    fi
    asegdkt_segfile="$1"
    shift # past value
    ;;
  --vol_segstats)
    echo "WARNING: The --vol_segstats flag is obsolete and will be removed, --vol_segstats ignored."
    ;;
  --segstats_legacy) segstats_legacy="true" ;;
  --fstess) fstess=1 ;;
  --fsqsphere) fsqsphere=1 ;;
  --fsaparc) fsaparc=1 ;;
  --no_surfreg) fssurfreg=0 ;;
  --3t) atlas3T="true" ;;
  --parallel) DoParallel=1 ;;
  --threads) threads="$1" ; shift ;;
  --py) python="$1" ; shift ;;
  --fs_license)
    if [ -f "$1" ]; then
      export FS_LICENSE="$1"
    else
      echo "ERROR: Provided FreeSurfer license file $1 could not be found."
      echo "  Make sure to provide the full path and name."
      exit 1;
    fi
    shift # past value
    ;;
  --ignore_fs_version) check_version=0 ;;
  --no_fs_t1 ) get_t1=0 ;;
  --allow_root) allow_root="--allow_root" ;;
  --base) base=1 ;;
  --long) long=1 ; baseid="$1" ; shift ;;
  -h|--help) usage ; exit ;;
  # unknown option
  *) echo "ERROR: Flag $key unrecognized." ; exit 1 ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

# CHECKS
echo ""
echo "sid $subject"
echo "T1  $t1"
echo "asegdkt_segfile $asegdkt_segfile"
echo ""


# Warning if run as root user
check_allow_root

if [ -z "$SUBJECTS_DIR" ]
then
  echo "ERROR: \$SUBJECTS_DIR not set. Either set it via the shell prior to"
  echo "  running recon-surf.sh or supply it via the --sd flag."
  exit 1
fi

if [ -z "$FREESURFER_HOME" ]
then
  echo "ERROR: Did not find \$FREESURFER_HOME. A working version of FreeSurfer $FS_VERSION_SUPPORT"
  echo "  is needed to run recon-surf locally."
  echo "  Make sure to export and source FreeSurfer before running recon-surf.sh: "
  echo "  export FREESURFER_HOME=/path/to/your/local/fs$FS_VERSION_SUPPORT"
  echo "  source \$FREESURFER_HOME/SetUpFreeSurfer.sh"
  exit 1
fi
# needed in FS72 due to a bug in recon-all --fill using FREESURFER instead of FREESURFER_HOME
export FREESURFER=$FREESURFER_HOME   

if [ "$check_version" == "1" ]
then
  if grep -q -v "${FS_VERSION_SUPPORT}" "$FREESURFER_HOME/build-stamp.txt"
  then
    echo "ERROR: You are trying to run recon-surf with FreeSurfer version $(cat "$FREESURFER_HOME/build-stamp.txt")."
    echo "  We are currently supporting only FreeSurfer $FS_VERSION_SUPPORT."
    echo "  Therefore, make sure to export and source the correct FreeSurfer version"
    echo "  before running recon-surf.sh: "
    echo "  export FREESURFER_HOME=/path/to/your/local/fs$FS_VERSION_SUPPORT"
    echo "  source \$FREESURFER_HOME/SetUpFreeSurfer.sh"
    exit 1
  fi
fi

if [ -z "$PYTHONUNBUFFERED" ]
then
  export PYTHONUNBUFFERED=0
fi

if [[ "$long" == "1" ]] && [[ "$base" == "1" ]]
then
  echo "ERROR: You specified both --long and --base. You need to setup and then run base template first,"
  echo "before you can run any longitudinal time points."
  exit 1;
fi

if [[ "$base" == "1" ]]
then
  if [ ! -f "$sd/$subject/base-tps.fastsurfer" ] ; then
    echo "ERROR: $subject is either not found in SUBJECTS_DIR"
    echo "or it is not a longitudinal template directory (base),"
    echo "which needs to contain base-tps.fastsurfer file. Please ensure that"
    echo "the base (template) has been created with long_prepare_template.sh."
    exit 1
  fi
fi

basedir=""
if [ "$long" == "1" ]
then
  basedir="$SUBJECTS_DIR/$baseid"
  if [ ! -f "$basedir/base-tps.fastsurfer" ] ; then
    echo "ERROR: $baseid is either not found in \$SUBJECTS_DIR or it is not a longitudinal"
    echo "  template directory, which needs to contain base-tps.fastsurfer file. Please"
    echo "  ensure that the base (template) has been created when running with --long flag."
    exit 1
  fi
  if ! grep -Fxq "$subject" "$basedir/base-tps.fastsurfer" ; then
    echo "ERROR: $subject id not found in base-tps.fastsurfer. Please ensure that this time"
    echo "  point was included during creation of the base (template)."
    exit 1
  fi
fi

if [ -z "$t1" ] || [ ! -f "$t1" ]
then
  echo "ERROR: T1 image ($t1) could not be found. Must supply an existing T1 input"
  echo "  (conformed, full head) via --t1 (absolute path and name)."
  # needed to create orig.mgz and to get file name. This will eventually be changed.
  exit 1
fi

if [ -z "$subject" ]
then
  echo "ERROR: must supply subject name via --sid"
  exit 1
fi

if [ -z "$asegdkt_segfile" ]
then
  # Set to default
  asegdkt_segfile="${SUBJECTS_DIR}/${subject}/mri/aparc.DKTatlas+aseg.deep.mgz"
fi

if [ ! -f "$asegdkt_segfile" ]
then
  # No segmentation found, exit with error
  echo "ERROR: Segmentation ($asegdkt_segfile) could not be found! "
  echo "  Segmentation must either exist in default location (\$SUBJECTS_DIR/\$SID/mri/"
  echo "  aparc.DKTatlas+aseg.deep.mgz) or you must supply the absolute path and name via"
  echo "  --asegdkt_segfile <filename>."
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

if [ "$(echo -n "${SUBJECTS_DIR}/${subject}" | wc -m)" -gt 185 ]
then
  echo "ERROR: Subject directory path is very long."
  echo "  This is known to cause errors due to some commands run by freesurfer versions built for Ubuntu."
  echo "  --sd + --sid should be less than 185 characters long."
  exit 1
fi

# Check if running on an existing subject directory
if [ -f "$SUBJECTS_DIR/$subject/mri/wm.mgz" ] || [ -f "$SUBJECTS_DIR/$subject/mri/aparc.DKTatlas+aseg.orig.mgz" ]
then
  echo "ERROR: Running on top of an existing subject directory!"
  echo "  The output directory must not contain data from a previous invocation of recon-surf."
  exit 1
fi

# collect info
StartTime=$(date)
tSecStart=$(date '+%s')
year=$(date +%Y)
month=$(date +%m)
day=$(date +%d)
hour=$(date +%H)
min=$(date +%M)


# Setup dirs
mkdir -p "$SUBJECTS_DIR/$subject/scripts"
mkdir -p "$SUBJECTS_DIR/$subject/mri/transforms"
mkdir -p "$SUBJECTS_DIR/$subject/mri/tmp"
mkdir -p "$SUBJECTS_DIR/$subject/surf"
mkdir -p "$SUBJECTS_DIR/$subject/label"
mkdir -p "$SUBJECTS_DIR/$subject/stats"

mdir="$SUBJECTS_DIR/$subject/mri"
sdir="$SUBJECTS_DIR/$subject/surf"
statsdir="$SUBJECTS_DIR/$subject/stats"
ldir="$SUBJECTS_DIR/$subject/label"

mask="$mdir/mask.mgz"


# Set up log file
DoneFile="$SUBJECTS_DIR/$subject/scripts/recon-surf.done"
if [ "$DoneFile" != /dev/null ] ; then  rm -f "$DoneFile" ; fi
LF="$SUBJECTS_DIR/$subject/scripts/recon-surf.log"
if [ "$LF" != /dev/null ] ; then  rm -f "$LF" ; fi
echo "Log file for recon-surf.sh" >> "$LF"
{ # all output tee -a "$LF"
  date 2>&1
  echo " "
  echo "export SUBJECTS_DIR=$SUBJECTS_DIR"
  echo "cd $(pwd)"
  echo_quoted "$0" "${inputargs[@]}"
  echo " "
  cat "$FREESURFER_HOME/build-stamp.txt" 2>&1
  echo "$VERSION"
  uname -a 2>&1
  echo " "
  if [ "$base" == "1" ] ; then
    echo " "
    echo "================== BASE - Longitudinal Template Creation ========================="
    echo " "
  fi
  if [ "$long" == "1" ] ; then
    echo " "
    echo "================== LONG - Longitudinal Timpe Point Creation ======================"
    echo "long: using template directory (base) $baseid"
    echo " "
  fi
  # Print parallelization parameters
  if [ "$DoParallel" == "1" ]
  then
    echo " RUNNING both hemis in PARALLEL"
  else
    echo " RUNNING both hemis SEQUENTIALLY"
  fi
  echo " RUNNING $OMP_NUM_THREADS number of OMP THREADS"
  echo " RUNNING $ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS number of ITK THREADS"
  echo " "

  # Check input segmentation quality
  echo "Checking Input Segmentation Quality ..."
} | tee -a "$LF"

cmd="$python $FASTSURFER_HOME/FastSurferCNN/quick_qc.py --asegdkt_segfile $asegdkt_segfile"
RunIt "$cmd" "$LF"
echo "" | tee -a "$LF"

########################################## START ########################################################

{
  echo " "
  echo "================== Creating orig and rawavg from input ========================="
  echo " "
} | tee -a "$LF"

CONFORM_LF=$SUBJECTS_DIR/$subject/scripts/conform.log
if [ "$CONFORM_LF" != /dev/null ] ; then  rm -f "$CONFORM_LF" ; fi
echo "Log file for Conform test" > "$CONFORM_LF"

# check for input conformance
cmd="$python $FASTSURFER_HOME/FastSurferCNN/data_loader/conform.py -i $t1 --check_only --vox_size min --verbose"
RunIt "$cmd" "$LF" 2>&1 | tee -a "$CONFORM_LF"

# look into the CONFORM_LF to find the voxel sizes, the second conform.py call will check the legality of vox_size
vox_size=$(grep -E " - Voxel Size " "$CONFORM_LF" | cut -d' ' -f5 | cut -d'x' -f1)
# remove the temporary conform_log (all info is also in the recon-surf logfile)
if [ -f "$CONFORM_LF" ]; then rm -f "$CONFORM_LF" ; fi

# here, we check the correct vox_size by passing it to the next conform, so errors in this line might be caused above
cmd="$python $FASTSURFER_HOME/FastSurferCNN/data_loader/conform.py -i $asegdkt_segfile --check_only --vox_size $vox_size --dtype any --verbose"
RunIt "$cmd" "$LF"

if (( $(echo "$vox_size < $hires_voxsize_threshold" | bc -l) ))
then
  echo "The voxel size $vox_size is less than $hires_voxsize_threshold, so we are proceeding with hires options." | tee -a "$LF"
  hiresflag="-hires"
  noconform_if_hires=" -noconform"
  hires_surface_suffix=".predec"
else
  echo "The voxel size $vox_size is not less than $hires_voxsize_threshold, so we are proceeding with standard options." | tee -a "$LF"
  hiresflag=""
  noconform_if_hires=""
  hires_surface_suffix=""
fi

# create orig.mgz and aparc.DKTatlas+aseg.orig.mgz (copy of T1 and segmentation)
# also ensures .mgz format (in case inputs are nifti)
cmd="mri_convert $t1 $mdir/orig.mgz"
RunIt "$cmd" "$LF"

cmd="mri_convert $asegdkt_segfile $mdir/aparc.DKTatlas+aseg.orig.mgz"
RunIt "$cmd" "$LF"

# link original T1 input to rawavg (needed by pctsurfcon)
pushd "$mdir" > /dev/null || ( echo "Could not change to $mdir" ; exit 1 )
  softlink_or_copy "orig.mgz" "rawavg.mgz" "$LF"
popd > /dev/null || ( echo "Could not change to subject_dir" ; exit 1 )



### The following steps are now usually done outside recon-surf already by the segmentation pipeline.
### However, if these files such as mask, aseg.auto_noCCseg, orig_nu or talairach transforms don't
### exist, we recreate them here, so that this can run on other type of input where only a T1 and
### segmentation is provided. This may need update if it changes in the segmentation pipeline.


# ============================= MASK & ASEG_noCC ========================================

if [ "$long" == "1" ] ; then
  # for long we copy mask from base
  cmda=(cp "$basedir/mri/mask.mgz" "$mask")
  run_it "$LF" "${cmda[@]}"
fi

if [ ! -f "$mask" ] || [ ! -f "$mdir/aseg.auto_noCCseg.mgz" ] ; then
  {
    # Mask or aseg.auto_noCCseg not found; create them from aparc.DKTatlas+aseg
    echo " "
    echo "============= Creating aseg.auto_noCCseg (map aparc labels back) ==============="
    echo " "
    echo "WARNING: mri/mask.mgz or mri/aseg.auto_noCCseg.mgz are missing, but these files are"
    echo "  required in recon-surf.sh and always created in the segmentation pipeline run."
    echo "  It is recommended to transfer these files from there!"
  } | tee -a "$LF"

  # reduce labels to aseg, then create mask (dilate 5, erode 4, largest component), also mask aseg to remove outliers
  # output will be uchar (else mri_cc will fail below)
  cmda=($python "$FASTSURFER_HOME/FastSurferCNN/reduce_to_aseg.py" -i "$mdir/aparc.DKTatlas+aseg.orig.mgz"
        -o "$mdir/$aseg_nocc" --fixwm)

  if [ "$base" == "1" ] && [ ! -f "$mask" ] ; then
    # for base we build union of mapped masks beforehand so it should be available
    echo "ERROR: $mask missing, but base run requires $mask!" | tee -a "$LF"
    exit 1
  elif [ "$long" != "1" ] && [ "$base" != 1 ] ; then
    # cross-sectional processing, add outmask to cmd (not for or base long stream)
    cmda+=(--outmask "$mask")
  fi

  run_it "$LF" "${cmda[@]}"
fi


# ============================= NU BIAS CORRECTION =======================================

if [ ! -f "$mdir/orig_nu.mgz" ] ; then
  # only run the bias field correction, if the bias field corrected does not exist already
  {
    echo " "
    echo "============= Computing NU (bias corrected) ============"
    echo " "
  } | tee -a "$LF"
  # nu processing is changed here compared to recon-all: we use the brainmask from the
  # segmentation to improve the nu correction (and speedup)
  # orig_nu N3 in FS6 took 44 sec, FS 7.3.2 uses --ants-n4 (takes 3 min and does not accept
  # the mask due to a bug in AntsN4BiasFieldCorrectionFs wrapper).
  # This re-implementation uses N4 from simpleITK with our brainmask, we also directly
  # scale WM to 110 using a ball at the center of the mask with radius 50 (similar to FS,
  # which uses origin of talairach.xfm and grabs quite a few non brain region in the
  # frontal head), we don't. Also this avoids a second call to nu correct.
  # talairach.xfm is also not needed here at all, it can be dropped if other places in the
  # stream can be changed to avoid it.
  pushd "$mdir" > /dev/null || ( echo "Cannot change to $mdir" ; exit 1 )
    #cmd="mri_nu_correct.mni --no-rescale --i $mdir/orig.mgz --o $mdir/orig_nu.mgz --n 1 --proto-iters 1000 --distance 50 --mask $mdir/mask.mgz"
    cmd="$python ${binpath}/N4_bias_correct.py --in $mdir/orig.mgz --rescale $mdir/orig_nu.mgz --aseg $mdir/aparc.DKTatlas+aseg.orig.mgz --threads $threads"
    RunIt "$cmd" "$LF"
  popd > /dev/null || (echo "Could not popd" ; exit 1)
fi


# ============================= TALAIRACH ==============================================

if [[ ! -f "$mdir/transforms/talairach.lta" ]] || [[ ! -f "$mdir/transforms/talairach_with_skull.lta" ]] ; then
  # if talairach registration is missing, compute it here
  # this also creates talairach.auto.xfm and talairach.xfm and talairach.xfm.lta
  # all transforms (also ltas) are the same
  cmda=("$binpath/talairach-reg.sh" "$LF"
        --dir "$mdir" --conformed_name "$mdir/orig.mgz" --norm_name "$mdir/orig_nu.mgz")
  if [[ "$long" == "1" ]] ; then cmda+=(--long "$basedir") ; fi
  if [[ "$atlas3T" == "true" ]] ; then cmda+=(--3T) ; fi

  {
    echo " "
    echo "============= Computing Talairach Transform ============"
    echo " "
  } | tee -a "$LF"
  echo_quoted "${cmda[@]}"
  "${cmda[@]}"
fi


# ============================= BRAINMASK ==============================================
{
  echo " "
  echo "============ Creating brainmask from aseg and nu or T1 ============"
  echo " "
} | tee -a $LF

# the difference between nu and orig_nu is the fact that nu has the talairach-registration header
# create norm by masking nu
cmda=(mri_mask "$mdir/nu.mgz" "$mask" "$mdir/norm.mgz")
run_it "$LF" "${cmda[@]}"
if [ "$get_t1" == "1" ]
then
  # create T1.mgz from nu (!! here we could also try passing aseg?)
  # T1.mgz was needed by some 3rd party downstream tools such as fmriprep, so we provide it

  # if base template run, write ctrl and bias vol files (maybe switch on later)
  # this has mainly an effect in FreeSurfer on the segmentation, but if images come
  # from different scanners it could hurt more than help. Here in FastSurfer
  # it is unclear what effect it would even have, given that segmentations come
  # from the FastSurferVINN. It could affect surface placement or partial volumes.
  #base_flags=""
  #if [ "$base" == "1" ]
  #then
  #  base_flags="-w $mdir/ctrl_vol.mgz $mdir/bias_vol.mgz"
  #fi
  # cmd="mri_normalize -g 1 -seed 1234 -mprage $base_flags $mdir/nu.mgz $mdir/T1.mgz $noconform_if_hires"
  cmda=(mri_normalize -g 1 -seed 1234 -mprage "$mdir/nu.mgz" "$mdir/T1.mgz" $noconform_if_hires)
  run_it "$LF" "${cmda[@]}"
  # create brainmask by masking T1
  cmda=(mri_mask "$mdir/T1.mgz" "$mask" "$mdir/brainmask.mgz")
  run_it "$LF" "${cmda[@]}"
else
  # create brainmask by linkage to norm.mgz (masked nu.mgz)
  pushd "$mdir" > /dev/null || ( echo "Could not cd to $mdir" ; exit 1 )
    softlink_or_copy "norm.mgz" "brainmask.mgz" "$LF"
  popd > /dev/null || (echo "Could not popd" ; exit 1 )
fi


# ============================= CC SEGMENTATION ============================================

{
  echo " "
  echo "============ Creating and adding CC Segmentation ============"
  echo " "
} | tee -a $LF
# create aseg.auto including corpus callosum segmentation and 46 sec, requires norm.mgz
# Note: if original input segmentation already contains CC, this will exit with ERROR
# in the future maybe check and skip this step (and next)
cmd="mri_cc -aseg aseg.auto_noCCseg.mgz -o aseg.auto.mgz -lta $mdir/transforms/cc_up.lta $subject"
RunIt "$cmd" "$LF"
# add CC into aparc.DKTatlas+aseg.deep (not sure if this is really needed)
cmd="$python ${binpath}paint_cc_into_pred.py -in_cc $mdir/aseg.auto.mgz -in_pred $asegdkt_segfile -out $mdir/aparc.DKTatlas+aseg.deep.withCC.mgz"
RunIt "$cmd" "$LF"


# ============================= FILLED =====================================================

{
  echo " "
  echo "========= Creating filled from brain (brainfinalsurfs, wm.asegedit, wm)  ======="
  echo " "
} | tee -a "$LF"

if [ "$long" == "1" ] ; then
  # in long we can skip fill as surfaces come from base
  # it would be great to also skip WM, but it is needed in place_surface to clip bright
  # maybe later add code to copy edits from base in maskbfs and wm segmentation, currently not supported!
  cmd="recon-all -s $subject -asegmerge -normalization2 -maskbfs -segmentation $hiresflag $fsthreads"
  RunIt "$cmd" "$LF"
  # copy over filled from base for stop-edits to transfer to long (a bit of a hack)
  cmd="cp $basedir/mri/filled.mgz $mdir/filled.mgz"
  RunIt "$cmd" "$LF"
else # cross and base
  # filled is needed to generate initial WM surfaces
  cmd="recon-all -s $subject -asegmerge -normalization2 -maskbfs -segmentation -fill $hiresflag $fsthreads"
  RunIt "$cmd" "$LF"
fi


# =======
# ================================================== SURFACES ==============================================================
# =======

CMDFS=()

for hemi in lh rh ; do

  CMDF="$SUBJECTS_DIR/$subject/scripts/$hemi.processing.cmdf"
  CMDFS+=("$CMDF")
  rm -rf "$CMDF"
  echo "#!/bin/bash" > "$CMDF"


# ============================= TESSELLATE - SMOOTH =====================================================

  # In Long stream we skip these
  if [ "$long" == "0" ] ; then

  {
    echo "echo \" \""
    echo "echo \"================== Creating surfaces $hemi - orig.nofix ==================\""
    echo "echo \" \""
  } | tee -a "$CMDF"

  if [ "$fstess" == "1" ]
  then
    cmd="recon-all -subject $subject -hemi $hemi -tessellate -smooth1 -no-isrunning $hiresflag $fsthreads"
    RunIt "$cmd" "$LF" "$CMDF"
  else
    # instead of mri_tesselate lego land use marching cube

    if [ $hemi == "lh" ] ; then
        hemivalue=255
    else
        hemivalue=127
    fi

    # extract initial surface "?h.orig.nofix"
    cmd="mri_pretess $mdir/filled.mgz $hemivalue $mdir/brain.mgz $mdir/filled-pretess$hemivalue.mgz"
    RunIt "$cmd" "$LF" "$CMDF"

    # Marching cube does not return filename and wrong volume info!
    outmesh=$sdir/$hemi.orig.nofix$hires_surface_suffix
    cmd="mri_mc $mdir/filled-pretess$hemivalue.mgz $hemivalue $outmesh"
    RunIt "$cmd" "$LF" "$CMDF"

    # Rewrite surface orig.nofix to fix vertex locs bug (scannerRAS instead of surfaceRAS set with mc)
    #cmd="$python ${binpath}rewrite_mc_surface.py --input $outmesh --output $outmesh --filename_pretess $mdir/filled-pretess$hemivalue.mgz"
    #RunIt "$cmd" "$LF" "$CMDF"

    # Check if the surfaceRAS was correctly set and exit otherwise (sanity check in case nibabel changes their default header behaviour)
    {
      cmd="mris_info $outmesh | tr -s ' ' | grep -q 'vertex locs : surfaceRAS'"
      echo "echo \"$cmd\""
      echo "$timecmd $cmd"
    } | tee -a "$CMDF"
    echo "if [ \${PIPESTATUS[1]} -ne 0 ] ; then echo \"Incorrect header information detected in $outmesh: vertex locs is not set to surfaceRAS. Exiting... \" ; exit 1 ; fi" >> "$CMDF"

    # Reduce to largest component (usually there should only be one)
    cmd="mris_extract_main_component $outmesh $outmesh"
    RunIt "$cmd" "$LF" "$CMDF"
    
    # for hires decimate mesh 
    if [ -n "$hiresflag" ] ; then
      DecimationFaceArea="0.5"
      # Reduce the number of faces such that the average face area is
      # DecimationFaceArea.  If the average face area is already more
      # than DecimationFaceArea, then the surface is not changed.
      # set cmd = (mris_decimate -a $DecimationFaceArea ../surf/$hemi.orig.nofix.predec ../surf/$hemi.orig.nofix)
      cmd="mris_remesh --desired-face-area $DecimationFaceArea --input $outmesh --output $sdir/$hemi.orig.nofix"
      RunIt "$cmd" "$LF" "$CMDF"
    fi
    # -smooth1 (explicitly state 10 iteration (default) but may change in future)
    cmd="mris_smooth -n 10 -nw -seed 1234 $sdir/$hemi.orig.nofix $sdir/$hemi.smoothwm.nofix"
    RunIt "$cmd" "$LF" "$CMDF"
  fi

  else # LONG

    # here we skip most steps above (and some below) and copy surfaces from base for initialization of placement later
    cmd="cp $basedir/surf/${hemi}.white $sdir/${hemi}.orig_white"
    RunIt "$cmd" $LF
    cmd="cp $basedir/surf/${hemi}.white $sdir/${hemi}.orig"
    RunIt "$cmd" $LF
    cmd="cp $basedir/surf/${hemi}.pial $sdir/${hemi}.orig_pial"
    RunIt "$cmd" $LF

  fi # end LONG


# ============================= INFLATE1 - QSPHERE =====================================================

  # In Long stream we skip these
  if [ "$long" == "0" ] ; then

  {
    echo "echo \"\""
    echo "echo \"=================== Creating surfaces $hemi - qsphere ====================\""
    echo "echo \"\""
  } | tee -a "$CMDF"

  #surface inflation (54sec both hemis) (needed for qsphere and for topo-fixer)
  cmd="recon-all -subject $subject -hemi $hemi -inflate1 -no-isrunning $hiresflag $fsthreads"
  RunIt "$cmd" "$LF" "$CMDF"

  if [ "$fsqsphere" == "1" ]
  then
    # quick spherical mapping (2min48sec)
    cmd="recon-all -subject $subject -hemi $hemi -qsphere -no-isrunning $hiresflag $fsthreads"
    RunIt "$cmd" "$LF" "$CMDF"
  else
    # instead of mris_sphere, directly project to sphere with spectral approach
    # equivalent to -qsphere
    # (23sec)
    cmd="$python ${binpath}spherically_project_wrapper.py --hemi $hemi --sdir $sdir"
    printf -v tmp %q "$python"
    cmd="$cmd --subject $subject --threads=$threads --py ${tmp} --binpath ${binpath}"
    RunIt "$cmd" "$LF" "$CMDF"
  fi

  fi # not long

# ============================= FIX - WHITEPREAPARC ==================================================

  # In Long stream we skip topo fix
  if [ "$long" == "0" ] ; then

    {
      echo "echo \"\""
      echo "echo \"=================== Creating surfaces $hemi - fix ========================\""
      echo "echo \"\""
    } | tee -a "$CMDF"

    cmd="recon-all -subject $subject -hemi $hemi -fix -no-isrunning $hiresflag $fsthreads"
    RunIt "$cmd" "$LF" "$CMDF"

    # fix the surfaces if they are corrupt
    cmd="$python ${binpath}rewrite_oriented_surface.py --file $sdir/$hemi.orig.premesh --backup $sdir/$hemi.orig.premesh.noorient"
    RunIt "$cmd" "$LF" "$CMDF"
    cmd="$python ${binpath}rewrite_oriented_surface.py --file $sdir/$hemi.orig --backup $sdir/$hemi.orig.noorient"
    RunIt "$cmd" "$LF" "$CMDF"

  # create first WM surface white.preaparc from topo fixed orig surf
  cmd="recon-all -subject $subject -hemi $hemi -autodetgwstats -white-preaparc -no-isrunning $hiresflag $fsthreads"
  RunIt "$cmd" "$LF" "$CMDF"

  else # longitudinal

    # in long we don't use orig.premesh (so switch off remesh for autodetgwstat)
    cmd="recon-all -subject $subject -hemi $hemi -autodetgwstats -no-remesh -no-isrunning $hiresflag $fsthreads"
    RunIt "$cmd" "$LF" "$CMDF"

    # for place_surfaces white.preparc we need to directly call it with special long parameter:
    # cmd="recon-all -subject $subject -hemi $hemi -white-preaparc -no-isrunning $hiresflag $fsthreads"
    cmd="mris_place_surface --adgws-in $sdir/autodet.gw.stats.$hemi.dat --wm $mdir/$wm_file --threads $threads --invol $mdir/brain.finalsurfs.mgz --$hemi --i $sdir/$hemi.orig --o $sdir/${hemi}.white.preaparc --white --seg $mdir/aseg.presurf.mgz --max-cbv-dist 3.5"
    RunIt "$cmd" "$LF" "$CMDF"

  fi # long


# ============================= CORTEXLABEL - INFLATE2 - CURVHK ==========================================

  {
    echo "echo \"\""
    echo "echo \"================== Creating surfaces $hemi - inflate2 ====================\""
    echo "echo \"\""
  } | tee -a "$CMDF"

  # create cortex label (1min)
  # create nicer inflated surface from topo fixed (not needed, just later for visualization)
  # identical for long processing
  cmd="recon-all -subject $subject -hemi $hemi -cortex-label -smooth2 -inflate2 -curvHK -no-isrunning $hiresflag $fsthreads"
  RunIt "$cmd" "$LF" "$CMDF"


# ============================= MAP-DKT ==========================================================

  {
    echo "echo \" \""
    echo "echo \"=========== Creating surfaces $hemi - map input asegdkt_segfile to surf ===============\""
    echo "echo \" \""
  } | tee -a "$CMDF"

  # sample input segmentation (aparc.DKTatlas+aseg orig) onto wm surface:
  # map input aparc to surface (requires thickness (and thus pail) to compute projfrac 0.5), here we do projmm which allows us to compute based only on white
  # this is dangerous, as some cortices could be < 0.6 mm, but then there is no volume label probably anyway.
  # Also note that currently we cannot mask non-cortex regions here, should be done in mris_anatomical stats later
  # the smoothing helps
  #cmd="mris_sample_parc -ct $FREESURFER_HOME/average/colortable_desikan_killiany.txt -file ${binpath}$hemi.DKTatlaslookup.txt -projmm 0.6 -f 5  -surf white.preaparc $subject $hemi aparc.DKTatlas+aseg.orig.mgz aparc.DKTatlas.mapped.prefix.annot"
  #RunIt "$cmd" "$LF" "$CMDF"
  #cmd="$python ${binpath}smooth_aparc.py --insurf $sdir/$hemi.white.preaparc --inaparc $ldir/$hemi.aparc.DKTatlas.mapped.prefix.annot --incort $ldir/$hemi.cortex.label --outaparc $ldir/$hemi.aparc.DKTatlas.mapped.annot"
  #RunIt "$cmd" "$LF" "$CMDF"
  cmd="$python ${binpath}sample_parc.py --inseg $mdir/aparc.DKTatlas+aseg.orig.mgz --insurf $sdir/$hemi.white.preaparc --incort $ldir/$hemi.cortex.label --outaparc $ldir/$hemi.aparc.DKTatlas.mapped.annot --seglut ${binpath}$hemi.DKTatlaslookup.txt --surflut ${binpath}DKTatlaslookup.txt --projmm 0.6 --radius 2"
  RunIt "$cmd" "$LF" "$CMDF"


# ============================= SPHERE - SURFREG (optional) ==============================================

  # if we segment with FS or if surface registration is requested do it here:
  if [ "$fsaparc" == "1" ] || [ "$fssurfreg" == "1" ] ; then
    {
      echo "echo \" \""
      echo "echo \"============ Creating surfaces $hemi - FS sphere, surfreg ===============\""
      echo "echo \" \""
    } | tee -a "$CMDF"

    if [ "$long" == "0" ] ; then

    # SPHERE: Inflate to sphere with minimal metric distortion
    cmd="recon-all -subject $subject -hemi $hemi -sphere $hiresflag -no-isrunning $fsthreads"
    RunIt "$cmd" "$LF" "$CMDF"

    # SURFREG (sphere.reg)
    # Surface registration for cross-subject correspondence (registration to fsaverage)
    # (mr) FIX: sometimes FreeSurfer Sphere Reg. fails and moves pre and post central
    # one gyrus too far posterior, FastSurferCNN's image-based segmentation does not
    # seem to do this, so we initialize the spherical registration with the better
    # cortical segmentation from FastSurferCNN, this replaces recon-all -surfreg
    # 1. get alpha, beta, gamma for global alignment (rotation) based on aseg centers
    # (note the former fix, initializing with pre-central label, is not working in FS7.2
    # as they broke the label initialization in mris_register)
    cmd="$python ${binpath}/rotate_sphere.py \
         --srcsphere $sdir/${hemi}.sphere \
         --srcaparc $ldir/$hemi.aparc.DKTatlas.mapped.annot \
         --trgsphere $FREESURFER_HOME/subjects/fsaverage/surf/${hemi}.sphere \
         --trgaparc $FREESURFER_HOME/subjects/fsaverage/label/${hemi}.aparc.annot \
         --out $sdir/${hemi}.angles.txt"
    RunIt "$cmd" "$LF" "$CMDF"
    # 2. use global rotation as initialization to non-linear registration:
    cmd="mris_register -curv -norot -rotate \`cat $sdir/${hemi}.angles.txt\` \
         $sdir/${hemi}.sphere \
         $FREESURFER_HOME/average/${hemi}.folding.atlas.acfb40.noaparc.i12.2016-08-02.tif \
         $sdir/${hemi}.sphere.reg"
    RunIt "$cmd" "$LF" "$CMDF"
    # command to generate new aparc to check if registration was OK
    # run only for debugging
    # cmd="mris_ca_label -l $SUBJECTS_DIR/$subject/label/${hemi}.cortex.label \
    #     -aseg $SUBJECTS_DIR/$subject/mri/aseg.presurf.mgz \
    #     -seed 1234 $subject $hemi $SUBJECTS_DIR/$subject/surf/${hemi}.sphere.reg \
    #     $SUBJECTS_DIR/$subject/label/${hemi}.aparc.DKTatlas-guided.annot"

    else # longitudinal

      # SPHERE (mapping with minimal distortion) we copy it from base:
      cmd="cp $basedir/surf/$hemi.sphere $sdir/$hemi.sphere"
      RunIt "$cmd" "$LF" "$CMDF"

      # SURFREG (sphere.reg)
      # Surface registration for cross-subject correspondence (registration to fsaverage)
      # In long we initialize with sphere.reg from base template (which was also
      # copied to sphere above) and use -nosulc and -norot:
      cmd="mris_register -curv -nosulc -norot \
           -threads $threads \
           $basedir/surf/${hemi}.sphere.reg \
           $FREESURFER_HOME/average/${hemi}.folding.atlas.acfb40.noaparc.i12.2016-08-02.tif \
           $sdir/${hemi}.sphere.reg"
      RunIt "$cmd" "$LF" "$CMDF"

    fi # LONG

    # in all cases where sphere.reg is available, create jacobian white (distortion to sphere)
    # and avgcurv (map atlas curvature to subject):
    cmd="recon-all -subject $subject -hemi $hemi -jacobian_white -avgcurv -no-isrunning $hiresflag $fsthreads"
    RunIt "$cmd" "$LF" "$CMDF"

  fi


# ============================= aparc.annot (optional) ==============================================


  # aparc only takes 20 seconds, and is created when -fsaparc is passed
  # it is then used also below for surface placement.
  # we should consider, always computing it (when surfreg is available) -> test later what consequences this has
  #if [ "$fsaparc" == "1" ] || [ "$fssurfreg" == "1" ] ; then
  if [ "$fsaparc" == "1" ] ; then
    {
      echo "echo \" \""
      echo "echo \"============ Creating surfaces $hemi - FS aparc ===============\""
      echo "echo \" \""
    } | tee -a "$CMDF"

    longflag=""
    if [ "$long" == "1" ] ; then
      # recon-all has different treatment for cortparc:
      # initialize with aparc.annot from base
      longflag="-long -R $basedir/label/${hemi}.aparc.annot"
    fi
    CPAtlas="$FREESURFER_HOME/average/${hemi}.DKaparc.atlas.acfb40.noaparc.i12.2016-08-02.gcs"
    cmd="mris_ca_label -l $ldir/${hemi}.cortex.label -aseg $mdir/aseg.presurf.mgz -seed 1234 $longflag $subject $hemi $sdir/${hemi}.sphere.reg $CPAtlas $ldir/${hemi}.aparc.annot"
    RunIt "$cmd" "$LF" "$CMDF"

  fi


# ============================= SURFACES: WHITE & PIAL  =======================================================


  # first select what cortical parcellation to use to guide surface placement:
  aparc=""
  if [ "$fsaparc" == "1" ] ; then
    {
      echo "echo \" \""
      echo "echo \"============ Creating surfaces $hemi - white and pial using FS aparc ===============\""
      echo "echo \" \""
    } | tee -a "$CMDF"
    # use FS aparc in surface placement below
    aparc="../label/${hemi}.aparc.annot"

  else # FastSurfer mapped
    {
      echo "echo \" \""
      echo "echo \"================ Creating surfaces $hemi - white and pial direct ===================\""
      echo "echo \" \""
    } | tee -a "$CMDF"
    # use our mapped aparcDKT for surface placement (not sure if and where this makes a difference)
    aparc="../label/${hemi}.aparc.DKTatlas.mapped.annot"
  fi

  # change into mri dir, for local paths below (not sure this is needed, but maybe global paths did not work)
  echo "pushd $mdir > /dev/null" >> "$CMDF"

  # CREATE WHITE SURFACE:
  # 4 min compute white :
  inputsurf="../surf/$hemi.white.preaparc"
  longmaxdist=""
  if [ "$long" == "1" ] ; then
    inputsurf="../surf/$hemi.orig_white"
    longmaxdist="--max-cbv-dist 3.5"
  fi
  cmd="mris_place_surface --adgws-in ../surf/autodet.gw.stats.${hemi}.dat --seg aseg.presurf.mgz --threads $threads --wm wm.mgz --invol brain.finalsurfs.mgz --$hemi --i $inputsurf --o ../surf/${hemi}.white --white --nsmooth 0 --rip-label ../label/${hemi}.cortex.label --rip-bg --rip-surf ../surf/${hemi}.white.preaparc --aparc $aparc $longmaxdist"
  RunIt "$cmd" "$LF" "$CMDF"

  # CREAT PIAL SURFACE
  # 4 min compute pial :
  inputsurf="../surf/$hemi.white"
  longmaxdist=""
  if [ "$long" == "1" ] ; then
    inputsurf="../surf/$hemi.orig_pial"
    longmaxdist="--max-cbv-dist 3.5 --blend-surf .25 ../surf/$hemi.white"
  fi
  cmd="mris_place_surface --adgws-in ../surf/autodet.gw.stats.${hemi}.dat --seg aseg.presurf.mgz --threads $threads --wm wm.mgz --invol brain.finalsurfs.mgz --$hemi --i $inputsurf --o ../surf/${hemi}.pial.T1 --pial --nsmooth 0 --rip-label ../label/${hemi}.cortex+hipamyg.label --pin-medial-wall ../label/${hemi}.cortex.label --aparc $aparc --repulse-surf ../surf/${hemi}.white --white-surf ../surf/${hemi}.white $longmaxdist"
  RunIt "$cmd" "$LF" "$CMDF"

  echo "popd > /dev/null" >> "$CMDF"

  # Here insert DoT2Pial  later --> if T2pial is not run, need to softlink pial.T1 to pial!
  echo "pushd $sdir > /dev/null" >> "$CMDF"
  softlink_or_copy "$hemi.pial.T1" "$hemi.pial" "$LF" "$CMDF"
  echo "popd > /dev/null" >> "$CMDF"

  # these are run automatically in fs7* recon-all and cannot be called directly without -pial flag (or other t2 flags)
  # they are the same for fsaparc and long
  echo "pushd $mdir > /dev/null" >> "$CMDF"
  cmd="mris_place_surface --curv-map ../surf/$hemi.white 2 10 ../surf/$hemi.curv"
  RunIt "$cmd" "$LF" "$CMDF"
  cmd="mris_place_surface --area-map ../surf/$hemi.white ../surf/$hemi.area"
  RunIt "$cmd" "$LF" "$CMDF"
  cmd="mris_place_surface --curv-map ../surf/$hemi.pial 2 10 ../surf/$hemi.curv.pial"
  RunIt "$cmd" "$LF" "$CMDF"
  cmd="mris_place_surface --area-map ../surf/$hemi.pial ../surf/$hemi.area.pial"
  RunIt "$cmd" "$LF" "$CMDF"
  cmd="mris_place_surface --thickness ../surf/$hemi.white ../surf/$hemi.pial 20 5 ../surf/$hemi.thickness"
  RunIt "$cmd" "$LF" "$CMDF"
  echo "popd > /dev/null" >> "$CMDF"



# ============================= CURVSTATS ===============================================

  # in FS7 curvstats moves here
  cmd="recon-all -subject $subject -hemi $hemi -curvstats -no-isrunning $hiresflag $fsthreads"
  RunIt "$cmd" "$LF" "$CMDF"




  if [ "$DoParallel" == "0" ] ; then
    {
      echo " "
      echo " RUNNING $hemi sequentially ... "
      echo " "
    } | tee -a "$LF"
    chmod u+x $CMDF
    RunIt "$CMDF" $LF
  fi


done  # hemi loop ----------------------------------



if [ "$DoParallel" == 1 ] ; then
  {
    echo ""
    echo " RUNNING HEMIs in PARALLEL !!! "
    echo ""
  } | tee -a "$LF"
  RunBatchJobs "$LF" "${CMDFS[@]}"
fi

# Skip rest in case we have a base run, we are done here (probably we can skip stuff already in surface creation above)
if [ "$base" != "1" ] ; then


# ============================= RIBBON ===============================================


{
  echo ""
  echo "============================ Creating surfaces - ribbon ==========================="
  echo ""
} | tee -a "$LF"
  # -cortribbon 4 minutes, ribbon is used in mris_anatomical stats to remove voxels from surface based volumes that should not be cortex
  # anatomical stats can run without ribbon, but will omit some surface based measures then
  # wmparc needs ribbon, probably other stuff (aparc to aseg etc).
  # So lets run it to have these measures below.
  cmd="recon-all -subject $subject -cortribbon $hiresflag $fsthreads"
  RunIt "$cmd" "$LF"


# ============================= FSAPARC - parc23 surfcon hypo ... =========================================

  if [ "$fsaparc" == "1" ] ; then
    {
      echo ""
      echo "============= Creating surfaces - other FS asegdkt_segfile and stats ======================="
      echo ""
    } | tee -a "$LF"

    # Destrieux Atlas (recon-all -cortparc2):
    longflag=""
    if [ "$long" == "1" ] ; then
      # recon-all has different treatment for cortparc:
      # initialize with destrieux annot from base
      longflag="-long -R $basedir/label/${hemi}.a2009s.annot"
    fi
    CPAtlas="$FREESURFER_HOME/average/${hemi}.CDaparc.atlas.acfb40.noaparc.i12.2016-08-02.gcs"
    annot="$ldir/${hemi}.aparc.a2009s.annot"
    cmd="mris_ca_label -l $ldir/${hemi}.cortex.label -aseg $mdir/aseg.presurf.mgz -seed 1234 $longflag $subject $hemi $sdir/${hemi}.sphere.reg $CPAtlas $annot"
    RunIt "$cmd" "$LF"

    # DKT Atlas (recon-all -cortparc3):
    longflag=""
    if [ "$long" == "1" ] ; then
      # recon-all has different treatment for cortparc:
      # initialize with destrieux annot from base
      longflag="-long -R $basedir/label/${hemi}.DKTatlas.annot"
    fi
    CPAtlas="$FREESURFER_HOME/average/${hemi}.DKTaparc.atlas.acfb40.noaparc.i12.2016-08-02.gcs"
    annot="$ldir/${hemi}.aparc.DKTatlas.annot"
    cmd="mris_ca_label -l $ldir/${hemi}.cortex.label -aseg $mdir/aseg.presurf.mgz -seed 1234 $longflag $subject $hemi $sdir/${hemi}.sphere.reg $CPAtlas $annot"
    RunIt "$cmd" "$LF"

    cmd="recon-all -subject $subject -pctsurfcon -hyporelabel -apas2aseg -aparc2aseg -wmparc -parcstats -parcstats2 -parcstats3 $hiresflag $fsthreads"
    RunIt "$cmd" "$LF"
    # removed -balabels here and do that below independent of fsaparc flag
    # removed -segstats here (now part of mri_segstats.py/segstats.py
  fi  # (FS-APARC)


# ============================= MAPPED SURF-STATS =========================================

{
  echo ""
  echo "===================== Creating surfaces - mapped stats ========================="
  echo ""
} | tee -a "$LF"

  # 2x18sec create stats from mapped aparc
  for hemi in lh rh ; do
    cmd="mris_anatomical_stats -th3 -mgz -cortex $ldir/$hemi.cortex.label -f $statsdir/$hemi.aparc.DKTatlas.mapped.stats -b -a $ldir/$hemi.aparc.DKTatlas.mapped.annot -c $ldir/aparc.annot.mapped.ctab $subject $hemi white"
    RunIt "$cmd" "$LF"
  done


# ============================= FASTSURFER - surfcon hypo stats =========================================

  if [ "$fsaparc" == "0" ] ; then
    {
      echo ""
      echo "============= Creating surfaces - pctsurfcon, hypo, segstats ===================="
      echo ""
    } | tee -a "$LF"
    # pctsurfcon (has no way to specify which annot to use, so we need to link ours as aparc is not available)
    pushd "$ldir" > /dev/null || (echo "Could not cd to $ldir" ; exit 1)
      softlink_or_copy "lh.aparc.DKTatlas.mapped.annot" "lh.aparc.annot" "$LF"
      softlink_or_copy "rh.aparc.DKTatlas.mapped.annot" "rh.aparc.annot" "$LF"
    popd > /dev/null || (echo "Could not popd" ; exit 1)
    for hemi in lh rh ; do
      cmd="pctsurfcon --s $subject --$hemi-only"
      RunIt "$cmd" "$LF"
    done
    pushd "$ldir" > /dev/null || (echo "Could not cd to $ldir" ; exit 1)
      cmd="rm *h.aparc.annot"
      RunIt "$cmd" "$LF"
    popd > /dev/null || (echo "Could not popd" ; exit 1)

    # 25 sec hyporelabel run whatever else can be done without sphere, cortical ribbon and segmentations
    # -hyporelabel creates aseg.presurf.hypos.mgz from aseg.presurf.mgz
    # -apas2aseg creates aseg.mgz by editing aseg.presurf.hypos.mgz with surfaces
    cmd="recon-all -subject $subject -hyporelabel -apas2aseg $hiresflag $fsthreads"
    RunIt "$cmd" "$LF"
  fi


# ============================= MAPPED-TO-VOL =========================================

  # creating aparc.DKTatlas+aseg.mapped.mgz by mapping aparc.DKTatlas.mapped from surface to aseg.mgz
  # (should be a nicer aparc+aseg compared to orig CNN segmentation, due to surface updates)
  cmd="mri_surf2volseg --o $mdir/aparc.DKTatlas+aseg.mapped.mgz --label-cortex --i $mdir/aseg.mgz --threads $threads --lh-annot $ldir/lh.aparc.DKTatlas.mapped.annot 1000 --lh-cortex-mask $ldir/lh.cortex.label --lh-white $sdir/lh.white --lh-pial $sdir/lh.pial --rh-annot $ldir/rh.aparc.DKTatlas.mapped.annot 2000 --rh-cortex-mask $ldir/rh.cortex.label --rh-white $sdir/rh.white --rh-pial $sdir/rh.pial"
  RunIt "$cmd" "$LF"


# ============================= FASTSURFER - STATS =========================================

  if [ "$fsaparc" == "0" ] ; then
    # get stats for the aseg (note these are surface fine tuned, that may be good or bad, below we also do the stats for the input aseg (plus some processing)
    # cmd="recon-all -subject $subject -segstats $hiresflag $fsthreads"
    if [[ "$segstats_legacy" == "true" ]] ; then
      cmda=($python "$FASTSURFER_HOME/FastSurferCNN/mri_brainvol_stats.py"
            --subject "$subject")
      run_it "$LF" "${cmda[@]}"

      cmda=($python "$FASTSURFER_HOME/FastSurferCNN/mri_segstats.py" --seed 1234
            --seg "$mdir/aseg.mgz" --sum "$statsdir/aseg.stats" --pv "$mdir/norm.mgz"
            "--in-intensity-name" norm "--in-intensity-units" MR --subject "$subject"
            --surf-wm-vol --ctab "$FREESURFER_HOME/ASegStatsLUT.txt" --etiv
            --threads "$threads")
#      cmd="$python $FASTSURFER_HOME/FastSurferCNN/mri_segstats.py --seed 1234 --seg $mdir/wmparc.mgz --sum $statsdir/wmparc.stats --pv $mdir/norm.mgz --in-intensity-name norm --in-intensity-units MR --subject $subject --surf-wm-vol --ctab $FREESURFER_HOME/WMParcStatsLUT.txt --etiv"
    else
      # calculate brainvol stats and aseg stats with segstats.py
      cmda=($python "$FASTSURFER_HOME/FastSurferCNN/segstats.py" --sid "$subject"
            --segfile "$mdir/aseg.mgz" --segstatsfile "$statsdir/aseg.stats"
            --pvfile "$mdir/norm.mgz" --normfile "$mdir/norm.mgz" --threads "$threads"
            # --excl-ctxgmwm: exclude Left/Right WM / Cortex despite ASegStatsLUT.txt
            --excludeid 0 2 3 41 42
            --lut "$FREESURFER_HOME/ASegStatsLUT.txt" --empty
            measures --compute "BrainSeg" "BrainSegNotVent" "VentricleChoroidVol"
                               "lhCortex" "rhCortex" "Cortex" "lhCerebralWhiteMatter"
                               "rhCerebralWhiteMatter" "CerebralWhiteMatter"
                               "SubCortGray" "TotalGray" "SupraTentorial"
                               "SupraTentorialNotVent" "Mask($mask)"
                               "BrainSegVol-to-eTIV" "MaskVol-to-eTIV")
      if [ "$long" == "0" ] ; then
        # in long we do not have orig_nofix for surface hole computation as surfaces
        # are inherited from base/template
        cmda+=("lhSurfaceHoles" "rhSurfaceHoles" "SurfaceHoles")
      fi
      cmda+=("EstimatedTotalIntraCranialVol")
      run_it "$LF" "${cmda[@]}"
      echo "Extract the brainvol stats section from segstats output." | tee -a "$LF"
      # ... so stats/brainvol.stats also exists (but it is slightly different
#      cmd="recon-all -subject $subject -segstats $hiresflag $fsthreads"
#      RunIt "$cmd" "$LF"

      # this call is only "required" to "compute" brainvol.stats, so --normfile/--pvfile
      # are not required
      cmda=($python "$FASTSURFER_HOME/FastSurferCNN/segstats.py" --sid "$subject"
            --segfile "$mdir/aseg.mgz" --pvfile "$mdir/norm.mgz"
            --measure_only --threads "$threads" --segstatsfile "$statsdir/brainvol.stats"
            measures --file "$statsdir/aseg.stats"
                     --import "BrainSeg" "BrainSegNotVent" "SupraTentorial"
                              "SupraTentorialNotVent" "SubCortGray" "lhCortex" "rhCortex"
                              "Cortex" "TotalGray" "lhCerebralWhiteMatter"
                              "rhCerebralWhiteMatter" "CerebralWhiteMatter" "Mask"
                     --compute "SupraTentorialNotVentVox" "BrainSegNotVentSurf"
                               "VentricleChoroidVol")
    fi
    run_it "$LF" "${cmda[@]}"
  fi

# ============================= MAPPED-WMPARC =========================================
{
  echo ""
  echo "===================== Creating wmparc from mapped ======================="
  echo ""
} | tee -a "$LF"

  if [[ "$segstats_legacy" == "true" ]] ; then
    # 1m 11sec also create stats for aseg.presurf.hypos (which is basically the aseg derived from the input with CC and
    # hypos) difference between this and the surface improved one above are probably tiny, so the surface improvement
    # above can probably be skipped to save time
    cmda=($python "$FASTSURFER_HOME/FastSurferCNN/mri_segstats.py" --seed 1234
          --seg "$mdir/aseg.presurf.hypos.mgz" --sum "$statsdir/aseg.presurf.hypos.stats"
          --pv "$mdir/norm.mgz" --empty --brainmask "$mdir/brainmask.mgz"
          --brain-vol-from-seg --excludeid 0 --excl-ctxgmwm --supratent --subcortgray
          "--in" "$mdir/norm.mgz" "--in-intensity-name" norm "--in-intensity-units" MR
          --etiv --surf-wm-vol --surf-ctx-vol --totalgray --euler
          --ctab "$FREESURFER_HOME/ASegStatsLUT.txt" --subject "$subject")
  else
    # segstats.py version of the mri_segstats call
    cmda=($python "$FASTSURFER_HOME/FastSurferCNN/segstats.py" --sid "$subject"
          --segfile "$mdir/aseg.presurf.hypos.mgz" --normfile "$mdir/norm.mgz"
          --pvfile "$mdir/norm.mgz" --segstatsfile "$statsdir/aseg.presurf.hypos.stats"
          # --excl-ctxgmwm: exclude Left/Right WM / Cortex despite ASegStatsLUT.txt
          --excludeid 0 2 3 41 42
          --lut "$FREESURFER_HOME/ASegStatsLUT.txt" --threads "$threads" --empty
          --volume_precision 1
          measures --file "$statsdir/aseg.stats" --import "all")
  fi
  run_it "$LF" "${cmda[@]}"
  # -wmparc based on mapped aparc labels (from input asegdkt_segfile) (1min40sec) needs ribbon and we need to point it to aparc.mapped:
  cmd="mri_surf2volseg --o $mdir/wmparc.DKTatlas.mapped.mgz --label-wm --i $mdir/aparc.DKTatlas+aseg.mapped.mgz --threads $threads --lh-annot $ldir/lh.aparc.DKTatlas.mapped.annot 3000 --lh-cortex-mask $ldir/lh.cortex.label --lh-white $sdir/lh.white --lh-pial $sdir/lh.pial --rh-annot $ldir/rh.aparc.DKTatlas.mapped.annot 4000 --rh-cortex-mask $ldir/rh.cortex.label --rh-white $sdir/rh.white --rh-pial $sdir/rh.pial"
  RunIt "$cmd" "$LF"

  # takes a few mins
  #cmd="mri_segstats --seed 1234 --seg $mdir/wmparc.DKTatlas.mapped.mgz --sum $mdir/../stats/wmparc.DKTatlas.mapped.stats --pv $mdir/norm.mgz --excludeid 0 --brainmask $mdir/brainmask.mgz --in $mdir/norm.mgz --in-intensity-name norm --in-intensity-units MR --subject $subject --surf-wm-vol --ctab $FREESURFER_HOME/WMParcStatsLUT.txt"
  if [[ "$segstats_legacy" == "true" ]] ; then
    cmda=($python "$FASTSURFER_HOME/FastSurferCNN/mri_segstats.py"
          --seed 1234 --seg "$mdir/wmparc.DKTatlas.mapped.mgz"
          --sum "$statsdir/wmparc.DKTatlas.mapped.stats" --pv "$mdir/norm.mgz"
          --excludeid 0 --brainmask "$mdir/brainmask.mgz" "--in" "$mdir/norm.mgz"
          "--in-intensity-name" norm "--in-intensity-units" MR
          --subject "$subject" --surf-wm-vol
          --ctab "$FREESURFER_HOME/WMParcStatsLUT.txt")
  else
    #
    cmda=($python "$FASTSURFER_HOME/FastSurferCNN/segstats.py"
          --sid "$subject" --sd "$SUBJECTS_DIR" --pvfile "$mdir/norm.mgz"
          --segfile "$mdir/wmparc.DKTatlas.mapped.mgz" --normfile "$mdir/norm.mgz"
          --lut "$FREESURFER_HOME/WMParcStatsLUT.txt" --threads "$threads"
          --segstatsfile "$statsdir/wmparc.DKTatlas.mapped.stats" --empty
          --volume_precision 1
          measures --file "$statsdir/brainvol.stats" --import "Mask"
                   "VentricleChoroidVol" "rhCerebralWhiteMatter" "lhCerebralWhiteMatter"
                   "CerebralWhiteMatter")
  fi
  run_it "$LF" "${cmda[@]}"


# ============================= FASTSURFER - SYMLINKS =========================================

  # Create symlinks for downstream analysis (sub-segmentations, TRACULA, etc.)
  if [ "$fsaparc" == "0" ] ; then
    # Symlink of aparc.DKTatlas+aseg.mapped.mgz
    pushd "$mdir" > /dev/null || (echo "Could not cd to $mdir" ; exit 1)
      softlink_or_copy "aparc.DKTatlas+aseg.mapped.mgz" "aparc.DKTatlas+aseg.mgz" "$LF"
      softlink_or_copy "aparc.DKTatlas+aseg.mapped.mgz" "aparc+aseg.mgz" "$LF"

      # Symlink of wmparc.mapped
      softlink_or_copy "wmparc.DKTatlas.mapped.mgz" "wmparc.mgz" "$LF"
    popd > /dev/null || ( echo "Could not popd" ; exit 1 )

      # Symbolic link for mapped surface parcellations
    pushd "$ldir" > /dev/null || (echo "Could not cd to $ldir" ; exit 1)
      softlink_or_copy "lh.aparc.DKTatlas.mapped.annot" "lh.aparc.DKTatlas.annot" "$LF"
      softlink_or_copy "rh.aparc.DKTatlas.mapped.annot" "rh.aparc.DKTatlas.annot" "$LF"
    popd > /dev/null || ( echo "Could not popd" ; exit 1 )
  fi


# ============================= BALABELS =========================================

  # balabels need sphere.reg
  if [ "$fssurfreg" == "1" ] ; then
    # can be produced if surf registration exists
    #cmd="recon-all -subject $subject -balabels $hiresflag $fsthreads"
    #RunIt "$cmd" "$LF"
    # here we run our version of balabels: mapping and annot creation is very fast
    # time is used in mris_anatomical_stats (called 4 times, BA and BA-thresh for each hemi)
    cmd="$python ${binpath}/fs_balabels.py --sd $SUBJECTS_DIR --sid $subject"
    RunIt "$cmd" "$LF"
  fi

fi # not base run


# Collect info
EndTime=$(date)
tSecEnd=$(date '+%s')
tRunHours=$(printf %6.3f "$(bc <<< "($tSecEnd - $tSecStart) / 3600")")

{
  echo ""
  echo "================= DONE ========================================================="
  echo ""

  echo "Started at $StartTime"
  echo "Ended   at $EndTime"
  echo "#@#%# recon-surf-run-time-hours $tRunHours"
} | tee -a "$LF"

# Create the Done File
{
  echo "------------------------------"
  echo "SUBJECT $subject"
  echo "START_TIME $StartTime"
  echo "END_TIME $EndTime"
  echo "RUNTIME_HOURS $tRunHours"
  # id -n sends an error message in docker (no user name), fall back to the USER environment variable or
  username=$(id -un 2>&1)
  if echo "$username" | grep -q "^id: " ; then
    if [[ -n "$USER" ]] ; then username="$USER"
    else username="$(id -u)"
    fi
  fi
  echo "USER $username"
  echo "HOST $(hostname)"
  echo "PROCESSOR $(uname -m)"
  echo "OS $(uname -s)"
  echo "UNAME $(uname -a)"
  echo "VERSION $VERSION"
  echo "CMDPATH $0"
  echo "CMDARGS ${inputargs[*]}"
} > "$DoneFile"
echo "recon-surf.sh $subject finished without error at $(date)" | tee -a "$LF"

cmd="$python ${binpath}utils/extract_recon_surf_time_info.py -i $LF -o $SUBJECTS_DIR/$subject/scripts/recon-surf_times.yaml"
RunIt "$cmd" "/dev/null"
