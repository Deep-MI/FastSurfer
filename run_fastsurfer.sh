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

fastsurfercnndir="$FASTSURFER_HOME/FastSurferCNN"
cerebnetdir="$FASTSURFER_HOME/CerebNet"
hypvinndir="$FASTSURFER_HOME/HypVINN"
reconsurfdir="$FASTSURFER_HOME/recon_surf"

# Regular flags defaults
subject=""
sd="$SUBJECTS_DIR"
t1=""
t2=""
merged_segfile=""
cereb_segfile=""
asegdkt_segfile=""
asegdkt_segfile_default="\$SUBJECTS_DIR/\$SID/mri/aparc.DKTatlas+aseg.deep.mgz"
asegdkt_statsfile=""
cereb_statsfile=""
cereb_flags=()
hypo_segfile=""
hypo_statsfile=""
hypvinn_flags=()
conformed_name=""
conformed_name_t2=""
norm_name=""
norm_name_t2=""
seg_log=""
run_talairach_registration="false"
atlas3T="false"
viewagg="auto"
device="auto"
batch_size="1"
run_seg_pipeline="1"
run_biasfield="1"
run_surf_pipeline="1"
surf_flags=()
vox_size="min"
run_asegdkt_module="1"
run_cereb_module="1"
run_hypvinn_module="1"
threads="1"
# python3.10 -s excludes user-directory package inclusion
python="python3.10 -s"
allow_root=()
version_and_quit=""
base=0                # flag for longitudinal template (base) run
long=0                # flag for longitudinal time point run
baseid=""             # baseid for logitudinal time point run


function usage()
{
#  --merged_segfile <filename>
#                          Name of the segmentation file, which includes all labels
#                            (currently similar to aparc+aseg). When using
#                            FastSurfer, this segmentation is already conformed,
#                            since inference is always based on a conformed image.
#                            Currently, this is the same as the aparc+aseg and just
#                            a symlink to the asegdkt_segfile.
#                            Requires an ABSOLUTE Path! Default location:
#                            \$SUBJECTS_DIR/\$sid/mri/fastsurfer.merged.mgz
# --asegdkt_segfile <name> If not provided,
#                            this intermediate DL-based segmentation will not be
#                            stored, but only the merged segmentation will be stored
#                            (see --merged_segfile <filename>).
cat << EOF

Usage: run_fastsurfer.sh --sid <sid> --sd <sdir> --t1 <t1_input> [OPTIONS]

run_fastsurfer.sh takes a T1 full head image and creates:
     (i)  a segmentation using FastSurferVINN (equivalent to FreeSurfer
          aparc.DKTatlas+aseg.mgz)
     (ii) surfaces, thickness etc as a FS subject dir using recon-surf

FLAGS:

  --fs_license <license>  Path to FreeSurfer license key file. Register at
                            https://surfer.nmr.mgh.harvard.edu/registration.html
                            for free to obtain it if you do not have FreeSurfer
                            installed already
  --sid <subjectID>       Subject ID to create directory inside \$SUBJECTS_DIR
  --sd  <subjects_dir>    Output directory \$SUBJECTS_DIR (or pass via env var)
  --t1  <T1_input>        T1 full head input (not bias corrected). Requires an
                            ABSOLUTE Path!
  --asegdkt_segfile <filename>
                          Name of the segmentation file, which includes the
                          aparc+DKTatlas-aseg segmentations.
                          Requires an ABSOLUTE Path! Default location:
                          \$SUBJECTS_DIR/\$sid/mri/aparc.DKTatlas+aseg.deep.mgz
  --vox_size <0.7-1|min>  Forces processing at a specific voxel size.
                            If a number between 0.7 and 1 is specified (below
                            is experimental) the T1w image is conformed to
                            that voxel size and processed.
                            If "min" is specified (default), the voxel size is
                            read from the size of the minimal voxel size
                            (smallest per-direction voxel size) in the T1w
                            image:
                              If the minimal voxel size is bigger than 0.98mm,
                                the image is conformed to 1mm isometric.
                              If the minimal voxel size is smaller or equal to
                                0.98mm, the T1w image will be conformed to
                                isometric voxels of that voxel size.
                            The voxel size (whether set manually or derived)
                            determines whether the surfaces are processed with
                            highres options (below 1mm) or not.
  --version <info>        Print version information and exit; <info> is optional.
                            <info> may be empty, just prints the version number,
                            +git_branch also prints the current branch, and any
                            combination of +git, +checkpoints, +pip to print
                            additional for the git status, the checkpoints and
                            installed python packages.
  -h --help               Print Help

  PIPELINES:
  By default, both the segmentation and the surface pipelines are run.

SEGMENTATION PIPELINE:
  --seg_only              Run only FastSurferVINN (generate segmentation, do not
                            run surface pipeline)
  --seg_log <seg_log>     Log-file for the segmentation (FastSurferVINN, CerebNet,
                            HypVINN)
                            Default: \$SUBJECTS_DIR/\$sid/scripts/deep-seg.log
  --conformed_name <conf.mgz>
                          Name of the file in which the conformed input
                            image will be saved. Requires an ABSOLUTE Path!
                            Default location:
                            \$SUBJECTS_DIR/\$sid/mri/orig.mgz.
  --no_biasfield          Create a bias field corrected image and enable the
                            calculation of partial volume-corrected stats-files.
  --norm_name             Name of the biasfield corrected image
                            Default location:
                            \$SUBJECTS_DIR/\$sid/mri/orig_nu.mgz
  --tal_reg               Perform the talairach registration for eTIV estimates
                            in --seg_only stream and stats files (is affected by
                            the --3T flag, see below).

  MODULES:
  By default, all modules are run.

  ASEGDKT MODULE:
  --no_asegdkt            Skip the asegdkt segmentation (aseg+aparc/DKT segmentation)
  --asegdkt_segfile <filename>
                          Name of the segmentation file, which includes the
                            aseg+aparc/DKTatlas segmentations.
                            Requires an ABSOLUTE Path! Default location:
                            \$SUBJECTS_DIR/\$sid/mri/aparc.DKTatlas+aseg.deep.mgz
  --no_biasfield          Deactivate the calculation of partial volume-corrected
                            statistics.

  CEREBELLUM MODULE:
  --no_cereb              Skip the cerebellum segmentation (CerebNet segmentation)
  --asegdkt_segfile <seg_input>
                          Name of the segmentation file (similar to aparc+aseg)
                            for cerebellum localization (typically the output of the
                            APARC module (see above). Requires an ABSOLUTE Path!
                            Default location:
                            \$SUBJECTS_DIR/\$sid/mri/aparc.DKTatlas+aseg.deep.mgz
  --cereb_segfile <seg_output>
                          Name of DL-based segmentation file of the cerebellum.
                            This segmentation is always at 1mm isotropic
                            resolution, since inference is always based on a
                            1mm conformed image, if the conformed image is *NOT*
                            already an 1mm image, an additional conformed image
                            at 1mm will be stored at the --conformed_name, but
                            with an additional file suffix of ".1mm".
                            Requires an ABSOLUTE Path! Default location:
                            \$SUBJECTS_DIR/\$sid/mri/cerebellum.CerebNet.nii.gz
  --no_biasfield          Deactivate the calculation of partial volume-corrected
                            statistics.

  HYPOTHALAMUS MODULE (HypVINN):
  --no_hypothal           Skip the hypothalamus segmentation.
  --no_biasfield          This option implies --no_hypothal, as the hypothalamus
                            sub-segmentation requires biasfield-corrected images.
  --t2 <T2_input>         *Optional* T2 full head input (does not have to be bias
                            corrected, a mandatory biasfield correction step is
                            performed). Requires an ABSOLUTE Path!
  --reg_mode <none|coreg|robust>
                          Ignored, if no T2 image is passed.
                            Specifies the registration method used to register T1
                            and T2 images. Options are 'coreg' (default) for
                            mri_coreg, 'robust' for mri_robust_register, and 'none'
                            to skip registration (this requires T1 and T2 are
                            externally co-registered).
  --qc_snap               Create QC snapshots in \$SUBJECTS_DIR/\$sid/qc_snapshots
                            to simplify the QC process.

SURFACE PIPELINE:
  --surf_only             Run surface pipeline only. The segmentation input has
                            to exist already in this case.
  --3T                    Use the 3T atlas for talairach registration (gives better
                            etiv estimates for 3T MR images, default: 1.5T atlas).
  --parallel              Run both hemispheres in parallel
  --threads <int>         Set openMP and ITK threads to <int>

Resource Options:
  --device                Set device on which inference should be run ("cpu" for
                            CPU, "cuda" for Nvidia GPU, or pass specific device,
                            e.g. cuda:1), default check GPU and then CPU
  --viewagg_device <str>  Define where the view aggregation should be run on.
                            Can be "auto" or a device (see --device). By default,
                            the program checks if you have enough memory to run
                            the view aggregation on the gpu. The total memory is
                            considered for this decision. If this fails, or you
                            actively overwrote the check with setting with "cpu"
                            view agg is run on the cpu. Equivalently, if you
                            pass a different device, view agg will be run on that
                            device (no memory check will be done).
  --batch <batch_size>    Batch size for inference. Default: 1
  --py <python_cmd>       Command for python, used in both pipelines.
                            Default: "$python"
                            (-s: do no search for packages in home directory)

 Dev Flags:
  --ignore_fs_version     Switch on to avoid check for FreeSurfer version.
                            Program will terminate if the supported version
                            (see recon-surf.sh) is not sourced. Can be used for
                            testing dev versions.
  --fstess                Switch on mri_tesselate for surface creation (default:
                            mri_mc)
  --fsqsphere             Use FreeSurfer iterative inflation for qsphere
                            (default: spectral spherical projection)
  --fsaparc               Additionally create FS aparc segmentations and ribbon.
                            Skipped by default (--> DL prediction is used which
                            is faster, and usually these mapped ones are fine)
  --no_fs_T1              Do not generate T1.mgz (normalized nu.mgz included in
                            standard FreeSurfer output) and create brainmask.mgz
                            directly from norm.mgz instead. Saves 1:30 min.
  --no_surfreg             Do not run Surface registration with FreeSurfer (for
                            cross-subject correspondence), Not recommended, but
                            speeds up processing if you e.g. just need the
                            segmentation stats!
  --allow_root            Allow execution as root user.

 Longitudinal Flags (non-expert users should use long_fastsurfers.sh for
                     sequential processing of longitudinal data):
  --base                  Longitudinal template (base) processing.
                            Only ASEGDKT in segmentation and differences in the
                            surface module. Requires longitudinal template
                            preparation (recon-surf/long_prepare_template.sh) to
                            be completed beforehand! No T2 can be passed. Also
                            no T1 is explicitly passed, as it is taken from
                            within the prepared template directory.
  --long <baseid>         Longitudinal time point processing.
                            Requires the base (template) already exists in the
                            same SUBJECTS_DIR under the SID <baseid>.
                            Processing is identical to the regular cross-sectional
                            pipeline for segmentation. Surface module skips
                            many steps and initializes from subject template.
                            No T2 can be passed. Also no T1 is explicitly passed,
                            as it is taken from the prepared template directory.


REFERENCES:

If you use this for research publications, please cite:

Henschel L, Conjeti S, Estrada S, Diers K, Fischl B, Reuter M, FastSurfer - A
 fast and accurate deep learning based neuroimaging pipeline, NeuroImage 219
 (2020), 117012. https://doi.org/10.1016/j.neuroimage.2020.117012

Henschel L*, Kuegler D*, Reuter M. (*co-first). FastSurferVINN: Building
 Resolution-Independence into Deep Learning Segmentation Methods - A Solution
 for HighRes Brain MRI. NeuroImage 251 (2022), 118933. 
 http://dx.doi.org/10.1016/j.neuroimage.2022.118933

For cerebellum sub-segmentation:
Faber J*, Kuegler D*, Bahrami E*, et al. (*co-first). CerebNet: A fast and
 reliable deep-learning pipeline for detailed cerebellum sub-segmentation.
 NeuroImage 264 (2022), 119703.
 https://doi.org/10.1016/j.neuroimage.2022.119703

For hypothalamus sub-segemntation:
Estrada S, Kuegler D, Bahrami E, Xu P, Mousa D, Breteler MMB, Aziz NA, Reuter M.
 FastSurfer-HypVINN: Automated sub-segmentation of the hypothalamus and adjacent
 structures on high-resolutional brain MRI. Imaging Neuroscience 2023; 1 1–32.
 https://doi.org/10.1162/imag_a_00034

For longitudinal processing:
Reuter M, Schmansky NJ, Rosas HD, Fischl B. Within-subject template estimation
 for unbiased longitudinal image analysis, NeuroImage 61:4 (2012).
 https://doi.org/10.1016/j.neuroimage.2012.02.084

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
  ##############################################################
  # general options
  ##############################################################
  --fs_license)
    if [[ -f "$1" ]]
    then
      export FS_LICENSE="$1"
    else
      echo "ERROR: Provided FreeSurfer license file $1 could not be found. Make sure to provide the full path and name. Exiting..."
      exit 1
    fi
    shift # past value
    ;;

  # options that *just* set a flag
  #=============================================================
  --allow_root) allow_root=("--allow_root") ;;
  # options that set a variable
  --sid) subject="$1" ; shift ;;
  --sd) sd="$1" ; shift ;;
  --t1) t1="$1" ; shift ;;
  --t2) t2="$1" ; shift ;;
  --seg_log) seg_log="$1" ; shift ;;
  --conformed_name) conformed_name="$1" ; shift ;;
  --norm_name) norm_name="$1" ; shift ;;
  --norm_name_t2) norm_name_t2="$1" ; shift ;;
  --seg|--asegdkt_segfile|--aparc_aseg_segfile)
    if [[ "$key" != "--asegdkt_segfile" ]]
    then
      echo "WARNING: --$key <filename> is deprecated and will be removed, use --asegdkt_segfile <filename>."
    fi
    asegdkt_segfile="$1"
    shift # past value
    ;;
  --vox_size) vox_size="$1" ; shift ;;
  # --3t: both for surface pipeline and the --tal_reg flag
  --3t) surf_flags=("${surf_flags[@]}" "--3T") ; atlas3T="true" ;;
  --threads) threads="$1" ; shift ;;
  --py) python="$1" ; shift ;;
  -h|--help) usage ; exit ;;
  --version)
    if [[ "$#" -lt 1 ]] || [[ "$1" =~ ^-- ]]; then
      # no more args or next arg starts with --
      version_and_quit="1"
    else
      case "$(echo "$1" | tr '[:upper:]' '[:lower:]')" in
        all) version_and_quit="+checkpoints+git+pip" ;;
        +*) version_and_quit="$1" ;;
        *) echo "ERROR: Invalid option for --version: '$1', must be 'all' or [+checkpoints][+git][+pip]"
          exit 1
          ;;
      esac
      shift
    fi
    ;;

  ##############################################################
  # seg-pipeline options
  ##############################################################

  # common options for seg
  #=============================================================
  --surf_only) run_seg_pipeline="0" ;;
  --no_biasfield) run_biasfield="0" ;;
  --tal_reg) run_talairach_registration="true" ;;
  --device) device="$1" ; shift ;;
  --batch) batch_size="$1" ; shift ;;
  --viewagg_device|--run_viewagg_on)
    if [[ "$key" == "--run_viewagg_on" ]]
    then
      echo "WARNING: --run_viewagg_on (cpu|gpu|check) is deprecated and will be removed, use --viewagg_device <device|auto>."
    fi
    case "$1" in
      check)
        echo "WARNING: the option \"check\" is deprecated for --viewagg_device <device>, use \"auto\"."
        viewagg="auto"
        ;;
      gpu) viewagg="cuda" ;;
      *) viewagg="$1" ;;
    esac
    shift # past value
    ;;
  --no_cuda)
    echo "WARNING: --no_cuda is deprecated and will be removed, use --device cpu."
    device="cpu"
    ;;

  # asegdkt module options
  #=============================================================
  --no_asegdkt|--no_aparc)
    if [[ "$key" == "--no_aparc" ]]
    then
      echo "WARNING: --no_aparc is deprecated and will be removed, use --no_asegdkt."
    fi
    run_asegdkt_module="0"
    ;;
  --asegdkt_statsfile) asegdkt_statsfile="$1" ; shift ;;
  --aseg_segfile) aseg_segfile="$1" ; shift ;;
  --mask_name) mask_name="$1" ; shift ;;
  --merged_segfile) merged_segfile="$1" ; shift ;;

  # cereb module options
  #=============================================================
  --no_cereb) run_cereb_module="0" ;;
  # several options that set a variable
  --cereb_segfile) cereb_segfile="$1" ; shift ;;
  --cereb_statsfile) cereb_statsfile="$1" ; shift ;;

  # hypothal module options
  #=============================================================
  --no_hypothal) run_hypvinn_module="0" ;;
  # several options that set a variable
  --hypo_segfile) hypo_segfile="$1" ; shift ;;
  --hypo_statsfile) hypo_statsfile="$1" ; shift ;;
  --reg_mode)
    mode=$(echo "$1" | tr "[:upper:]" "[:lower:]")
    if [[ "$mode" =~ ^(none|coreg|robust)$ ]] ; then
      hypvinn_flags+=(--regmode "$mode")
    else
      echo "Invalid --reg_mode option, must be 'none', 'coreg' or 'robust'."
      exit 1
    fi
    shift # past value
    ;;
  # several options that set a variable
  --qc_snap) hypvinn_flags+=(--qc_snap) ;;

  ##############################################################
  # surf-pipeline options
  ##############################################################
  --seg_only) run_surf_pipeline="0" ;;
  # several flag options that are *just* passed through to recon-surf.sh
  --fstess|--fsqsphere|--fsaparc|--no_surfreg|--parallel|--ignore_fs_version)
    surf_flags=("${surf_flags[@]}" "$key")
    ;;
  --no_fs_t1) surf_flags=("${surf_flags[@]}" "--no_fs_T1") ;;

  # temporary segstats development flag
  --segstats_legacy)
    surf_flags=("${surf_flags[@]}" "$key")
    ;;

  ##############################################################
  # longitudinal options
  ##############################################################
  --base) base=1 ; run_cereb_module="0" ; run_hypvinn_module="0" ; surf_flags=("${surf_flags[@]}" "--base") ;;
  --long) long=1 ; baseid="$1" ; surf_flags=("${surf_flags[@]}" "--long" "$1") ; shift ;;

  *)    # unknown option
    # if not empty arguments, error & exit
    if [[ "$key" != "" ]] ; then echo "ERROR: Flag '$key' unrecognized." ;  exit 1 ; fi
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

# make sure FastSurfer is in the PYTHONPATH
if [[ "$PYTHONPATH" == "" ]]
then
  export PYTHONPATH="$FASTSURFER_HOME"
else
  export PYTHONPATH="$FASTSURFER_HOME:$PYTHONPATH"
fi

########################################## VERSION AND QUIT HERE ########################################
version_args=()
if [[ -f "$FASTSURFER_HOME/BUILD.info" ]]
then
  version_args=(--build_cache "$FASTSURFER_HOME/BUILD.info" --prefer_cache)
fi

if [[ -n "$version_and_quit" ]]
then
  # if version_and_quit is 1, it should only print the version number+git branch
  if [[ "$version_and_quit" != "1" ]]
  then
    version_args=("${version_args[@]}" --sections "$version_and_quit")
  fi
  $python "$FASTSURFER_HOME/FastSurferCNN/version.py" "${version_args[@]}"
  exit
fi

# make sure the python  executable is valid and found
if [[ -z "$(which "${python/ */}")" ]]; then
  echo "Cannot find the python interpreter ${python/ */}."
  exit 1
fi

# Warning if run as root user
if [[ "${#allow_root}" == 0 ]] && [[ "$(id -u)" == "0" ]]
then
  echo "You are trying to run '$0' as root. We advice to avoid running FastSurfer as root, "
  echo "because it will lead to files and folders created as root."
  echo "If you are running FastSurfer in a docker container, you can specify the user with "
  echo "'-u \$(id -u):\$(id -g)' (see https://docs.docker.com/engine/reference/run/#user)."
  echo "If you want to force running as root, you may pass --allow_root to run_fastsurfer.sh."
  exit 1;
fi

# CHECKS

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

if [[ -z "$subject" ]]
then
  echo "ERROR: must supply subject name via --sid"
  exit 1;
fi

if [[ -z "$merged_segfile" ]]
then
  merged_segfile="${sd}/${subject}/mri/fastsurfer.merged.mgz"
fi

if [[ -z "$asegdkt_segfile" ]]
then
  asegdkt_segfile="${sd}/${subject}/mri/aparc.DKTatlas+aseg.deep.mgz"
fi

if [[ -z "$aseg_segfile" ]]
then
  aseg_segfile="${sd}/${subject}/mri/aseg.auto_noCCseg.mgz"
fi

if [[ -z "$asegdkt_statsfile" ]]
then
  asegdkt_statsfile="${sd}/${subject}/stats/aseg+DKT.stats"
fi

if [[ -z "$cereb_segfile" ]]
then
  cereb_segfile="${sd}/${subject}/mri/cerebellum.CerebNet.nii.gz"
fi

if [[ -z "$cereb_statsfile" ]]
then
  cereb_statsfile="${sd}/${subject}/stats/cerebellum.CerebNet.stats"
fi

if [[ -z "$hypo_segfile" ]]
then
  hypo_segfile="${sd}/${subject}/mri/hypothalamus.HypVINN.nii.gz"
fi

if [[ -z "$hypo_statsfile" ]]
then
  hypo_statsfile="${sd}/${subject}/stats/hypothalamus.HypVINN.stats"
fi

if [[ -z "$mask_name" ]]
then
  mask_name="${sd}/${subject}/mri/mask.mgz"
fi

if [[ -z "$conformed_name" ]]
then
  conformed_name="${sd}/${subject}/mri/orig.mgz"
fi

if [[ -z "$conformed_name_t2" ]]
  then
    conformed_name_t2="${sd}/${subject}/mri/T2orig.mgz"
fi

if [[ -z "$norm_name" ]]
then
  norm_name="${sd}/${subject}/mri/orig_nu.mgz"
fi

if [[ -z "$norm_name_t2" ]]
then
  norm_name_t2="${sd}/${subject}/mri/T2_nu.mgz"
fi

if [[ -z "$seg_log" ]]
then
  seg_log="${sd}/${subject}/scripts/deep-seg.log"
fi

if [[ -z "$build_log" ]]
then
  build_log="${sd}/${subject}/scripts/build.log"
fi

if [[ -n "$t2" ]]
then
  if [[ ! -f "$t2" ]]
    then
      echo "ERROR: T2 file $t2 does not exist!"
      exit 1;
  fi
  copy_name_T2="${sd}/${subject}/mri/orig/T2.001.mgz"
fi

if [[ -z "$PYTHONUNBUFFERED" ]]
then
  export PYTHONUNBUFFERED=0
fi

# check the vox_size setting
if [[ "$vox_size" =~ ^[0-9]+([.][0-9]+)?$ ]]
then
  # a number
  if (( $(echo "$vox_size < 0" | bc -l) || $(echo "$vox_size > 1" | bc -l) ))
  then
    echo "ERROR: negative voxel sizes and voxel sizes beyond 1 are not supported."
    exit 1;
  elif (( $(echo "$vox_size < 0.7" | bc -l) ))
  then
    echo "WARNING: support for voxel sizes smaller than 0.7mm iso. is experimental."
  fi
elif [[ "$vox_size" != "min" ]]
then
  # not a number or "min"
  echo "Invalid option for --vox_size, only a number or 'min' are valid."
  exit 1;
fi

#if [[ "${asegdkt_segfile: -3}" != "${merged_segfile: -3}" ]]
#  then
#    # This is because we currently only do a symlink
#    echo "ERROR: Specified segmentation outputs do not have same file type."
#    echo "You passed --asegdkt_segfile ${asegdkt_segfile} and --merged_segfile ${merged_segfile}."
#    echo "Make sure these have the same file-format and adjust the names passed to the flags accordingly!"
#    exit 1;
#fi

if [[ "${asegdkt_segfile: -3}" != "${conformed_name: -3}" ]]
then
  echo "ERROR: Specified segmentation output and conformed image output do not have same file type."
  echo "You passed --asegdkt_segfile ${asegdkt_segfile} and --conformed_name ${conformed_name}."
  echo "Make sure these have the same file-format and adjust the names passed to the flags accordingly!"
  exit 1;
fi

if [[ "$run_surf_pipeline" == "1" ]] && { [[ "$run_asegdkt_module" == "0" ]] || [[ "$run_seg_pipeline" == "0" ]]; }
then
  if [[ ! -f "$asegdkt_segfile" ]]
  then
    echo "ERROR: To run the surface pipeline, a whole brain segmentation must already exist."
    echo "You passed --surf_only or --no_asegdkt, but the whole-brain segmentation ($asegdkt_segfile) could not be found."
    echo "If the segmentation is not saved in the default location ($asegdkt_segfile_default), specify the absolute path and name via --asegdkt_segfile"
    exit 1;
  fi
  if [[ ! -f "$conformed_name" ]]
  then
    echo "ERROR: To run the surface pipeline only, a conformed T1 image must already exist."
    echo "You passed --surf_only but the conformed image ($conformed_name) could not be found."
    echo "If the conformed image is not saved in the default location (\$SUBJECTS_DIR/\$SID/mri/orig.mgz),"
    echo "specify the absolute path and name via --conformed_name."
    exit 1;
  fi
fi

if [[ "$run_seg_pipeline" == "1" ]] && { [[ "$run_asegdkt_module" == "0" ]] && [[ "$run_cereb_module" == "1" ]]; }
then
  if [[ ! -f "$asegdkt_segfile" ]]
  then
    echo "ERROR: To run the cerebellum segmentation but no asegdkt, the aseg segmentation must already exist."
    echo "You passed --no_asegdkt but the asegdkt segmentation ($asegdkt_segfile) could not be found."
    echo "If the segmentation is not saved in the default location ($asegdkt_segfile_default), specify the absolute path and name via --asegdkt_segfile"
    exit 1;
  fi
fi


if [[ "$run_surf_pipeline" == "0" ]] && [[ "$run_seg_pipeline" == "0" ]]
then
  echo "ERROR: You specified both --surf_only and --seg_only. Therefore neither part of the pipeline will be run."
  echo "To run the whole FastSurfer pipeline, omit both flags."
  exit 1;
fi

if [[ "$run_surf_pipeline" == "1" ]] || [[ "$run_talairach_registration" == "true" ]]
then
  msg="The surface pipeline and the talairach-registration in the segmentation pipeline require a FreeSurfer License"
  if [[ -z "$FS_LICENSE" ]]
  then
    msg="$msg, but no license was provided via --fs_license or the FS_LICENSE environment variable."
    if [[ "$DO_NOT_SEARCH_FS_LICENSE_IN_FREESURFER_HOME" != "true" ]] && [[ -n "$FREESURFER_HOME" ]]
    then
      echo "WARNING: $msg Checking common license files in \$FREESURFER_HOME."
      for filename in "license.dat" "license.txt" ".license"
      do
        if [[ -f "$FREESURFER_HOME/$filename" ]]
        then
          echo "Trying with '$FREESURFER_HOME/$filename', specify a license with --fs_license to overwrite."
          export FS_LICENSE="$FREESURFER_HOME/$filename"
          break
        fi
      done
      if [[ -z "$FS_LICENSE" ]]; then echo "ERROR: No license found..." ; exit 1 ; fi
    else
      echo "ERROR: $msg"
      exit 1;
    fi
  elif [[ ! -f "$FS_LICENSE" ]]
  then
    echo "ERROR: $msg, but the provided path is not a file: $FS_LICENSE."
    exit 1;
  fi
fi

# checks and t1 setup for longitudinal pipeline
# generally any t1 input per command line is overwritten here
if [[ "$long" == "1" ]] && [[ "$base" == "1" ]]
then
  echo "ERROR: You specified both --long and --base. You need to setup and then run base template first,"
  echo "  before you can run any longitudinal time points."
  exit 1;
fi

if [[ "$base" == "1" ]]
then
  if [ ! -f "$sd/$subject/base-tps.fastsurfer" ] ; then
    echo "ERROR: $subject is either not found in \$SUBJECTS_DIR or it is not a longitudinal template"
    echo "  directory (base), which needs to contain base-tps.fastsurfer file. Please ensure that"
    echo "  the base (template) has been created with long_prepare_template.sh."
    exit 1
  fi
  if [[ -z "$t1" ]] ; then
    echo "WARNING: --t1 was passed but will be overwritten with T1 from base template."
  fi
  # base can only be run with the template image from base-setup:
  t1=$sd/$subject/mri/orig.mgz 
fi

if [[ "$long" == "1" ]]
then
  if [ ! -f "$sd/$baseid/base-tps.fastsurfer" ] ; then
    echo "ERROR: $baseid is either not found in \$SUBJECTS_DIR or it is not a longitudinal template"
    echo "  directory (base), which needs to contain base-tps.fastsurfer file. Please ensure that"
    echo "  the base (template) has been created with long_prepare_template.sh."
    exit 1
  fi
  if ! grep -Fxq "$subject" "$sd/$baseid/base-tps.fastsurfer" ; then
    echo "ERROR: $subject id not found in base-tps.fastsurfer. Please ensure that this time point"
    echo "  was included during creation of the base (template)."
    exit 1
  fi
  if [[ -z "$t1" ]] ; then
    echo "WARNING: --t1 was passed but will be overwritten with T1 in base space."
  fi
  # this is the default longitudinal input from base directory:
  t1="$sd/$baseid/long-inputs/$subject/long_conform.nii.gz"
fi


if [[ "$run_seg_pipeline" == "1" ]] && { [[ -z "$t1" ]] || [[ ! -f "$t1" ]]; }
then
  echo "ERROR: T1 image ($t1) could not be found. Must supply an existing T1 input (full head) via "
  echo "--t1 (absolute path and name) for generating the segmentation."
  echo "NOTES: If running in a container, make sure symlinks are valid!"
  exit 1;
fi


########################################## START ########################################################
mkdir -p "$(dirname "$seg_log")"

source "${reconsurfdir}/functions.sh"

if [[ -f "$seg_log" ]]; then log_existed="true"
else log_existed="false"
fi

VERSION=$($python "$FASTSURFER_HOME/FastSurferCNN/version.py" "${version_args[@]}")
echo "Version: $VERSION" | tee -a "$seg_log"

### IF THE SCRIPT GETS TERMINATED, ADD A MESSAGE
trap "{ echo \"run_fastsurfer.sh terminated via signal at \$(date -R)!\" >> \"$seg_log\" ; }" SIGINT SIGTERM

# create the build log, file with all version info in parallel
printf "%s %s\n%s\n" "$THIS_SCRIPT" "${inputargs[*]}" "$(date -R)" >> "$build_log"
$python "$FASTSURFER_HOME/FastSurferCNN/version.py" --sections all -o "$build_log" --prefer_cache &

if [[ "$run_seg_pipeline" != "1" ]]
then
  echo "Running run_fastsurfer.sh without segmentation ; expecting previous --seg_only run in ${sd}/${subject}" | tee -a "$seg_log"
fi


if [[ "$run_seg_pipeline" == "1" ]]
then
  # "============= Running FastSurferCNN (Creating Segmentation aparc.DKTatlas.aseg.mgz) ==============="
  # use FastSurferCNN to create cortical parcellation + anatomical segmentation into 95 classes.
  echo "Log file for segmentation FastSurferCNN/run_prediction.py" >> "$seg_log"
  { date 2>&1 ; echo "" ; } | tee -a "$seg_log"

  if [[ "$run_asegdkt_module" == "1" ]]
  then
    cmd=($python "$fastsurfercnndir/run_prediction.py" --t1 "$t1"
         --asegdkt_segfile "$asegdkt_segfile" --conformed_name "$conformed_name"
         --brainmask_name "$mask_name" --aseg_name "$aseg_segfile" --sid "$subject"
         --seg_log "$seg_log" --vox_size "$vox_size" --batch_size "$batch_size"
         --viewagg_device "$viewagg" --device "$device" "${allow_root[@]}")
    # specify the subject dir $sd, if asegdkt_segfile explicitly starts with it
    if [[ "$sd" == "${asegdkt_segfile:0:${#sd}}" ]]; then cmd=("${cmd[@]}" --sd "$sd"); fi
    echo_quoted "${cmd[@]}" | tee -a "$seg_log"
    "${cmd[@]}"
    exit_code="${PIPESTATUS[0]}"
    if [[ "${exit_code}" == 2 ]]
    then
      echo "ERROR: FastSurfer asegdkt segmentation failed QC checks." | tee -a "$seg_log"
      exit 1
    elif [[ "${exit_code}" -ne 0 ]]
    then
      echo "ERROR: FastSurfer asegdkt segmentation failed." | tee -a "$seg_log"
      exit 1
    fi
  fi
  if [[ -n "$t2" ]]
  then
    {
      echo "INFO: Copying T2 file to ${copy_name_T2}..."
      cmd=("nib-convert" "$t2" "$copy_name_T2")
      echo_quoted "${cmd[@]}"
      "${cmd[@]}" 2>&1

      echo "INFO: Robust scaling (partial conforming) of T2 image..."
      cmd=($python "${fastsurfercnndir}/data_loader/conform.py" --no_strict_lia
           --no_vox_size --no_img_size "$t2" "$conformed_name_t2")
      echo_quoted "${cmd[@]}"
      "${cmd[@]}" 2>&1
      echo "Done."
    } | tee -a "$seg_log"
  fi

  if [[ "$run_biasfield" == "1" ]]
  then
    {
      # this will always run, since norm_name is set to subject_dir/mri/orig_nu.mgz, if it is not passed/empty
      cmd=($python "${reconsurfdir}/N4_bias_correct.py" "--in" "$conformed_name"
           --rescale "$norm_name" --aseg "$asegdkt_segfile" --threads "$threads")
      echo "INFO: Running N4 bias-field correction"
      echo_quoted "${cmd[@]}"
      "${cmd[@]}" 2>&1
    } | tee -a "$seg_log"
    if [[ "${PIPESTATUS[0]}" -ne 0 ]]
    then
      echo "ERROR: Biasfield correction failed" | tee -a "$seg_log"
      exit 1
    fi

    if [[ "$run_talairach_registration" == "true" ]]
    then
      cmd=("$reconsurfdir/talairach-reg.sh" "$sd/$subject/mri" "$atlas3T" "$seg_log")
      {
        echo "INFO: Running talairach registration"
        echo_quoted "${cmd[@]}"
      } | tee -a "$seg_log"
      "${cmd[@]}"
      if [[ "${PIPESTATUS[0]}" -ne 0 ]]
      then
        echo "ERROR: talairach registration failed" | tee -a "$seg_log"
        exit 1
      fi
    fi

    if [[ "$run_asegdkt_module" ]]
    then
      cmd=($python "${fastsurfercnndir}/segstats.py" --segfile "$asegdkt_segfile"
           --segstatsfile "$asegdkt_statsfile" --normfile "$norm_name"
           --threads "$threads" "${allow_root[@]}" --empty --excludeid 0
           --sd "${sd}" --sid "${subject}"
           --ids 2 4 5 7 8 10 11 12 13 14 15 16 17 18 24 26 28 31 41 43 44 46 47
                 49 50 51 52 53 54 58 60 63 77 251 252 253 254 255 1002 1003 1005
                 1006 1007 1008 1009 1010 1011 1012 1013 1014 1015 1016 1017 1018
                 1019 1020 1021 1022 1023 1024 1025 1026 1027 1028 1029 1030 1031
                 1034 1035 2002 2003 2005 2006 2007 2008 2009 2010 2011 2012 2013
                 2014 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025 2026
                 2027 2028 2029 2030 2031 2034 2035
           --lut "$fastsurfercnndir/config/FreeSurferColorLUT.txt"
           measures --compute "Mask($mask_name)" "BrainSeg" "BrainSegNotVent"
                              "SupraTentorial" "SupraTentorialNotVent"
                              "SubCortGray" "rhCerebralWhiteMatter"
                              "lhCerebralWhiteMatter" "CerebralWhiteMatter"
           # make sure to read white matter hypointensities from the
           )
      if [[ "$run_talairach_registration" == "true" ]]
      then
        cmd=("${cmd[@]}" "EstimatedTotalIntraCranialVol"
             "BrainSegVol-to-eTIV" "MaskVol-to-eTIV")
      fi
      {
        echo_quoted "${cmd[@]}"
        "${cmd[@]}" 2>&1
      } | tee -a "$seg_log"
      if [[ "${PIPESTATUS[0]}" -ne 0 ]]
      then
        echo "ERROR: asegdkt statsfile generation failed" | tee -a "$seg_log"
        exit 1
      fi
    fi
  fi  # [[ "$run_biasfield" == "1" ]]

  if [[ -n "$t2" ]]
  then
    if [[ "$run_biasfield" == "1" ]]
    then
      # ... we have a t2 image, bias field-correct it (save robustly scaled uchar)
      cmd=($python "${reconsurfdir}/N4_bias_correct.py" "--in" "$copy_name_T2"
           --out "$norm_name_t2" --threads "$threads" --uchar)
      {
        echo "INFO: Running N4 bias-field correction of the t2"
        echo_quoted "${cmd[@]}"
      } | tee -a "$seg_log"
      "${cmd[@]}" 2>&1 | tee -a "$seg_log"
      if [[ "${PIPESTATUS[0]}" -ne 0 ]]
      then
        echo "ERROR: T2 Biasfield correction failed" | tee -a "$seg_log"
        exit 1
      fi
    else
      # no biasfield, but a t2 is passed; presumably, this is biasfield corrected
      cmd=($python "${fastsurfercnndir}/data_loader/conform.py" --no_strict_lia
           --no_iso_vox --no_img_size "$t2" "$norm_name_t2")
      {
        echo "INFO: Robustly rescaling $t2 to uchar ($norm_name_t2), which is assumed to already be biasfield corrected."
        echo "WARNING: --no_biasfield is activated, but FastSurfer does not check, if "
        echo "  passed T2 image is properly scaled and typed. T2 needs to be uchar and"
        echo "  robustly scaled (see FastSurferCNN/utils/data_loader/conform.py)!"
      } | tee -a "$seg_log"
      "${cmd[@]}" 2>&1 | tee -a "$seg_log"
    fi
  fi

  if [[ "$run_cereb_module" == "1" ]]
  then
    if [[ "$run_biasfield" == "1" ]]
    then
      cereb_flags=("${cereb_flags[@]}" --norm_name "$norm_name"
                   --cereb_statsfile "$cereb_statsfile")
    else
      echo "INFO: Running CerebNet without generating a statsfile, since biasfield correction deactivated '--no_biasfield'." | tee -a "$seg_log"
    fi

    cmd=($python "$cerebnetdir/run_prediction.py" --t1 "$t1"
         --asegdkt_segfile "$asegdkt_segfile" --conformed_name "$conformed_name"
         --cereb_segfile "$cereb_segfile" --seg_log "$seg_log" --async_io
         --batch_size "$batch_size" --viewagg_device "$viewagg" --device "$device"
         --threads "$threads" "${cereb_flags[@]}" "${allow_root[@]}")
    # specify the subject dir $sd, if asegdkt_segfile explicitly starts with it
    if [[ "$sd" == "${cereb_segfile:0:${#sd}}" ]] ; then cmd=("${cmd[@]}" --sd "$sd"); fi
    echo_quoted "${cmd[@]}" | tee -a "$seg_log"
    "${cmd[@]}"  # no tee, directly logging to $seg_log
    if [[ "${PIPESTATUS[0]}" -ne 0 ]]
    then
      echo "ERROR: Cerebellum Segmentation failed" | tee -a "$seg_log"
      exit 1
    fi
  fi

  if [[ "$run_hypvinn_module" == "1" ]]
  then
        # currently, the order of the T2 preprocessing only is registration to T1w
    cmd=($python "$hypvinndir/run_prediction.py" --sd "${sd}" --sid "${subject}"
         "${hypvinn_flags[@]}" "${allow_root[@]}" --threads "$threads" --async_io
         --batch_size "$batch_size" --seg_log "$seg_log" --device "$device"
         --viewagg_device "$viewagg" --t1)
    if [[ "$run_biasfield" == "1" ]]
    then
      cmd+=("$norm_name")
      if [[ -n "$t2" ]] ; then cmd+=(--t2 "$norm_name_t2"); fi
    else
      echo "WARNING: We strongly recommend to *not* exclude the biasfield (--no_biasfield) with the hypothal module!"
      cmd+=("$t1")
      if [[ -n "$t2" ]] ; then cmd+=(--t2 "$t2"); fi
    fi
    echo_quoted "${cmd[@]}" | tee -a "$seg_log"
    "${cmd[@]}"
    if [[ "${PIPESTATUS[0]}" -ne 0 ]]
    then
      echo "ERROR: Hypothalamus Segmentation failed" | tee -a "$seg_log"
      exit 1
    fi
  fi

#    if [[ ! -f "$merged_segfile" ]]
#      then
#        ln -s -r "$asegdkt_segfile" "$merged_segfile"
#    fi
fi

if [[ "$run_surf_pipeline" == "1" ]]
then
  # ============= Running recon-surf (surfaces, thickness etc.) ===============
  # use recon-surf to create surface models based on the FastSurferCNN segmentation.
  pushd "$reconsurfdir" > /dev/null || exit 1
  echo "cd $reconsurfdir" | tee -a "$seg_log"
  cmd=("./recon-surf.sh" --sid "$subject" --sd "$sd" --t1 "$conformed_name"
       --asegdkt_segfile "$asegdkt_segfile" --threads "$threads" --py "$python"
       "${surf_flags[@]}" "${allow_root[@]}")
  echo_quoted "${cmd[@]}" | tee -a "$seg_log"
  "${cmd[@]}"
  if [[ "${PIPESTATUS[0]}" -ne 0 ]] ; then exit 1 ; fi
  popd > /dev/null || return
fi

########################################## End ########################################################
