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

case $key in
    --fs_license)
    if [[ -f "$2" ]]
    then
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
    --t2)
    t2="$2"
    shift # past argument
    shift # past value
    ;;
    --merged_segfile)
    merged_segfile="$2"
    shift # past argument
    shift # past value
    ;;
    --seg | --asegdkt_segfile | --aparc_aseg_segfile)
    if [[ "$key" == "--seg" ]]
    then
      echo "WARNING: --seg <filename> is deprecated and will be removed, use --asegdkt_segfile <filename>."
    fi
    if [[ "$key" == "--aparc_aseg_segfile" ]]
    then
      echo "WARNING: --aparc_aseg_segfile <filename> is deprecated and will be removed, use --asegdkt_segfile <filename>"
    fi
    asegdkt_segfile="$2"
    shift # past argument
    shift # past value
    ;;
    --asegdkt_statsfile)
    asegdkt_statsfile="$2"
    shift # past argument
    shift # past value
    ;;
    --cereb_segfile)
    cereb_segfile="$2"
    shift # past argument
    shift # past value
    ;;
    --cereb_statsfile)
    cereb_statsfile="$2"
    shift # past argument
    shift # past value
    ;;
    --hypo_segfile)
    hypo_segfile="$2"
    shift # past argument
    shift # past value
    ;;
    --hypo_statsfile)
    hypo_statsfile="$2"
    shift # past argument
    shift # past value
    ;;
    --reg_mode)
    mode=$(echo "$2" | tr "[:upper:]" "[:lower:]")
    if [[ "$mode" =~ ^(none|coreg|robust)$ ]] ; then
      hypvinn_flags+=(--regmode "$mode")
    else
      echo "Invalid --reg_mode option, must be 'none', 'coreg' or 'robust'."
      exit 1
    fi
    shift # past argument
    shift # past value
    ;;
    --qc_snap)
    hypvinn_flags+=(--qc_snap)
    shift # past argument
    ;;
    --mask_name)
    mask_name="$2"
    shift # past argument
    shift # past value
    ;;
    --norm_name)
    norm_name="$2"
    shift # past argument
    shift # past value
    ;;
    --norm_name_t2)
    norm_name_t2="$2"
    shift # past argument
    shift # past value
    ;;
    --aseg_segfile)
    aseg_segfile="$2"
    shift # past argument
    shift # past value
    ;;
    --conformed_name)
    conformed_name="$2"
    shift # past argument
    shift # past value
    ;;
    --conformed_name_t2)
    conformed_name_t2="$2"
    shift # past argument
    shift # past value
    ;;
    --seg_log)
    seg_log="$2"
    shift # past argument
    shift # past value
    ;;
    --viewagg_device | --run_viewagg_on)
    if [[ "$key" == "--run_viewagg_on" ]]
    then
      echo "WARNING: --run_viewagg_on (cpu|gpu|check) is deprecated and will be removed, use --viewagg_device <device|auto>."
    fi
    case "$2" in
      check)
      echo "WARNING: the option \"check\" is deprecated for --viewagg_device <device>, use \"auto\"."
      viewagg="auto"
      ;;
      gpu)
      viewagg="cuda"
      ;;
      *)
      viewagg="$2"
      ;;
    esac
    shift # past argument
    shift # past value
    ;;
    --no_cuda)
    echo "WARNING: --no_cuda is deprecated and will be removed, use --device cpu."
    device="cpu"
    shift # past argument
    ;;
    --no_biasfield)
    run_biasfield="0"
    shift # past argument
    ;;
    --no_asegdkt | --no_aparc)
    if [[ "$key" == "--no_aparc" ]]
    then
      echo "WARNING: --no_aparc is deprecated and will be removed, use --no_asegdkt."
    fi
    run_asegdkt_module="0"
    shift  # past argument
    ;;
    --no_cereb)
    run_cereb_module="0"
    shift  # past argument
    ;;
    --no_hypothal)
    run_hypvinn_module="0"
    shift  # past argument
    ;;
    --tal_reg)
    run_talairach_registration="true"
    shift
    ;;
    --device)
    device=$2
    shift # past argument
    shift # past value
    ;;
    --batch)
    batch_size="$2"
    shift # past argument
    shift # past value
    ;;
    --seg_only)
    run_surf_pipeline="0"
    shift # past argument
    ;;
    --surf_only)
    run_seg_pipeline="0"
    shift # past argument
    ;;
    --fstess)
    surf_flags=("${surf_flags[@]}" "--fstess")
    shift # past argument
    ;;
    --fsqsphere)
    surf_flags=("${surf_flags[@]}" "--fsqsphere")
    shift # past argument
    ;;
    --fsaparc)
    surf_flags=("${surf_flags[@]}" "--fsaparc")
    shift # past argument
    ;;
    --no_surfreg)
    surf_flags=("${surf_flags[@]}" "--no_surfreg")
    shift # past argument
    ;;
    --vox_size)
    vox_size="$2"
    shift # past argument
    shift # past value
    ;;
    --3t)
    surf_flags=("${surf_flags[@]}" "--3T")
    atlas3T="true"
    shift
    ;;
    --parallel)
    surf_flags=("${surf_flags[@]}" "--parallel")
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
    # Dev flag
    surf_flags=("${surf_flags[@]}" "--ignore_fs_version")
    shift # past argument
    ;;
    --no_fs_t1 )
    # Dev flag
    surf_flags=("${surf_flags[@]}" "--no_fs_T1")
    shift # past argument
    ;;
    --allow_root)
    allow_root=("--allow_root")
    shift # past argument
    ;;
    -h|--help)
    usage
    exit
    ;;
    --version)
    if [[ "$#" -lt 2 ]]; then
      version_and_quit="1"
    else
      case "$2" in
        all)
        version_and_quit="+checkpoints+git+pip"
        shift
        ;;
        +*)
        version_and_quit="$2"
        shift
        ;;
        --*)
        version_and_quit="1"
        ;;
        *)
        echo "Invalid option for --version: '$2', must be 'all' or [+checkpoints][+git][+pip]"
        exit 1
        ;;
      esac
    fi
    shift
    ;;
    *)    # unknown option
    if [[ "$1" == "" ]]
    then
      # skip empty arguments
      shift
    else
      echo "ERROR: Flag '$1' unrecognized."
      exit 1
    fi
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
if [[ "$run_seg_pipeline" == "1" ]] && { [[ -z "$t1" ]] || [[ ! -f "$t1" ]]; }
  then
    echo "ERROR: T1 image ($t1) could not be found. Must supply an existing T1 input (full head) via "
    echo "--t1 (absolute path and name) for generating the segmentation."
    echo "NOTES: If running in a container, make sure symlinks are valid!"
    exit 1;
fi

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


########################################## START ########################################################
mkdir -p "$(dirname "$seg_log")"

if [[ -f "$seg_log" ]]; then log_existed="true"
else log_existed="false"
fi

VERSION=$($python "$FASTSURFER_HOME/FastSurferCNN/version.py" "${version_args[@]}")
echo "Version: $VERSION" | tee -a "$seg_log"

### IF THE SCRIPT GETS TERMINATED, ADD A MESSAGE
trap "{ echo \"run_fastsurfer.sh terminated via signal at \$(date -R)!\" >> \"$seg_log\" ; }" SIGINT SIGTERM

# create the build log, file with all version info in parallel
printf "%s %s\n%s\n" "$THIS_SCRIPT" "${inputargs[*]}" "$(date -R)" >> "$build_log"
$python "$FASTSURFER_HOME/FastSurferCNN/version.py" "${version_args[@]}" >> "$build_log" &

if [[ "$run_seg_pipeline" != "1" ]]
  then
    echo "Running run_fastsurfer.sh without segmentation ; expecting previous --seg_only run in ${sd}/${subject}" | tee -a "$seg_log"
fi


if [[ "$run_seg_pipeline" == "1" ]]
  then
    # "============= Running FastSurferCNN (Creating Segmentation aparc.DKTatlas.aseg.mgz) ==============="
    # use FastSurferCNN to create cortical parcellation + anatomical segmentation into 95 classes.
    echo "Log file for segmentation FastSurferCNN/run_prediction.py" >> "$seg_log"
    date  2>&1 | tee -a "$seg_log"
    echo "" | tee -a "$seg_log"

    if [[ "$run_asegdkt_module" == "1" ]]
      then
        cmd=($python "$fastsurfercnndir/run_prediction.py" --t1 "$t1"
             --asegdkt_segfile "$asegdkt_segfile" --conformed_name "$conformed_name"
             --brainmask_name "$mask_name" --aseg_name "$aseg_segfile" --sid "$subject"
             --seg_log "$seg_log" --vox_size "$vox_size" --batch_size "$batch_size"
             --viewagg_device "$viewagg" --device "$device" "${allow_root[@]}")
        # specify the subject dir $sd, if asegdkt_segfile explicitly starts with it
        if [[ "$sd" == "${asegdkt_segfile:0:${#sd}}" ]]; then cmd=("${cmd[@]}" --sd "$sd"); fi
        echo "${cmd[@]}" | tee -a "$seg_log"
        "${cmd[@]}"
        exit_code="${PIPESTATUS[0]}"
        if [[ "${exit_code}" == 2 ]]
          then
            echo "ERROR: FastSurfer asegdkt segmentation failed QC checks."
            exit 1
        elif [[ "${exit_code}" -ne 0 ]]
          then
            echo "ERROR: FastSurfer asegdkt segmentation failed."
            exit 1
        fi
    fi
    if [[ -n "$t2" ]]
      then
        printf "INFO: Copying T2 file to %s..." "${copy_name_T2}" | tee -a "$seg_log"
        cmd=("nib-convert" "$t2" "$copy_name_T2")
        "${cmd[@]}" 2>&1 | tee -a "$seg_log"

        echo "INFO: Robust scaling (partial conforming) of T2 image..." | tee -a "$seg_log"
        cmd=($python "${fastsurfercnndir}/data_loader/conform.py" --no_strict_lia
             --no_vox_size --no_img_size "$t2" "$conformed_name_t2")
        "${cmd[@]}" 2>&1 | tee -a "$seg_log"
        echo "Done." | tee -a "$seg_log"
    fi

    if [[ "$run_biasfield" == "1" ]]
      then
        # this will always run, since norm_name is set to subject_dir/mri/orig_nu.mgz, if it is not passed/empty
        echo "INFO: Running N4 bias-field correction" | tee -a "$seg_log"
        cmd=($python "${reconsurfdir}/N4_bias_correct.py" "--in" "$conformed_name"
             --rescale "$norm_name" --aseg "$asegdkt_segfile" --threads "$threads")
        echo "${cmd[@]}" | tee -a "$seg_log"
        "${cmd[@]}" 2>&1 | tee -a "$seg_log"
        if [[ "${PIPESTATUS[0]}" -ne 0 ]]
          then
            echo "ERROR: Biasfield correction failed" | tee -a "$seg_log"
            exit 1
        fi

        if [[ "$run_talairach_registration" == "true" ]]
          then
            echo "INFO: Running talairach registration" | tee -a "$seg_log"
            cmd=("$reconsurfdir/talairach-reg.sh" "$sd/$subject/mri" "$atlas3T" "$seg_log")
            echo "${cmd[@]}" | tee -a "$seg_log"
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
                                    "EstimatedTotalIntraCranialVol"
                                    "BrainSegVol-to-eTIV" "MaskVol-to-eTIV"
                 )
            echo "${cmd[@]}" | tee -a "$seg_log"
            "${cmd[@]}" 2>&1 | tee -a "$seg_log"
            if [[ "${PIPESTATUS[0]}" -ne 0 ]]
              then
                echo "ERROR: asegdkt statsfile generation failed" | tee -a "$seg_log"
                exit 1
            fi
        fi

        if [[ -n "$t2" ]]
        then
          # ... we have a t2 image, bias field-correct it
          # (use the T2 image, not the conformed save  and robustly scaled uchar)
          echo "INFO: Running N4 bias-field correction of the t2" | tee -a "$seg_log"
          cmd=($python "${reconsurfdir}/N4_bias_correct.py" "--in" "$copy_name_T2"
               --out "$norm_name_t2" --threads "$threads" --uchar)
          echo "${cmd[@]}" | tee -a "$seg_log"
          "${cmd[@]}"
          if [[ "${PIPESTATUS[0]}" -ne 0 ]]
            then
              echo "ERROR: T2 Biasfield correction failed" | tee -a "$seg_log"
              exit 1
          fi
        fi
    else
      if [[ -n "$t2" ]]
      then
        # no biasfield, but a t2 is passed; presumably, this is biasfield corrected
        echo "INFO: Robustly rescaling $t2 to uchar ($norm_name_t2), which is assumed to already be biasfield corrected." | tee -a "$seg_log"
        cmd=($python "${fastsurfercnndir}/data_loader/conform.py" --no_force_lia
             --no_force_vox_size --no_force_img_size "$t2" "$norm_name_t2")
        echo "WARNING: --no_biasfield is activated, but FastSurfer does not check, if "
        echo "  passed T2 image is properly scaled and typed. T2 needs to be uchar and"
        echo "  robustly scaled (see FastSurferCNN/utils/data_loader/conform.py)!"
        # TODO implement/validate no changes to affine parameters for conform
        # "${cmd[@]}" 2>&1 | tee -a "$seg_log"
      fi
    fi

    if [[ "$run_cereb_module" == "1" ]]
      then
        if [[ "$run_biasfield" == "1" ]]
          then
            cereb_flags=("${cereb_flags[@]}" --norm_name "$norm_name"
                         --cereb_statsfile "$cereb_statsfile")
        else
            echo "INFO: Running CerebNet without generating a statsfile, since biasfield correction deactivated '--no_biasfield'." | tee -a $seg_log
        fi

        cmd=($python "$cerebnetdir/run_prediction.py" --t1 "$t1"
             --asegdkt_segfile "$asegdkt_segfile" --conformed_name "$conformed_name"
             --cereb_segfile "$cereb_segfile" --seg_log "$seg_log" --async_io
             --batch_size "$batch_size" --viewagg_device "$viewagg" --device "$device"
             --threads "$threads" "${cereb_flags[@]}" "${allow_root[@]}")
        # specify the subject dir $sd, if asegdkt_segfile explicitly starts with it
        if [[ "$sd" == "${cereb_segfile:0:${#sd}}" ]] ; then cmd=("${cmd[@]}" --sd "$sd"); fi
        echo "${cmd[@]}" | tee -a "$seg_log"
        "${cmd[@]}"
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
        echo "${cmd[@]}" | tee -a "$seg_log"
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
    pushd "$reconsurfdir" || exit 1
    cmd=("./recon-surf.sh" --sid "$subject" --sd "$sd" --t1 "$conformed_name"
         --asegdkt_segfile "$asegdkt_segfile" --threads "$threads" --py "$python"
         "${surf_flags[@]}" "${allow_root[@]}")
    echo "${cmd[@]}" | tee -a "$seg_log"
    "${cmd[@]}"
    if [[ "${PIPESTATUS[0]}" -ne 0 ]] ; then exit 1 ; fi
    popd || return
fi

########################################## End ########################################################
