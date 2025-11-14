#!/bin/bash --login
# --login to read bashrc for conda inside docker

# This file downloads the FreeSurfer tar ball and extracts from it only what is needed to run
# FastSurfer
#
# In order to update to a new FreeSurfer version you need to update the fslink and then build a 
# docker with this setup. Run it and whenever it crashes/exits, find the missing file (binary,
# atlas, datafile, or dependency) and add it here or if a dependency is missing install it in the 
# docker and rebuild and re-run. Repeat until recon-surf finishes successfully. Then repeat with
# all supported recon-surf flags (--hires, --fsaparc etc.).


if [[ -z "${BASH_SOURCE[0]}" ]]; then THIS_SCRIPT="$0"
else THIS_SCRIPT="${BASH_SOURCE[0]}"
fi
# Link where to find the FreeSurfer tarball:
fslink="default"

if [[ "$#" -lt 1 ]]; then
    echo
    echo "Usage: install_fs_pruned.sh install_dir [--upx] [--url freesurfer_download_url]"
    echo 
    echo "--upx is optional, if passed, fs/bin will be packed"
    echo "--url is recommended! This is the download link for freesurfer."
    echo "  The link can be found in pyproject.toml:tool.freesurfer.url!"
    echo
    exit 2
fi

where=/opt
if [[ "$#" -ge 1 ]]; then
  where=$1
  shift
fi

upx="false"
while [[ "$#" -ge 1 ]]; do
  lowercase=$(echo "$1" | tr '[:upper:]' '[:lower:]')
  case $lowercase in
  --upx) upx="true" ; shift ;;
  --url) fslink=$2 ; shift ; shift ;;
  *) echo "Invalid argument $1" ; exit 1 ;;
  esac
done
fss=$where/fs-tmp
fsd=$where/freesurfer

if [[ "$fslink" == "default" ]]
then
  # --url not provided, try getting it from pyproject.toml
link=$(python3 <<EOF
import sys, pathlib
if sys.version_info >= (3, 11): import tomllib
else:
  try: import tomli as tomllib
  except Exception: sys.exit()

for path in pathlib.Path("$THIS_SCRIPT").parents:
  try:
    if (path / "pyproject.toml").exists():
      with open(path / "pyproject.toml", "rb") as fp: dat = tomllib.load(fp)["tool"]["freesurfer"]
      print(dat["urls"]["linux"].format(**dat))
      break
  except Exception:
    continue # ignore all errors
EOF
)
  if [[ -n "$link" ]] ; then fslink="$link"
  else echo "ERROR: Please provide the --url argument, could not find/parse pyproject.toml!" ; exit 1
  fi
fi

echo
echo "Will install FreeSurfer to $fsd"
echo
echo "FreeSurfer package to download:"
echo
echo "$fslink"
echo


function run_parallel ()
{
  # param 1 num_parallel_processes
  # param 2 command (printf string)
  # param 3 how many entries to consume from $@ per "run"
  # param ... parameters to format, ie. we are executing $(printf $command $@...)
  i=0
  pids=()
  num_parallel_processes=$1
  command=$2
  num=$3
  shift
  shift
  shift
  args=("$@")
  j=0
  while [[ "$j" -lt "${#args}" ]]
  do
    cmd=$(printf "$command" "${args[@]:$j:$num}")
    j=$((j + num))
    $cmd &
    pids=("${pids[@]}" "$!")
    i=$((i + 1))
    if [[ "$i" -ge "$num_parallel_processes" ]]
    then
      wait "${pids[0]}"
      pids=("${pids[@]:1}")
    fi
  done
  for pid in "${pids[@]}"
  do
    wait "$pid"
  done
}


# get FreeSurfer and unpack (some of it)
echo "Downloading FS and unpacking portions ..."

# temp freesurfer dl filename (to save the dl)
freesurfer_dl="freesurfer_$(date +%s)"

# dl aria2c if that exists, else wget or curl
if [[ -n "$(which aria2c)" ]] ; then aria2c -x 16 -s 16 -c --check-certificate=false -o $freesurfer_dl "$fslink" 
elif [[ -n "$(which wget)" ]] ; then wget --no-check-certificate -qO- "$fslink" >> $freesurfer_dl
else curl -L --insecure "$fslink" >> $freesurfer_dl
fi

tar zxv --no-same-owner -C "$where" \
      --exclude='freesurfer/average/*.gca' \
      --exclude='freesurfer/average/Buckner_JNeurophysiol11_MNI152' \
      --exclude='freesurfer/average/Choi_JNeurophysiol12_MNI152' \
      --exclude='freesurfer/average/mult-comp-cor' \
      --exclude='freesurfer/average/samseg' \
      --exclude='freesurfer/average/Yeo_Brainmap_MNI152' \
      --exclude='freesurfer/average/Yeo_JNeurophysiol11_MNI152' \
      --exclude='freesurfer/bin/freeview.bin' \
      --exclude='freesurfer/bin/freeview' \
      --exclude='freesurfer/bin/fs_spmreg.glnxa64' \
      --exclude='freesurfer/bin/mris_decimate_gui.bin' \
      --exclude='freesurfer/bin/mris_decimate_gui' \
      --exclude='freesurfer/bin/qdec_glmfit' \
      --exclude='freesurfer/bin/qdec.bin' \
      --exclude='freesurfer/bin/qdec' \
      --exclude='freesurfer/bin/SegmentSubfieldsT1Longitudinal' \
      --exclude='freesurfer/bin/SegmentSubjectT1_autoEstimateAlveusML' \
      --exclude='freesurfer/bin/SegmentSubjectT1T2_autoEstimateAlveusML' \
      --exclude='freesurfer/bin/SegmentSubjectT2_autoEstimateAlveusML' \
      --exclude='freesurfer/diffusion' \
      --exclude='freesurfer/fsafd' \
      --exclude='freesurfer/fsfast' \
      --exclude='freesurfer/lib/cuda' \
      --exclude='freesurfer/lib/images' \
      --exclude='freesurfer/lib/qt' \
      --exclude='freesurfer/lib/tcl' \
      --exclude='freesurfer/lib/tktools' \
      --exclude='freesurfer/lib/vtk' \
      --exclude='freesurfer/matlab' \
      --exclude='freesurfer/mni-1.4' \
      --exclude='freesurfer/mni' \
      --exclude='freesurfer/models' \
      --exclude='freesurfer/python/bin' \
      --exclude='freesurfer/python/include' \
      --exclude='freesurfer/python/lib' \
      --exclude='freesurfer/python/share' \
      --exclude='freesurfer/subjects/bert' \
      --exclude='freesurfer/subjects/cvs_avg35_inMNI152' \
      --exclude='freesurfer/subjects/cvs_avg35' \
      --exclude='freesurfer/subjects/fsaverage_sym' \
      --exclude='freesurfer/subjects/fsaverage3' \
      --exclude='freesurfer/subjects/fsaverage4' \
      --exclude='freesurfer/subjects/fsaverage5' \
      --exclude='freesurfer/subjects/fsaverage6' \
      --exclude='freesurfer/subjects/lh.EC_average' \
      --exclude='freesurfer/subjects/rh.EC_average' \
      --exclude='freesurfer/subjects/V1_average' \
      --exclude='freesurfer/tktools' \
      --exclude='freesurfer/trctrain' \
      -f $freesurfer_dl

 rm $freesurfer_dl

# rename download to tmp
mv $where/freesurfer $fss

# mk directories
mkdir -p $fsd/average
mkdir -p $fsd/bin
mkdir -p $fsd/etc
mkdir -p $fsd/lib/bem
mkdir -p $fsd/python/scripts
mkdir -p $fsd/python/packages/fsbindings
mkdir -p $fsd/subjects/fsaverage/label
mkdir -p $fsd/subjects/fsaverage/surf

# We need these
copy_files="
  ASegStatsLUT.txt
  build-stamp.txt
  DefectLUT.txt
  FreeSurferColorLUT.txt
  FreeSurferEnv.sh
  SegmentNoLUT.txt
  SetUpFreeSurfer.sh
  Simple_surface_labels2009.txt
  sources.csh  
  SubCorticalMassLUT.txt
  WMParcStatsLUT.txt
  average/3T18yoSchwartzReactN32_as_orig.4dfp.hdr
  average/3T18yoSchwartzReactN32_as_orig.4dfp.ifh
  average/3T18yoSchwartzReactN32_as_orig.4dfp.img
  average/3T18yoSchwartzReactN32_as_orig.4dfp.img.rec
  average/3T18yoSchwartzReactN32_as_orig.4dfp.mat
  average/3T18yoSchwartzReactN32_as_orig.lst
  average/711-2B_as_mni_average_305_mask.4dfp.hdr
  average/711-2B_as_mni_average_305_mask.4dfp.ifh
  average/711-2B_as_mni_average_305_mask.4dfp.img
  average/711-2B_as_mni_average_305_mask.4dfp.img.rec
  average/711-2C_as_mni_average_305.4dfp.hdr
  average/711-2C_as_mni_average_305.4dfp.ifh
  average/711-2C_as_mni_average_305.4dfp.img
  average/711-2C_as_mni_average_305.4dfp.img.rec
  average/711-2C_as_mni_average_305.4dfp.mat
  average/colortable_BA.txt
  average/colortable_desikan_killiany.txt
  average/colortable_vpnl.txt
  average/lh.CDaparc.atlas.acfb40.noaparc.i12.2016-08-02.gcs
  average/lh.DKaparc.atlas.acfb40.noaparc.i12.2016-08-02.gcs
  average/lh.DKTaparc.atlas.acfb40.noaparc.i12.2016-08-02.gcs
  average/lh.folding.atlas.acfb40.noaparc.i12.2016-08-02.tif
  average/mni305.cor.mgz
  average/mni305.mask.cor.mgz
  average/rh.CDaparc.atlas.acfb40.noaparc.i12.2016-08-02.gcs
  average/rh.DKaparc.atlas.acfb40.noaparc.i12.2016-08-02.gcs
  average/rh.DKTaparc.atlas.acfb40.noaparc.i12.2016-08-02.gcs
  average/rh.folding.atlas.acfb40.noaparc.i12.2016-08-02.tif
  bin/analyzeto4dfp
  bin/AntsDenoiseImageFs
  bin/asegstats2table
  bin/aparcstats2table
  bin/avi2talxfm
  bin/compute_vox2vox
  bin/defect2seg
  bin/fname2stem
  bin/fspython
  bin/fs_temp_dir
  bin/fs_temp_file
  bin/fs-check-version
  bin/fsr-getxopts
  bin/gauss_4dfp
  bin/ifh2hdr
  bin/imgreg_4dfp
  bin/isanalyze
  bin/isnifti
  bin/lta_convert
  bin/make_upright
  bin/mpr2mni305
  bin/mri_add_xform_to_header
  bin/mri_annotation2label
  bin/mri_binarize
  bin/mri_brainvol_stats
  bin/mri_cc
  bin/mri_concat
  bin/mri_concatenate_lta
  bin/mri_convert
  bin/mri_coreg
  bin/mri_diff
  bin/mri_edit_wm_with_aseg
  bin/mri_fill
  bin/mri_fuse_segmentations
  bin/mri_glmfit
  bin/mri_info
  bin/mri_label2label
  bin/mri_label2vol
  bin/mri_mask
  bin/mri_matrix_multiply
  bin/mri_mc
  bin/mri_normalize
  bin/mri_pretess
  bin/mri_relabel_hypointensities
  bin/mri_robust_register
  bin/mri_robust_template
  bin/mri_segment
  bin/mri_segstats
  bin/mri_surf2surf
  bin/mri_surf2volseg
  bin/mri_tessellate
  bin/mri_vol2surf
  bin/mri_vol2vol
  bin/mris_anatomical_stats
  bin/mris_autodet_gwstats
  bin/mris_ca_label
  bin/mris_calc
  bin/mris_convert
  bin/mris_curvature
  bin/mris_curvature_stats
  bin/mris_defects_pointset
  bin/mris_diff
  bin/mris_euler_number
  bin/mris_extract_main_component
  bin/mris_fix_topology
  bin/mris_inflate
  bin/mris_info
  bin/mris_jacobian
  bin/mris_label2annot
  bin/mris_place_surface
  bin/mris_preproc
  bin/mris_register
  bin/mris_remesh
  bin/mris_remove_intersection
  bin/mris_sample_parc
  bin/mris_smooth
  bin/mris_sphere
  bin/mris_topo_fixer
  bin/mris_volmask
  bin/mrisp_paint
  bin/pctsurfcon
  bin/rca-config
  bin/rca-config2csh
  bin/recon-all
  bin/talairach_avi
  bin/UpdateNeeded
  bin/vertexvol
  etc/recon-config.yaml
  lib/bem/ic4.tri
  lib/bem/ic7.tri
  python/packages/fsbindings/legacy.py
  python/scripts/asegstats2table
  python/scripts/aparcstats2table
  python/scripts/rca-config
  python/scripts/rca-config2csh
  subjects/fsaverage/label/lh.aparc.annot
  subjects/fsaverage/label/lh.BA1_exvivo.label
  subjects/fsaverage/label/lh.BA1_exvivo.thresh.label
  subjects/fsaverage/label/lh.BA2_exvivo.label
  subjects/fsaverage/label/lh.BA2_exvivo.thresh.label
  subjects/fsaverage/label/lh.BA3a_exvivo.label
  subjects/fsaverage/label/lh.BA3a_exvivo.thresh.label
  subjects/fsaverage/label/lh.BA3b_exvivo.label
  subjects/fsaverage/label/lh.BA3b_exvivo.thresh.label
  subjects/fsaverage/label/lh.BA44_exvivo.label
  subjects/fsaverage/label/lh.BA44_exvivo.thresh.label
  subjects/fsaverage/label/lh.BA45_exvivo.label
  subjects/fsaverage/label/lh.BA45_exvivo.thresh.label
  subjects/fsaverage/label/lh.BA4a_exvivo.label
  subjects/fsaverage/label/lh.BA4a_exvivo.thresh.label
  subjects/fsaverage/label/lh.BA4p_exvivo.label
  subjects/fsaverage/label/lh.BA4p_exvivo.thresh.label
  subjects/fsaverage/label/lh.BA6_exvivo.label
  subjects/fsaverage/label/lh.BA6_exvivo.thresh.label
  subjects/fsaverage/label/lh.cortex.label
  subjects/fsaverage/label/lh.entorhinal_exvivo.label
  subjects/fsaverage/label/lh.entorhinal_exvivo.thresh.label
  subjects/fsaverage/label/lh.FG1.mpm.vpnl.label
  subjects/fsaverage/label/lh.FG2.mpm.vpnl.label
  subjects/fsaverage/label/lh.FG3.mpm.vpnl.label
  subjects/fsaverage/label/lh.FG4.mpm.vpnl.label
  subjects/fsaverage/label/lh.hOc1.mpm.vpnl.label
  subjects/fsaverage/label/lh.hOc2.mpm.vpnl.label
  subjects/fsaverage/label/lh.hOc3v.mpm.vpnl.label
  subjects/fsaverage/label/lh.hOc4v.mpm.vpnl.label
  subjects/fsaverage/label/lh.MT_exvivo.label
  subjects/fsaverage/label/lh.MT_exvivo.thresh.label
  subjects/fsaverage/label/lh.perirhinal_exvivo.label
  subjects/fsaverage/label/lh.perirhinal_exvivo.thresh.label
  subjects/fsaverage/label/lh.V1_exvivo.label
  subjects/fsaverage/label/lh.V1_exvivo.thresh.label
  subjects/fsaverage/label/lh.V2_exvivo.label
  subjects/fsaverage/label/lh.V2_exvivo.thresh.label
  subjects/fsaverage/label/rh.aparc.annot
  subjects/fsaverage/label/rh.BA1_exvivo.label
  subjects/fsaverage/label/rh.BA1_exvivo.thresh.label
  subjects/fsaverage/label/rh.BA2_exvivo.label
  subjects/fsaverage/label/rh.BA2_exvivo.thresh.label
  subjects/fsaverage/label/rh.BA3a_exvivo.label
  subjects/fsaverage/label/rh.BA3a_exvivo.thresh.label
  subjects/fsaverage/label/rh.BA3b_exvivo.label
  subjects/fsaverage/label/rh.BA3b_exvivo.thresh.label
  subjects/fsaverage/label/rh.BA44_exvivo.label
  subjects/fsaverage/label/rh.BA44_exvivo.thresh.label
  subjects/fsaverage/label/rh.BA45_exvivo.label
  subjects/fsaverage/label/rh.BA45_exvivo.thresh.label
  subjects/fsaverage/label/rh.BA4a_exvivo.label
  subjects/fsaverage/label/rh.BA4a_exvivo.thresh.label
  subjects/fsaverage/label/rh.BA4p_exvivo.label
  subjects/fsaverage/label/rh.BA4p_exvivo.thresh.label
  subjects/fsaverage/label/rh.BA6_exvivo.label
  subjects/fsaverage/label/rh.BA6_exvivo.thresh.label
  subjects/fsaverage/label/rh.cortex.label
  subjects/fsaverage/label/rh.entorhinal_exvivo.label
  subjects/fsaverage/label/rh.entorhinal_exvivo.thresh.label
  subjects/fsaverage/label/rh.FG1.mpm.vpnl.label
  subjects/fsaverage/label/rh.FG2.mpm.vpnl.label
  subjects/fsaverage/label/rh.FG3.mpm.vpnl.label
  subjects/fsaverage/label/rh.FG4.mpm.vpnl.label
  subjects/fsaverage/label/rh.hOc1.mpm.vpnl.label
  subjects/fsaverage/label/rh.hOc2.mpm.vpnl.label
  subjects/fsaverage/label/rh.hOc3v.mpm.vpnl.label
  subjects/fsaverage/label/rh.hOc4v.mpm.vpnl.label
  subjects/fsaverage/label/rh.MT_exvivo.label
  subjects/fsaverage/label/rh.MT_exvivo.thresh.label
  subjects/fsaverage/label/rh.perirhinal_exvivo.label
  subjects/fsaverage/label/rh.perirhinal_exvivo.thresh.label
  subjects/fsaverage/label/rh.V1_exvivo.label
  subjects/fsaverage/label/rh.V1_exvivo.thresh.label
  subjects/fsaverage/label/rh.V2_exvivo.label
  subjects/fsaverage/label/rh.V2_exvivo.thresh.label
  subjects/fsaverage/surf/lh.curv
  subjects/fsaverage/surf/lh.pial
  subjects/fsaverage/surf/lh.pial_semi_inflated
  subjects/fsaverage/surf/lh.sphere
  subjects/fsaverage/surf/lh.sphere.reg
  subjects/fsaverage/surf/lh.white
  subjects/fsaverage/surf/rh.curv
  subjects/fsaverage/surf/rh.pial
  subjects/fsaverage/surf/rh.pial_semi_inflated
  subjects/fsaverage/surf/rh.sphere
  subjects/fsaverage/surf/rh.sphere.reg
  subjects/fsaverage/surf/rh.white"
echo
for file in $copy_files
do
  echo "copying $file"
  cp -r $fss/$file $fsd/$file
done

# pack if desired with upx (do this before adding all the links
if [[ "$upx" == "true" ]] ; then
  echo "finding executables in $fsd/bin/..."
  exe=$(find $fsd/bin -exec file {} \; | grep ELF | cut -d: -f1)
  echo "packing $fsd/bin/ executables (this can take a while) ..."
  run_parallel 8 "upx -9 %s %s %s %s" 4 $exe
fi

# Modify fsbindings Python package to allow calling scripts like asegstats2table directly:
echo "from . import legacy" > "$fsd/python/packages/fsbindings/__init__.py"

# FS looks for them, but does not call them
touch_files="/average/RB_all_2020-01-02.gca"
echo
for file in $touch_files
do
  echo "touching $file"
  touch $fsd/$file 
done

#cleanup
rm -rf $fss
