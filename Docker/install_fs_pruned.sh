#!/bin/bash --login
# --login to read bashrc for conda inside docker

# This file downloads the FreeSurfer tar ball and extracts from it only what is needed to run
# FastSurfer
#
# In order to update to a new FreeSurfer version you need to update the fslink and then build a 
# docker with this setup. Run it and whenever it crashes/exits, find the missing file (binary,
# atlas, datafile, or dependency) and add it here or if a dependeny is missing install it in the 
# docker and rebuild and re-run. Repeat until recon-surf finishes sucessfullly. Then repeat with
# all supported recon-surf flags (--hires, --fsaparc etc.).


# Link where to find the FreeSurfer tarball: 
fslink="https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/7.3.2/freesurfer-linux-ubuntu20_amd64-7.3.2.tar.gz"


if [ "$#" -lt 1 ]; then
    echo
    echo "Usage: install_fs_prunded install_dir <--upx>"
    echo 
    echo "--upx is optional, if passed, fs/bin will be packed"
    echo 
    exit 2
fi


where=/opt
if [ "$#" -ge 1 ]; then
  where=$1
fi

fss=$where/fs-tmp
fsd=$where/freesurfer
echo
echo "Will install FreeSurfer to $fsd"
echo
echo "FreeSurfer package to download:"
echo
echo "$fslink"
echo


# get Freesurfer and upack (some of it)
echo "Downloading FS and unpacking portions ..."
wget --no-check-certificate -qO- $fslink  | tar zxv --no-same-owner -C $where \
      --exclude='freesurfer/average/*.gca' \
      --exclude='freesurfer/average/Buckner_JNeurophysiol11_MNI152' \
      --exclude='freesurfer/average/Choi_JNeurophysiol12_MNI152' \
      --exclude='freesurfer/average/mult-comp-cor' \
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
      --exclude='freesurfer/python/packages' \
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
      --exclude='freesurfer/trctrain'


# rename download to tmp
mv $where/freesurfer $fss

# mk directories
mkdir -p $fsd/average
mkdir -p $fsd/bin
mkdir -p $fsd/etc
mkdir -p $fsd/lib/bem
mkdir -p $fsd/python/scripts
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
  bin/avi2talxfm
  bin/compute_vox2vox
  bin/defect2seg
  bin/fs_temp_dir
  bin/fs_temp_file
  bin/fs-check-version
  bin/fsr-getxopts
  bin/gauss_4dfp
  bin/imgreg_4dfp
  bin/lta_convert
  bin/mpr2mni305
  bin/mri_add_xform_to_header
  bin/mri_annotation2label
  bin/mri_binarize
  bin/mri_brainvol_stats
  bin/mri_cc
  bin/mri_concat
  bin/mri_concatenate_lta
  bin/mri_convert
  bin/mri_diff
  bin/mri_edit_wm_with_aseg
  bin/mri_fill
  bin/mri_fuse_segmentations
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
  bin/mri_surf2volseg
  bin/mri_tessellate
  bin/mri_vol2surf
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
  bin/mris_inflate
  bin/mris_info
  bin/mris_jacobian
  bin/mris_label2annot
  bin/mris_place_surface
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
  subjects/fsaverage/surf/lh.sphere
  subjects/fsaverage/surf/lh.sphere.reg
  subjects/fsaverage/surf/lh.white
  subjects/fsaverage/surf/rh.sphere
  subjects/fsaverage/surf/rh.sphere.reg
  subjects/fsaverage/surf/rh.white"
echo
for file in $copy_files
do
  echo "copying $file"
  cp $fss/$file $fsd/$file
done


# FS looks for them, but does not call them
touch_files="/average/RB_all_2020-01-02.gca"
echo
for file in $touch_files
do
  echo "touching $file"
  touch $fsd/$file 
done

# FS calls these for version info, but we don't need them
# so we link them to mri_info to save space. 
link_files="
  bin/mri_and
  bin/mri_aparc2aseg
  bin/mri_ca_label
  bin/mri_ca_normalize
  bin/mri_ca_register
  bin/mri_compute_overlap
  bin/mri_compute_seg_overlap
  bin/mri_em_register
  bin/mri_fwhm
  bin/mri_gcut
  bin/mri_log_likelihood
  bin/mri_motion_correct.fsl
  bin/mri_normalize_tp2
  bin/mri_or
  bin/mri_relabel_nonwm_hypos
  bin/mri_remove_neck
  bin/mri_stats2seg
  bin/mri_surf2surf
  bin/mri_surf2vol
  bin/mri_surfcluster
  bin/mri_vol2vol
  bin/mri_voldiff
  bin/mri_watershed
  bin/mris_divide_parcellation
  bin/mris_left_right_register
  bin/mris_surface_stats
  bin/mris_thickness
  bin/mris_thickness_diff
  bin/nu_correct
  bin/tkregister2_cmdl"

# create target for link with ERROR message if called
ltrg=$fsd/bin/not-here.sh
echo '#!/bin/bash
if [ "$1" == "-all-info" ]; then
  echo "$0 not included ..."
  exit 0
fi
echo
echo "ERROR: The binary $0 is not included, your call is forwared to not-here.sh"
echo
exit 1
' > $ltrg
chmod a+x $ltrg
echo
for file in $link_files
do
  echo "linking $file"
  ln -s $ltrg $fsd/$file 
done

# use our python (not really needed in recon-all anyway)
p3=`which python3`
if [ "$p3" == "" ]; then
  echo "No python3 found, please install first!"
  echo
  exit 1
fi
ln -s $p3 $fsd/bin/fspython

#cleanup
rm -rf $fss

# pack if desired with upx
if [ "$#" -ge 2 ]; then
  if [ "${2^^}" == "--UPX" ] ; then
    echo "packing $fsd/bin/ executables (this can take a while) ..."
    exe=`find $fsd/bin -exec file {} \; | grep ELF | cut -d: -f1`
    upx -9 $exe
  fi
fi
