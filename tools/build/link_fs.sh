#!/usr/bin/env bash

# usage: link_fs.sh [<path-to-python-interpreter> [<FREESURFER_HOME>]]

if [[ "$#" -gt 0 ]] && { [[ "${*/-h/}" != "$*" ]] || [[ "${*/--help/}" != "$*" ]] ; } ; then
  echo "usage: $0 [<path-to-python-interpreter> [<FREESURFER_HOME>]]"
  exit 0
elif [[ "$#" == 1 ]] || [[ "$#" == 2 ]]
then
  if [[ ! -e "$1" ]] ; then echo "ERROR: $1 does not exit!" ; exit 1 ; fi
  PYTHON="$1"
  if [[ "$#" == 2 ]] ; then FREESURFER_HOME="$2" ; fi
else
  PYTHON=$(which python3)
fi
if [[ -z "$FREESURFER_HOME" ]] || [[ ! -d "$FREESURFER_HOME" ]]
then
  echo "ERROR: FREESURFER_HOME not defined correctly!"
  exit 1
fi

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
  bin/mri_surf2vol
  bin/mri_surfcluster
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
ltrg=$FREESURFER_HOME/bin/not-here.sh
echo '#!/bin/bash
if [ "$1" == "-all-info" ]; then
  echo "$0 not included ..."
  exit 0
fi
echo
echo "ERROR: The binary $0 is not included, your call is forwarded to not-here.sh"
echo
exit 1
' > $ltrg
chmod a+x $ltrg
echo
for file in $link_files
do
  echo "linking $file"
  ln -s "$ltrg" "$FREESURFER_HOME/$file"
done

# use our python (not really needed in recon-all anyway)
ln -sf "$PYTHON" "$FREESURFER_HOME/bin/fspython"
