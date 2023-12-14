#!/bin/bash
# Copyright 2022 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
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


# IMPORTS
if [ "$ANTSPATH" = "" ] || [ -f "$ANTSPATH/antsRegistrationSyNQuick.sh" ]
then
  exit "environment \$ANTSPATH not defined or invalid. \$ANTSPATH must contain antsRegistrationSyNQuick.sh."
fi

SELF=$0
labeled_subjects="UNDEFINED"
unlabeled_subjects="UNDEFINED"
output_dir="UNDEFINED"
unlabeled_subject="UNDEFINED"
labeled_subject="UNDEFINED"
unlabeled_t1="/mri/orig.mgz"
labeled_t1="/mri/orig.mgz"
output_warp="/warp.mgz"

while [[ $# -gt 0 ]]
do
    key="$1"

    case $key in
        -h | --help)
            echo "subject to subject registration for realistic data augmentation script"
            echo ""
            echo "usage:"
            echo "  $SELF [-h] --unlabeled_subjects <path> --labeled_subjects <path> --output_dir <path> "
            echo "    [--unlabeled_t1 <image name>] [--labeled_t1 <image name>] [--output_warp <output name>]"
            echo "    [[--unlabeled_subject <sid> --labeled_subject <sid>]"
            echo "optional parameters:"
            echo "    -h | --help: print this help"
            echo "    --unlabeled_subject: define the subject(s) to register to (may be a comma-separated list)."
            echo "    --labeled_subject: define the subjects(s) to register from (may be a comma-separated list)."
            echo "        labeled_subject and unlabeled_subject cannot be both lists, and if one is given the other "
            echo "        must as well be given, and if none is given stdin is read for pairs."
            echo "    --unlabeled_t1: file name of the t1 image (default: /mri/orig.mgz)"
            echo "    --labeled_t1: file name of the t1 image (default: /mri/orig.mgz)"
            echo "    --output_warp: file name of the output warped (default: /warp.nii.gz)"
            echo "required parameters"
            echo "    --labeled_subjects: path to subject directory of labeled subjects"
            echo "    --unlabeled_subjects: path to subject directory of unlabeled subjects"
            echo "    --output_dir: path to the output directory"
            exit
            ;;
        --unlabeled_subjects)
            unlabeled_subjects="$2"
            ;;
        --labeled_subjects)
            labeled_subjects="$2"
            ;;
        --labeled_subject)
            labeled_subject="$2"
            ;;
        --unlabeled_subject)
            unlabeled_subject="$2"
            ;;
        --output_dir)
            output_dir="$2"
            ;;
        --output_warp)
            output_warp="$2"
            ;;
        --unlabeled_t1)
            unlabeled_t1="$2"
            ;;
        --labeled_t1)
            labeled_t1="$2"
            ;;
    esac
    shift
    shift
done

if [ "$unlabeled_subjects" == "UNDEFINED" ] || [ "$labeled_subjects" == "UNDEFINED" ] || [ "$output_dir" == "UNDEFINED" ]
then
    exit "Missing unlabeled_subjects, labeled_subjects or output_dir!"
fi

! [ "$unlabeled_subject" == "UNDEFINED" ]; unlabeled_subject_defined=$?
! [ "$labeled_subject" == "UNDEFINED" ]; labeled_subject_defined=$?

if [ $unlabeled_subject_defined -ne $labeled_subject_defined ]
then
    exit "Either both or none of unlabeled_subject and labeled_subject need to be defined."
fi

shopt -s expand_aliases
shopt -s extglob

# cat list_of_subjects.txt | xargs -P10 -n 1 -I % sh -c './suit_gen.sh -s % -b [basepath] -i native_t1.nii'

moving_dataroot=/groups/ag-reuter/projects/cerebellum/datasets/cereb-dzne/08-03-21-jenny-final

# moving_dataroot=/groups/ag-reuter/projects/cerebellum/datasets/160_cases
testsuitv3=/groups/ag-reuter/datasets/testsuite/v3/data/all
output_dir=$moving_dataroot/subj2subj_reg
# t1_dataroot=$moving_dataroot/T1_fuer_GM-WM-boundary

mkdir -p $output_dir

if [ "$unlabeled_subject" != "UNDEFINED" ]
then
    IFS=',' read -r -a unlabeled_subject_array <<< "$unlabeled_subject"
fi

if [ "$labeled_subject" != "UNDEFINED" ]
then
    IFS=',' read -r -a labeled_subject_array <<< "$labeled_subject"
fi

if [ ${#labeled_subject_array} != 1 ] && [ ${#unlabeled_subject_array} != 1 ] && [ ${#labeled_subject_array} != ${#unlabeled_subject_array} ]
then
    exit "invalid parameters for labeled_subject and unlabeled_subject"
fi
i=0

function nextpair()
{
  if [ "$unlabeled_subject" == "UNDEFINED" ]
  then
      IFS="," read sub1 sub2
  else
      if [ $i >= ${#labeled_subject_array} ] && [ $i >= ${#unlabeled_subject_array} ]
      then
        sub1=""
      else
          if [ ${#unlabeled_subject_array} == 1 ]
          then
              sub1="${unlabeled_subject_array[0]}"
          else
              sub1="${unlabeled_subject_array[i]}"
          fi
          if [ ${#labeled_subject_array} == 1 ]
          then
              sub2="${labeled_subject_array[0]}"
          else
              sub2="${labeled_subject_array[i]}"
          fi
          let i=i+1
      fi
  fi
}

nextpair
while [ sub1 != "" ]
do
    unlab_path=$(find $unlabeled_subjects -name "*$sub1*")
    unlab_img="$unlab_path$unlabeled_t1"
    lab_path=$(find $labeled_subjects -name "*$sub2*")
    lab_img="$lab_path/$labeled_t1"
    if [ -f $unlab_img ] || [ -f $lab_img ]; then
        result_path=$output_dir/"${sub2}_to_${sub1}"/
        if [ ! -f "$result_path$output_warp" ]; then
            tmp_dir=`mkdir -d`
            echo "Registering $sub1 -> $sub2 ..."
            $ANTSPATH/antsRegistrationSyNQuick.sh -d 3 \
              -f $unlab_img \
              -m $lab_img \
              -t s \
              -o $tmp_dir
            echo "Done. Renaming/moving output file to $output_dir$output_warp"
            mkdir -p $result_path
            mv "$tmp_dir/1Warp.nii.gz" "$result_path$output_warp"
            rm "$tmp_dir/1InverseWarp.nii.gz" "$tmp_dir/InverseWarped.nii.gz" "$tmp_dir/Warped.nii.gz"
            rm $tmp_dir -R
        else
            echo "$result_path$output_warp already exists, skipping $sub1 -> $sub2."
        fi
    else
        echo "WARNING: $lab_img or $unlab_img does not exist, skipping $sub1 -> $sub2."
    fi

    nextpair
done
