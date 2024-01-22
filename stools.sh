#!/bin/bash

# script for functions used by srun_fastsurfer.sh and srun_freesufer.sh

function read_cases ()
{
  # param1 data_dir
  # param2 search_pattern
  # param3 (optional, alternate mount_dir to print paths relative to)
  # outputs one line per subject, format: "subject_id=input_path"
  # cases are read based on globbing the search_pattern in data_dir,
  # but may be transformed to the optional mount_dir
  prev_pwd="$(pwd)"
  { pushd "$1" > /dev/null || (echo "Could not go to $1" && exit 1)} >&2
    # pattern without fixed postfixes, e.g. */mri/orig.mgz -> *
    no_fixed_postfix="${2/%\/[^*{[]*}"
    for file_match in $(eval echo "./$2"); do
      if [[ -e "$file_match" ]]
      then
        file_match="${file_match/#.\/}"
        # remove postfix from file_match
        subject="${file_match/%${2:${#no_fixed_postfix}}}"
        subject="${subject//\//_}" # / -> _
        # remove common file extensions
        subject="${subject/%.nii.gz/}"
        subject="${subject/%.nii/}"
        subject="${subject/%.mgz/}"
        if [[ $(($#)) -gt 2 ]]
        then
          echo "$subject=$3/$file_match"
        else
          echo "$subject=$1/$file_match"
        fi
      fi
    done
  { popd > /dev/null || (echo "Could not return to $prev_pwd" && exit 1)} >&2
}

function translate_cases ()
{
  #param1 root_dir
  #param2 subject_list file
  #param3 target_dir
  #param4 optional, delimiter
  #param5 optional, awk snippets to modify the subject_id and the subject_path (split by :), default '$1=$2'
  if [[ "$#" -gt 3 ]]
  then
    delimiter=$4
  else
    delimiter=","
  fi
  if [[ "$#" -gt 4 ]]
  then
    subid_awk="$(echo "$5" | cut -f1 -d=)"
    subpath_awk="$(echo "$5" | cut -f2 -d=)"
  else
    subid_awk='$1'
    subpath_awk='$2'
  fi
  script="
  BEGIN {
    regex=\"^(\" source_dir \"|\" target_dir \")\";
    regex2=\",(\" source_dir \"|\" target_dir \")/*\";
  }
  length(\$NF) > 1 {
    subid=$subid_awk;
    subpath=$subpath_awk;
    gsub(regex, \"\", subpath);
    gsub(regex2, \",\" target_dir \"/\", subpath);
    print subid \"=\" target_dir \"/\" subpath;
  }"
  #>&2 echo "awk -F \"$delimiter\" -v target_dir=\"$3\" -v source_dir=\"$1\" \"$script\" \"$2\""
  awk -F "$delimiter" -v target_dir="$3" -v source_dir="$1" "$script" "$2"
}

# step zero: make directories
function check_hpc_work ()
{
  #param1 hpc_work
  #param2 true/false, optional check empty, default false
  if [[ -z "$1" ]] || [[ ! -d "$1" ]]; then
    echo "The hpc_work directory $1 is not defined or does not exists."
    exit 1
  fi
  if [[ "$#" -gt 1 ]] && [[ "$2" == "true" ]] && [[ "$(ls $1 | wc -w)" -gt "0" ]]; then
    echo "The hpc_work directory $1 not empty."
    exit 1
  fi
}
function check_out_dir ()
{
  #param1 out_dir
  #param2 true/false, optional check empty, default false
  if [[ -z "$1" ]]; then
    echo "The subject directory (output directory) is not defined."
  elif [[ ! -d "$1" ]]; then
    echo "The subject directory $1 (output directory) does not exists."
    read -r -p "Create the directory? [y/N]" -n 1 retval
    echo ""
    if [[ "$retval" == "y" ]] || [[ "$retval" == "Y" ]] ; then mkdir -p "$1" ;
    else exit 1; fi
  fi
}
function check_singularity_image ()
{
  #param1 singularity_image
  if [[ -z "$1" ]] || [[ ! -f "$1" ]]; then
    echo "The singularity image $1 did not exist."
    exit 1
  fi
}
function check_fs_license ()
{
  #param1 fs_license
  if [[ -z "$1" ]] || [[ ! -f "$1" ]]; then
    echo "Cannot find the FreeSurfer license (--fs_license, at \"$1\")"
    exit 1
  fi
}
function check_seg_surf_only ()
{
  #param1 seg_only
  #param2 surf_only
  if [[ "$1" == "true" ]] && [[ "$2" == "true" ]]; then
    echo "Selecting both --seg_only and --surf_only is invalid!"
    exit 1
  fi
}
function check_subject_images ()
{
  #param1 cases
  if [[ "$#" -lt 1 ]]; then >&2 echo "check_subject_images is missing parameters!"; exit 1; fi
  missing_subject_ids=""
  missing_subject_imgs=""
  for subject in $1
  do
    subject_id=$(echo "$subject" | cut -d= -f1)
    image_path=$(echo "$subject" | cut -d= -f2)
    #TODO: also check here, if any of the folders up to the mounted dir leading to the file are symlinks
    #TODO: if so, this will lead to problems
    if [[ ! -e "$image_path" ]]
    then
      if [[ -n "$missing_subject_ids" ]]
      then
        missing_subject_ids="$missing_subject_ids, "
        missing_subject_imgs="$missing_subject_imgs, "
      fi
      missing_subject_ids="$missing_subject_ids$subject_id"
      missing_subject_imgs="$missing_subject_imgs$image_path"
    fi
  done
  if [[ -n "$missing_subject_ids" ]]
  then
    echo "ERROR: Some images are missing!"
    echo "Subject IDs: $missing_subject_ids"
    echo "Files: $missing_subject_imgs"
    exit 1
  fi
}

function make_hpc_work_dirs ()
{
  # param1: hpc_work
  mkdir "$1/images" &
  mkdir "$1/scripts" &
  mkdir "$1/cases" &
  mkdir "$1/logs" &
}

# copy results back and clean the output directory
function make_cleanup_job ()
{
  # param1: hpc_work directory
  # param2: output directory
  # param3: dependency tag
  # param4: optional: true/false (delete hpc_work directory, default=false)
  # param5: optional: true/false (submit jobs, default=true)

  # param4: optional: true/false (delete hpc_work, default=false)
  # param5: optional: true/false (submit jobs, default=true)
  local clean_cmd_file
  local submit_jobs
  local delete_hpc_work_dir
  if [[ "$#" -gt 3 ]] && [[ "$4" == "true" ]]
  then
    delete_hpc_work_dir="true"
  else
    delete_hpc_work_dir="false"
  fi
  if [[ "$#" -gt 4 ]] && [[ "$5" == "false" ]]
  then
    submit_jobs="false"
    clean_cmd_file=$(mktemp)
  else
    submit_jobs="true"
    clean_cmd_file=$hpc_work/scripts/slurm_cleanup.sh
  fi
  local clean_cmd_filename=$hpc_work/scripts/slurm_cleanup.sh

  local hpc_work=$1
  local out_dir=$2
  local clean_slurm_sched=(-d "$3" -J "FastSurfer-Cleanup-$USER"
    --ntasks=1 --cpus-per-task=4 -o "$out_dir/slurm/logs/cleanup_%A.log"
    "$clean_cmd_filename")

  mkdir -p "$out_dir/slurm/logs"
  {
    echo "#!/bin/bash"
    echo "set -e"
    echo "mkdir -p $out_dir/slurm/logs"
    echo "mkdir -p $out_dir/slurm/scripts"
    echo "pids=()"
    echo "if [[ -d \"$hpc_work/scripts\" ]] && [[ -n \"\$(ls $hpc_work/scripts)\" ]]"
    echo "then"
    echo "  mv -t \"$out_dir/slurm/scripts\" $hpc_work/scripts/* &"
    echo "  pids=(\$!)"
    echo "fi"
    echo "if [[ -d \"$hpc_work/logs\" ]] && [[ -n \"\$(ls $hpc_work/logs)\" ]]"
    echo "then"
    echo "  mv -t \"$out_dir/slurm/logs\" $hpc_work/logs/* &"
    echo "  pids=(\${pids[@]} \$!)"
    echo "fi"
    echo "if [[ -d \"$hpc_work/cases\" ]] && [[ -n \"\$(ls $hpc_work/cases)\" ]]"
    echo "then"
    echo "  for s in $hpc_work/cases/*"
    echo "  do"
    echo "    mv -f -t \"$out_dir\" \$s &"
    echo "    pids=(\${pids[@]} \$!)"
    echo "  done"
    echo "fi"
    echo "echo \"Waiting to copy data... (will be confirmed by 'Finished!')\""
    echo "success=true"
    echo "for p in \${pids[@]};"
    echo "do"
    echo "  wait \$p"
    echo "  if [[ \"\$?\" != 0 ]] ; then success=false; fi"
    echo "done"
    echo "if [[ \$success == true ]]"
    echo "then"
    echo "  echo \"Finished!\""
    if [[ "$delete_hpc_work_dir" == "true" ]]
    then
      echo "  rm -R $hpc_work"
    else
      echo "  rm -R $hpc_work/*"
    fi
    echo "else"
    echo "  echo \"Cleanup finished with errors!\""
    echo "fi"
  } > $clean_cmd_file

  chmod +x $clean_cmd_file
  echo "sbatch --parsable ${clean_slurm_sched[*]}"
  echo "--- sbatch script $clean_cmd_filename ---"
  cat $clean_cmd_file
  echo "--- end of script ---"

  if [[ "$submit_jobs" == "false" ]]
  then
    echo "Not submitting the Cleanup Jobs to slurm (--dry)."
    clean_jobid=CLEAN_JOB_ID
  else
    clean_jobid=$(sbatch --parsable ${clean_slurm_sched[*]})
    echo "Submitted Cleanup Jobs $clean_jobid"
  fi
}

# copy initial state over to work from output directory
function make_copy_job ()
{
  # param1: hpc_work directory
  # param2: output directory
  # param3: subject_list file
  # param4: optional: true/false (submit jobs, default=true)

  local copy_cmd_file
  if [[ "$#" -gt 3 ]] && [[ "$4" == "false" ]]
  then
    copy_cmd_file=$(mktemp)
  else
    copy_cmd_file=$hpc_work/scripts/slurm_copy.sh
  fi
  local copy_cmd_filename=$hpc_work/scripts/slurm_copy.sh

  local hpc_work=$1
  local out_dir=$2
  local subject_list=$3
  local copy_slurm_sched=(-J "FastSurfer-Copyseg-$USER"
    --ntasks=1 --cpus-per-task=4 -o "$out_dir/slurm/logs/copy_%A.log"
    "$copy_cmd_filename")

  if [[ ! -d "$out_dir" ]]
  then
    echo "Trying to copy from $out_dir, but that is not a valid directory!"
    exit 1
  fi
  {
    echo "#!/bin/bash"
    echo "set -e"
    echo "mkdir -p $hpc_work/cases"
    echo "while read subject; do"
    echo "  subject_id=\$(echo \"\$subject\" | cut -d= -f1)"
    echo "  echo \"cp -R -t \\\"$hpc_work/cases/\\\" \\\"$out_dir/\$subject_id\\\" &\""
    echo "  cp -R -t \"$hpc_work/cases/\" \"$out_dir/\$subject_id\" &"
    echo "done < $subject_list"
    echo "echo \"Waiting to copy data... (will be confirmed by 'Finished!')\""
    echo "wait"
    echo "echo \"Finished!\""
  } > $copy_cmd_file

  chmod +x $copy_cmd_file
  echo "sbatch --parsable ${copy_slurm_sched[*]}"
  echo "--- sbatch script $copy_cmd_filename ---"
  cat $copy_cmd_file
  echo "--- end of script ---"

  if [[ "$#" -gt 3 ]] && [[ "$4" == "false" ]]
  then
    echo "Not submitting the Copyseg Jobs to slurm (--dry)."
    copy_jobid=COPY_JOB_ID
  else
    copy_jobid=$(sbatch --parsable ${copy_slurm_sched[*]})
    echo "Submitted Copyseg Jobs $copy_jobid"
  fi
}

function first_non_empty_partition ()
{
  for i in "$@"
  do
    if [[ -n "$i" ]]
    then
      echo "-p $i"
      break
    fi
  done
}
function first_non_empty_arg ()
{
  for i in "$@"
  do
    if [[ -n "$i" ]]
    then
      echo "$i"
      break
    fi
  done
}
function print_status ()
{
  #param1 subject_id
  #param2 command info
  #param3 return value

  local subject_id=$1
  local info=$2
  local retval=$3
  local text
  if [[ "$retval" != "0" ]]
  then
    text="Failed $info with exit code $retval"
  else
    text="Finished $info successfully"
  fi
  echo "$subject_id: $text"
}
function prepend ()
{
  #param1 string to prepend to every line
  # https://serverfault.com/questions/72744/command-to-prepend-string-to-each-line
  while read -r line;
  do
    echo "${1}${line}"
  done
}