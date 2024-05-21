#!/bin/bash

# Functions

check_input_dirs() {
    local dirs=("$@")
    for dir in "${dirs[@]}"; do
        if [[ ! -d "$dir" ]]; then
            echo "Error: Directory $dir does not exist."
            exit 1
        fi
    done
}

check_installation() {
    if command -v singulardity &> /dev/null; then
        echo "singularity"
    elif command -v docker &> /dev/null; then
        echo "docker"
    else
        echo "none"
    fi
}

set_defaults() {
    : "${FREESURFER_HOME:=/path/to/freesurfer}"
    : "${FS_LICENSE:=${FREESURFER_HOME}/license.txt}"
    
    if [[ -z "$subject_dir" ]]; then
        subject_dir=$(find /path/to/images -name "orig.mgz" -print -quit)
    fi
}

output_message() {
    echo "Running FastSurfer with the following settings:"
    echo "Subject Directory: $subject_dir"
    echo "Output Directory: $output_dir"
    echo "Using License: $FS_LICENSE"
    echo "Home Directory: $home_dir"
}

check_output_folder() {
    if [[ -d "$output_dir" ]]; then
        echo "####Ouptu dir is present $output_dir "
        if ls "$output_dir"/subjectX/stats/*.stats 1> /dev/null 2>&1; then
            echo "Warning: Output directory already contains FastSurfer files. Overwriting..."
        fi
    fi
}

run_in_background() {
    local log_file="$output_dir/fastsurfer.log"
    echo inside run_in_background
    echo $@
    eval $@ > "$log_file" 2>&1 &
    echo "FastSurfer is running in the background. Logs are being written to $log_file."
    echo "To cancel FastSurfer, use 'kill $!'"
}

# Argument parsing
background_mode=false
subject_dir=""
output_dir=""
t1_image=""
FS_LICENSE=""
home_dir=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -b|--background) background_mode=true ;;
        -sd|--subject_dir) subject_dir="$2"; shift ;;
        -o|--output_dir) output_dir="$2"; shift ;;
        -t1) t1_image="$2"; shift ;;
        --fs_license) FS_LICENSE="$2"; shift ;;
        -hd|--home_dir) home_dir="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Check for required input directories
input_dirs=("$subject_dir" "$output_dir" "$FS_LICENSE" "$home_dir")
check_input_dirs "${input_dirs[@]}"


# Set default values for variables
set_defaults

# Output the settings to the user
output_message

# Check output folder
check_output_folder

# Determine if Singularity or Docker is available
container_tool=$(check_installation)

# Command to run FastSurfer
fastsurfer_cmd=""
if [[ "$container_tool" == "singulardity" ]]; then
    echo "singularity branch"
    fastsurfer_cmd="singularity exec --nv --no-home -B \"$subject_dir\":/data -B \"$FS_LICENSE\":/fs -B \"$output_dir\":/output $home_dir/fastsurfer-gpu.sif fastsurfer/run_fastsurfer.sh --fs_license /fs/license.txt --t1 /data/subjectX/orig.mgz --sid subjectX --sd /output --seg_only --parallel --3T"
elif [[ "$container_tool" == "docker" ]]; then
    echo "docker branch"
    fastsurfer_cmd="docker run --gpus all -v \"$subject_dir\":/data -v \"$FS_LICENSE\":/fs -v \"$output_dir\":/output --rm --user $(id -u):$(id -g) deepmi/fastsurfer:latest  --fs_license /fs/license.txt --t1 /data/subjectX/orig.mgz --sid subjectX --sd /output --seg_only --parallel --3T"
    echo $fastsurfer_cmd
else
    echo "Error: Neither Singularity nor Docker is installed."
    exit 1
fi

# Run FastSurfer, either normally or in the background
if $background_mode; then
    run_in_background $fastsurfer_cmd
else
    eval $fastsurfer_cmd
fi
