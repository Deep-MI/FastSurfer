-------- Running FastSurfer --------

- If you want to run only segmentation:

</path/to/FastSurfer_installation>/run_fastsurfer.sh --seg_only --sd <path/to/output/dir> --sid <subject_id> --t1 <path/to/subjects/t1/image>

- To full run fastsurfer:

</path/to/FastSurfer_installation>/run_fastsurfer.sh --device mps --sd <path/to/output/dir> --sid <subject_id> --t1 <path/to/subjects/t1/image> --fs_license </path/to/freesurfer/license>

- If some files of freesurfer files require bypassing MacOS security, 
- it might take a long time to allow access to every one of them one by one. 
- Instead, this command might be used:  xattr -dr com.apple.quarantine /Applications/freesurfer/*
