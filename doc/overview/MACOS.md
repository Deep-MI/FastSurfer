Running FastSurfer as MacOS package
===================================

[Installation Instructions](INSTALL.md#macos)

If you want to run only segmentation (replace placeholders starting with "<" and ending with ">", see https://deep-mi.org/FastSurfer/stable):

```bash
run_fastsurfer.sh --seg_only --sd <path/to/output/dir> --sid <subject_id> --t1 <path/to/subjects/t1/image>
```
To full run fastsurfer:

```bash
run_fastsurfer.sh --device mps --sd <path/to/output/dir> --sid <subject_id> --t1 <path/to/subjects/t1/image> --fs_license </path/to/freesurfer/license>
```

Some files of **FreeSurfer** binaries require bypassing MacOS security, which is
significantly easier to do with the following command than manually and one by one. 

```bash
xattr -dr com.apple.quarantine /Applications/freesurfer/*
```
