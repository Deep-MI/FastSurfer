Running FastSurfer as MacOS package
===================================

[Installation Instructions](INSTALL.md#macos)

If you want to run only segmentation (replace placeholders starting with "<" and ending with ">", see https://deep-mi.org/FastSurfer/stable):

```bash
run_fastsurfer.sh --seg_only --sd <path/to/output/dir> --sid <subject_id> --t1 <path/to/subjects/t1/image>
```
To run the full FastSurfer:

```bash
run_fastsurfer.sh --device mps --sd <path/to/output/dir> --sid <subject_id> --t1 <path/to/subjects/t1/image> --fs_license </path/to/freesurfer/license>
```
