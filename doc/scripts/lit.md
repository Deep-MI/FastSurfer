# LIT Integration

FastSurfer does not ship a separate `lit` wrapper script. The Lesion Inpainting Tool (LIT) is
enabled directly from `run_fastsurfer.sh` via the `--lesion_mask` flag:

```bash
./run_fastsurfer.sh \
    --t1 /path/to/T1.nii.gz \
    --lesion_mask /path/to/lesion_mask.nii.gz \
    --sid subject_id \
    --sd /path/to/subjects_dir \
    --fs_license /path/to/license.txt
```

When `--lesion_mask` is present, FastSurfer runs the standalone `lit-inpainting` CLI on the
input image, continues the requested FastSurfer pipeline on the inpainted image, and then runs
`lit-postprocessing` to map the lesion back into the final outputs.

## Outputs

The most important LIT-modified outputs are:

- `mri/inpainted.lit.nii.gz`
- `mri/mask.lit.nii.gz`
- `mri/orig/mask.lit.nii.gz`
- `mri/aparc.DKTatlas+aseg.deep.mgz` with backup `mri/aparc.DKTatlas+aseg.deep.lit.mgz`
- `stats/aseg+DKT.VINN.stats` with backup `stats/aseg+DKT.VINN.lit.stats`
- `stats/aparc.DKTatlas+aseg.lesion_report.txt`
- `stats/aseg.lesion_report.txt`
- `stats/lesion_impact_summary.yaml`

If the surface pipeline is enabled, the public annotation paths
`label/{lh,rh}.aparc.DKTatlas.annot` and `label/{lh,rh}.aparc.DKTatlas.lit.annot`
are provided as symlinks to the mapped annotation files, and the pre-lesion cortical surface
statistics remain available in `stats/{lh,rh}.aparc.DKTatlas.mapped.stats`.

LIT updates the main FastSurfer outputs in place and stores the pre-lesion versions as
`.lit` backups or mapped backup files, depending on the output family.

## Standalone neuroLIT

For standalone inpainting or direct postprocessing outside the FastSurfer pipeline, use the
`neurolit` package and its CLIs:

- `lit-inpainting`
- `lit-download-models`
- `lit-postprocessing`

The standalone tooling and container wrapper live in the separate
[neurolit repository](https://github.com/Deep-MI/neurolit).
