# Lesion Inpainting Tool (LIT)

The **Lesion Inpainting Tool (LIT)** is a deep learning-based tool designed to inpaint lesions (such as tumors, cavities, or abnormalities) in T1-weighted MRI images. This allows downstream analysis tools like FastSurfer to perform more accurate whole-brain segmentation and cortical surface reconstruction in cases with significant structural alterations.

## Key Features

- **Shape-Independent Inpainting**: Effectively inpaints lesions regardless of their shape or appearance.
- **FastSurfer Integration**: Can be run as part of the main FastSurfer pipeline using the `--lesion_mask` flag.
- **Standalone Mode**: Can be used independently to create inpainted images for other tools.
- **Postprocessing Tools**: Includes scripts to map lesion masks back into FastSurfer/FreeSurfer outputs and generate anatomy reports.

## Usage in FastSurfer

To use LIT within FastSurfer, simply provide a binary lesion mask using the `--lesion_mask` flag:

```bash
./run_fastsurfer.sh --t1 /path/to/T1.nii.gz \
                    --lesion_mask /path/to/lesion_mask.nii.gz \
                    --sid subject_id --sd /path/to/output_dir \
                    --fs_license /path/to/license.txt
```

When this flag is provided, FastSurfer will:
1. Inpaint the lesion region using the LIT model.
2. Run the segmentation and surface pipelines on the inpainted image.
3. Automatically map the lesion mask into the final segmentation and surface outputs.

## Postprocessing and Anatomy Reports

LIT provides comprehensive postprocessing tools to integrate lesions into FastSurfer outputs. This includes:
- **Lesion Mapping**: Mapping the lesion mask into the relevant FastSurfer segmentations and stats files.
- **Anatomy Reports**: Identifying structures that are replaced, reduced, or adjacent to the lesion.
- **Surface Masking**: Projecting the lesion onto cortical surfaces.

The current integration updates the main FastSurfer output files in place and preserves the pre-lesion versions either as `.lit` backups or, for some surface-derived files, in the original `.mapped.*` files. The most important lesion-specific outputs are:

- `mri/inpainted.lit.nii.gz`, `mri/mask.lit.nii.gz`, and `mri/orig/mask.lit.nii.gz` as the inpainting inputs stored in the subject directory
- `mri/aparc.DKTatlas+aseg.deep.mgz` with backup `mri/aparc.DKTatlas+aseg.deep.lit.mgz`
- `stats/aseg+DKT.VINN.stats` with backup `stats/aseg+DKT.VINN.lit.stats`
- `stats/aparc.DKTatlas+aseg.lesion_report.txt`
- `stats/aseg.lesion_report.txt`
- `stats/lesion_impact_summary.yaml`

If the surface pipeline is enabled, FastSurfer also exposes lesion-aware cortical annotations at
`label/{lh,rh}.aparc.DKTatlas.annot` and the corresponding pre-lesion annotations at
`label/{lh,rh}.aparc.DKTatlas.lit.annot`, both as symlinks to the mapped annotation files. The
pre-lesion cortical surface statistics remain available in `stats/{lh,rh}.aparc.DKTatlas.mapped.stats`.

For more details on postprocessing, refer to the [neurolit repository](https://github.com/Deep-MI/neurolit).

## References

If you use LIT in your research, please cite:

*Pollak C, Kuegler D, Bauer T, Rueber T, Reuter M, FastSurfer-LIT: Lesion Inpainting Tool for Whole Brain MRI Segmentation with Tumors, Cavities and Abnormalities, Imaging Neuroscience 2025. https://doi.org/10.1162/imag_a_00446*
