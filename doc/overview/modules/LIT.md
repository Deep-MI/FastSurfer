# Lesion Inpainting Tool (LIT)

With the **Lesion Inpainting Tool (LIT)** extension, FastSurfer is able to process T1-weighted
images with lesions, such as tumors, cavities, or abnormalities. Since the deep learning-based
tool LIT is designed to paint healthy-looking tissue into those lesions, downstream analysis with
FastSurfer can produce more reliable whole-brain segmentation and cortical surface reconstruction
in cases with significant structural alterations.

> **Note:** The FastSurfer LIT extension is currently experimental. Review the LIT-modified
> outputs before using them for downstream analyses.

## FastSurfer Usage

FastSurfer runs LIT when a lesion mask is passed with `--lesion_mask <path to file>`:

```bash
./run_fastsurfer.sh --t1 /path/to/T1.nii.gz \
                    --lesion_mask /path/to/lesion_mask.nii.gz \
                    --sid subject_id --sd /path/to/output_dir \
                    --fs_license /path/to/license.txt
```

Lesion inpainting is not compatible with separate processing of the segmentation and surfaces with `--seg_only` and `--surf_only`. If you want to run the surface pipeline, avoid `--seg_only`!

With `--lesion_mask <path to file>`, FastSurfer:

1. inpaints the lesion area in the input T1w image,
2. runs the requested FastSurfer segmentation and surface pipeline on the inpainted image, and
3. maps the lesion into the final FastSurfer outputs and regenerates affected statistics.

## Lesion Mask Requirements

The lesion annotation must match the MRI volume passed as `--t1`: it should use the same voxel
grid and vox2ras/affine, and it should be an integer or binary-compatible numeric image. Non-zero
voxels are treated as lesion. The mask defines which voxels are inpainted and later marked in the
FastSurfer outputs.

Underestimation of lesion areas affects the segmentation more than oversegmentation of lesion masks,
hence we recommend generous annotation of all damaged tissue and potentially dilating the mask.

## Output Behavior

FastSurfer with LIT updates the standard subject directory instead of writing a separate LIT output
tree. The primary FastSurfer files are lesion-integrated, while pre-lesion versions are preserved
as `.lit` backups or, for selected surface-derived files, as mapped backup files.

`lesion_impact_summary.yaml` is currently emitted as YAML by neurolit. The accompanying text
reports provide a human-readable summary of the affected anatomical structures.
The complete list of LIT-related output files is documented in the
[FastSurfer output files overview](../OUTPUT_FILES.md#lesion-inpainting-tool-lit-optional).

## References

If you use LIT in your research, please cite:

*Pollak C, Kuegler D, Bauer T, Rueber T, Reuter M, FastSurfer-LIT: Lesion Inpainting Tool for Whole Brain MRI Segmentation with Tumors, Cavities and Abnormalities, Imaging Neuroscience 2025. https://doi.org/10.1162/imag_a_00446*
