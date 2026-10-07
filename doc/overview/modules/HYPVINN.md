Hypothalamus Sub-Segmentation (HypVINN)
=======================================
The hypothalamus module (`hypothal`): a deep learning network (HypVINN) that segments the hypothalamus into its
subunits together with adjacent structures, and computes summary statistics for them. It can use a T2-weighted image
in addition to the T1-weighted one.

What it computes
----------------
- A sub-segmentation of the hypothalamus and adjacent structures, including the third ventricle, the mammillary
  bodies, the fornix and the optic tracts (`hypothalamus.HypVINN.nii.gz`), and a mask of it.
- Summary statistics for these structures (`hypothalamus.HypVINN.stats`), based on the bias-field corrected T1-weighted
  image.
- With a T2-weighted image: the bias-field corrected T2 image and its registration to the T1 image.
- With `--qc_snap`: a snapshot of the segmentation for visual quality control.

The output files are listed in the [output files overview](../OUTPUT_FILES.md#hypvinn-module).

What it needs
-------------
- A T1-weighted image is highly recommended, see the
  [requirements to input images](../../../README.md#requirements-to-input-images). Voxel sizes down to 0.7 mm are supported
  natively (smaller voxels are experimental).
- Optionally a T2-weighted image of the same subject. FastSurfer registers it to the T1 image with FreeSurfer tools,
  so a T2 image also needs a [FreeSurfer license](../INSTALL.md#freesurfer-license).
- Bias-field corrected images, which the segmentation computes for you. With `--no_biasfield`, the module expects
  images that were corrected beforehand.

The module adds a few minutes on a GPU to the segmentation.

Options
-------
The module runs by default, as part of the segmentation. The options of `run_fastsurfer.sh` for it:

- `--no_hypothal`: skip this module.
- `--t2 <t2_path>`: also use a T2-weighted image.
- `--reg_mode <coreg|robust|none>`: how the T2 image is registered to the T1 image: `coreg` (default, FreeSurfer's
  `mri_coreg`), `robust` (`mri_robust_register`), or `none` if both images are already co-registered.
- `--qc_snap`: create quality control snapshots in `qc_snapshots`.

Usage:

```text
run_fastsurfer.sh --sd <subjects_dir> --sid <subject_id> --t1 <t1_path> \
    --t2 <t2_path> --fs_license <freesurfer_license_path>
```

All options are described in the [run_fastsurfer.sh reference](../../scripts/RUN_FASTSURFER.md). To run the network
script directly, for example on an existing FastSurfer subject that was processed without this module, see
[HypVINN in the command reference](../../scripts/hypvinn.rst).

References
----------
If you use the hypothalamus sub-segmentation in your research, please cite:

- Estrada S, Kuegler D, Bahrami E, Xu P, Mousa D, Breteler MMB, Aziz NA, Reuter M.
  **FastSurfer-HypVINN: Automated sub-segmentation of the hypothalamus and adjacent structures on high-resolutional brain MRI.**
  *Imaging Neuroscience* 1 (2023), 1–32.
  [doi:10.1162/imag_a_00034](https://doi.org/10.1162/imag_a_00034)
